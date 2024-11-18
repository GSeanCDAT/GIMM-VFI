# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------

import logging

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from utils.accumulator import AccmStageINR
from .trainer import TrainerTemplate

logger = logging.getLogger(__name__)
from utils.loss import LapLoss, Ternary, Charbonnier_L1
from utils.lpips import LPIPS


class xytSubSampler:
    def __init__(self, subsamper_config):
        self.config = subsamper_config
        if self.config.type is not None and self.config.ratio == 1.0:
            self.config.type = None

    def subsample_coords_idx(self, xs, upsample_ratio=1):
        if self.config.type is None:
            subcoord_idx = None
        elif self.config.type == "random":
            subcoord_idx = self.subsample_random_idx(
                xs, ratio=self.config.ratio, upsample_ratio=upsample_ratio
            )
        else:
            raise NotImplementedError
        return subcoord_idx

    def subsample_random_idx(self, xs, ratio=None, upsample_ratio=1):
        batch_size = xs.shape[0]
        spatial_dims = list(xs.shape[3:])

        subcoord_idx = []
        num_spatial_dims = int(np.prod(spatial_dims) * upsample_ratio**2)
        num_subcoord = int(num_spatial_dims * ratio)
        for _ in range(batch_size):
            rand_idx = torch.randperm(num_spatial_dims, device=xs.device)
            rand_idx = rand_idx[:num_subcoord]
            subcoord_idx.append(rand_idx.unsqueeze(0))
        return torch.cat(subcoord_idx, dim=0)

    @staticmethod
    def subsample_xs(xs, subcoord_idx=None):
        if subcoord_idx is None:
            return xs

        batch_size = xs.shape[0]
        permute_idx_range = [
            i for i in range(2, xs.ndim)
        ]  # note: xs is originally channel-fist data format
        xs = xs.permute(0, *permute_idx_range, 1)  # convert xs into channel last type

        xs = xs.reshape(batch_size, -1, xs.shape[-1])
        sub_xs = []
        for idx in range(batch_size):
            sub_xs.append(xs[idx : idx + 1, subcoord_idx[idx]])
        sub_xs = torch.cat(sub_xs, dim=0)
        return sub_xs


class Trainer(TrainerTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coord_subsampler = xytSubSampler(self.config.loss.subsample)
        self.using_lpips = self.config.loss.perceptual_loss
        self.lap = LapLoss()
        self.census = Ternary()
        self.l1 = Charbonnier_L1()
        self.lpips = LPIPS(net="alex", version="0.1").eval()
        for name, param in self.lpips.named_parameters():
            param.requires_grad = False

    def get_accm(self):
        accm = AccmStageINR(
            scalar_metric_names=(
                "loss_total",
                "lap",
                "census",
                "l1",
                "rec",
                "lpips",
                "psnr",
            ),
            device=self.device,
        )
        return accm

    @torch.no_grad()
    def eval(self, valid=True, ema=False, verbose=False, epoch=0):
        model = self.model_ema if ema else self.model
        loader = self.loader_val if valid else self.loader_trn
        n_inst = len(self.dataset_val) if valid else len(self.dataset_trn)

        accm = self.get_accm()

        if self.distenv.master:
            pbar = tqdm(enumerate(loader), total=len(loader))
        else:
            pbar = enumerate(loader)

        model.eval()
        for it, xs in pbar:
            model.zero_grad()
            flows_gt = xs["flows"] if "flows" in xs.keys() else None
            timesteps = (
                [xs["t"].to(self.device)]
                if "t" in xs.keys()
                else [
                    0.5 * torch.ones(flows_gt.shape[0]).to(self.device).to(torch.float)
                ]
            )

            xs = xs["xs"]
            xs = xs.to(self.device)  # [B, C, T, H, W]

            img_xs, gt = xs[:, :, :2], xs[:, :, 2]
            batch_size = img_xs.shape[0]
            s_shape = img_xs.shape[-2:]
            coord_inputs = [
                (
                    model.module.sample_coord_input(
                        batch_size, s_shape, timesteps[0], device=img_xs.device
                    ),
                    None,
                )
            ]
            all_outputs = model(img_xs, coord_inputs, t=timesteps)
            targets = gt.detach()

            ######Loss Calculation######
            cur_count = targets.shape[0]
            ## i. image loss
            loss_lap = (self.lap(all_outputs["imgt_pred"][0], targets)).mean()
            loss_census = self.census(all_outputs["imgt_pred"][0], targets)
            loss_l1 = self.l1(all_outputs["imgt_pred"][0], targets)
            loss_lpips = self.lpips(
                all_outputs["imgt_pred"][0], targets, normalize=True
            ).mean()
            ## ii. flow loss
            psnr = model.module.compute_psnr(
                all_outputs["imgt_pred"][0], targets, reduction="sum"
            )
            metrics = dict(
                lap=loss_lap * cur_count,
                census=loss_census * cur_count,
                l1=loss_l1 * cur_count,
                psnr=psnr,
                lpips=loss_lpips,
            )

            accm.update(metrics, count=cur_count, sync=True, distenv=self.distenv)

            if self.distenv.master:
                line = accm.get_summary().print_line()
                pbar.set_description(line)
        line = accm.get_summary(n_inst).print_line()

        if self.distenv.master and verbose:
            mode = "valid" if valid else "train"
            mode = "%s_ema" % mode if ema else mode
            logger.info(f"""{mode:10s}, """ + line)
            self.reconstruct(xs, epoch=0, mode=mode)

        summary = accm.get_summary(n_inst)
        summary["xs"] = xs
        summary["t"] = timesteps[-1]
        return summary

    def train(self, optimizer=None, scheduler=None, scaler=None, epoch=0):
        model = self.model
        model_ema = self.model_ema
        total_step = len(self.loader_trn) * epoch

        accm = self.get_accm()

        if self.distenv.master:
            pbar = tqdm(enumerate(self.loader_trn), total=len(self.loader_trn))
        else:
            pbar = enumerate(self.loader_trn)

        self.lpips.to(self.device)
        model.train()
        for it, xs in pbar:
            timesteps = (
                xs["t"].to(self.device, non_blocking=True)
                if "t" in xs.keys()
                else 0.5
                * torch.ones(xs["xs"].shape[0])
                .to(self.device, non_blocking=True)
                .to(torch.float)
            )
            xs = xs["xs"]

            model.zero_grad(set_to_none=True)
            xs = xs.to(self.device, non_blocking=True)
            # NOTE: img_xs.shape b,c,2,h,w | gt.shape b,c,h,w
            img_xs, gt = xs[:, :, :2], xs[:, :, 2]
            batch_size = img_xs.shape[0]
            s_shape = img_xs.shape[-2:]

            timesteps = [
                torch.zeros_like(timesteps),
                torch.ones_like(timesteps),
                timesteps,
            ]

            subsample_idx0 = self.coord_subsampler.subsample_coords_idx(
                img_xs[:, :, 0:1], upsample_ratio=1
            )
            subsample_idx1 = self.coord_subsampler.subsample_coords_idx(
                img_xs[:, :, 0:1], upsample_ratio=1
            )

            coord_inputs = [
                (
                    model.module.sample_coord_input(
                        batch_size, s_shape, timesteps[0], device=xs.device
                    ),
                    subsample_idx0,
                ),
                (
                    model.module.sample_coord_input(
                        batch_size, s_shape, timesteps[1], device=xs.device
                    ),
                    subsample_idx1,
                ),
                (
                    model.module.sample_coord_input(
                        batch_size, s_shape, timesteps[2], device=xs.device
                    ),
                    None,
                ),
            ]

            all_outputs = model(img_xs, coord_inputs, t=timesteps)

            targets = [gt.detach()]
            mid_id = 0
            assert len(all_outputs["imgt_pred"]) == 1
            psnr = model.module.compute_psnr(
                all_outputs["imgt_pred"][mid_id], targets[mid_id]
            )

            ######Loss Calculation######
            ## i. image loss
            loss_lap = 0
            loss_census = 0
            loss_l1 = 0
            loss_lpips = 0

            if all_outputs["other_pred"][0][0] is not None:
                for i in range(len(all_outputs["other_pred"][0])):
                    loss_lap = (
                        loss_lap
                        + 0.5
                        * (
                            self.lap(all_outputs["other_pred"][0][i], targets[mid_id])
                        ).mean()
                    )
                    loss_census = loss_census + 0.5 * self.census(
                        all_outputs["other_pred"][0][i], targets[mid_id]
                    )
                    loss_l1 = loss_l1 + 0.5 * self.l1(
                        all_outputs["other_pred"][0][i], targets[mid_id]
                    )

            loss_lap = (
                loss_lap
                + (self.lap(all_outputs["imgt_pred"][0], targets[mid_id])).mean()
            )
            loss_census = loss_census + self.census(
                all_outputs["imgt_pred"][0], targets[mid_id]
            )
            loss_l1 = loss_l1 + self.l1(all_outputs["imgt_pred"][0], targets[mid_id])
            if self.using_lpips:
                loss_lpips = (
                    loss_lpips
                    + 0.5
                    * self.lpips(
                        all_outputs["other_pred"][0][i], targets[mid_id], normalize=True
                    ).mean()
                )

            ## ii. flow loss
            loss_rec_flow = 0
            if all_outputs["ninrflow"][0] is not None:
                loss_rec_flow = 0.5 * F.mse_loss(
                    all_outputs["ninrflow"][0],
                    self.coord_subsampler.subsample_xs(
                        all_outputs["nflow"][:, :, 0:1], subsample_idx0
                    ).detach(),
                ) + 0.5 * F.mse_loss(
                    all_outputs["ninrflow"][1],
                    self.coord_subsampler.subsample_xs(
                        all_outputs["nflow"][:, :, 1:2], subsample_idx1
                    ).detach(),
                )

            loss = (
                loss_census
                + loss_l1
                + self.config.arch.rec_weight * loss_rec_flow
                + loss_lap
                + loss_lpips
            )

            loss.backward()
            if self.config.optimizer.max_gn is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.optimizer.max_gn
                )
            optimizer.step()
            scheduler.step()
            if model_ema:
                model_ema.module.update(model.module, total_step)

            metrics = dict(
                loss_total=loss,
                lap=loss_lap,
                rec=loss_rec_flow,
                census=loss_census,
                l1=loss_l1,
                lpips=loss_lpips,
                psnr=psnr,
            )
            accm.update(metrics, count=1)
            total_step += 1
            if self.distenv.master:
                line = f"""(epoch {epoch} / iter {it}) """
                line += accm.get_summary().print_line()
                line += f""", lr: {scheduler.get_last_lr()[0]:e}"""
                pbar.set_description(line)

        summary = accm.get_summary()
        summary["xs"] = xs
        summary["t"] = timesteps[-1]
        return summary

    def logging(self, summary, scheduler=None, epoch=0, mode="train"):
        if epoch % 10 == 1 or epoch % self.config.experiment.test_imlog_freq == 0:
            self.reconstruct(summary, upsample_ratio=1, epoch=epoch, mode=mode)
            # self.reconstruct(summary, upsample_ratio=3, epoch=epoch, mode=mode)

        self.writer.add_scalar("loss/lap", summary["lap"], mode, epoch)
        self.writer.add_scalar("loss/census", summary["census"], mode, epoch)
        self.writer.add_scalar("loss/l1", summary["l1"], mode, epoch)
        self.writer.add_scalar("loss/rec", summary["rec"], mode, epoch)
        self.writer.add_scalar("loss/psnr", summary["psnr"], mode, epoch)
        self.writer.add_scalar("loss/lpips", summary["lpips"], mode, epoch)

        if mode == "train":
            self.writer.add_scalar("lr", scheduler.get_last_lr()[0], mode, epoch)

        line = f"""ep:{epoch}, {mode:10s}, """
        line += summary.print_line()
        line += f""", """
        if scheduler:
            line += f"""lr: {scheduler.get_last_lr()[0]:e}"""

        logger.info(line)

    @torch.no_grad()
    def reconstruct(self, summary, upsample_ratio=1, epoch=0, mode="valid"):
        xs = summary["xs"]
        timesteps = [summary["t"][:8]]

        def get_recon_imgs(xs_real, xs_recon, upsample_ratio=1):
            xs_real = xs_real
            if not upsample_ratio == 1:
                xs_real = torch.nn.functional.interpolate(
                    xs_real, scale_factor=upsample_ratio
                )
            xs_recon = xs_recon
            xs_recon = torch.clamp(xs_recon, 0, 1)
            return xs_real, xs_recon

        model = self.model_ema if "ema" in mode else self.model
        model.eval()

        xs_real = xs[:8]

        img_xs, xs_real = xs_real[:, :, :2], xs_real[:, :, 2]
        batch_size = img_xs.shape[0]
        s_shape = img_xs.shape[-2:]
        coord_inputs = [
            (
                model.module.sample_coord_input(
                    batch_size, s_shape, timesteps[0], device=img_xs.device
                ),
                None,
            )
        ]

        xs_pred = model(img_xs, coord_inputs, t=timesteps)["imgt_pred"][0]

        xs_real, xs_recon = get_recon_imgs(xs_real, xs_pred, upsample_ratio)
        grid = torch.cat([xs_real, xs_recon], dim=0)
        grid = torchvision.utils.make_grid(grid, nrow=xs_real.shape[0])
        self.writer.add_image(f"reconstruction_x{upsample_ratio}", grid, mode, epoch)
