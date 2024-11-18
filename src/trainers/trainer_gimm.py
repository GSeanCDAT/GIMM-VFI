# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------

import logging

import torch
import torchvision
from tqdm import tqdm

from utils.accumulator import AccmStageINR
from .trainer import TrainerTemplate

logger = logging.getLogger(__name__)
from utils.flow_viz import flow_to_image

import random


class Trainer(TrainerTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_accm(self):
        accm = AccmStageINR(
            scalar_metric_names=("loss_total", "mse", "psnr"),
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
            xs, flow_scaler, ori_flow = xs["xs"], xs["flow_scaler"], xs["ori_flows"]
            flow_scaler = flow_scaler.to(self.device, non_blocking=True).reshape(-1, 1)
            xs = xs.to(self.device)  # [B, C, *]
            ori_flow = ori_flow.to(self.device)

            batch_size = xs.shape[0]
            s_shape = xs.shape[-2:]
            t_id = 1
            timesteps = (
                0.5
                * t_id
                * torch.ones(batch_size)
                .to(self.device, non_blocking=True)
                .to(torch.float)
            )
            coord_inputs = model.module.sample_coord_input(
                batch_size, s_shape, timesteps, device=xs.device
            )

            assert xs.shape[2] == 3
            input_xs = torch.cat((xs[:, :, :1], xs[:, :, 2:]), dim=2)
            assert input_xs.shape[2] == 2
            outputs = model(
                input_xs, coord=coord_inputs, ori_flow=ori_flow, timesteps=timesteps
            )
            targets = xs.detach()[:, :, t_id : t_id + 1]
            loss = model.module.compute_loss(outputs, targets, reduction="sum")

            metrics = dict(
                loss_total=loss["loss_total"], mse=loss["mse"], psnr=loss["psnr"]
            )
            accm.update(metrics, count=xs.shape[0], sync=True, distenv=self.distenv)

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
        summary["flow_scaler"] = flow_scaler
        summary["ori_flows"] = ori_flow
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

        model.train()
        for it, xs in pbar:
            xs, flow_scaler, ori_flow = xs["xs"], xs["flow_scaler"], xs["ori_flows"]
            flow_scaler = flow_scaler.to(self.device, non_blocking=True).reshape(-1, 1)
            model.zero_grad(set_to_none=True)
            xs = xs.to(self.device, non_blocking=True)
            ori_flow = ori_flow.to(self.device)

            batch_size = xs.shape[0]
            s_shape = xs.shape[-2:]
            t_id = random.randint(0, 2)
            timesteps = (
                0.5
                * t_id
                * torch.ones(batch_size)
                .to(self.device, non_blocking=True)
                .to(torch.float)
            )

            coord_inputs = model.module.sample_coord_input(
                batch_size, s_shape, timesteps, device=xs.device
            )

            keep_xs_shape = True
            assert xs.shape[2] == 3
            input_xs = torch.cat((xs[:, :, :1], xs[:, :, 2:]), dim=2)
            assert input_xs.shape[2] == 2
            outputs = model(
                input_xs,
                coord=coord_inputs,
                keep_xs_shape=keep_xs_shape,
                ori_flow=ori_flow,
                timesteps=timesteps,
            )

            targets = xs.detach()[:, :, t_id : t_id + 1]
            loss = model.module.compute_loss(outputs, targets)
            loss["loss_total"].backward()
            if self.config.optimizer.max_gn is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.optimizer.max_gn
                )
            optimizer.step()
            scheduler.step()

            if model_ema:
                model_ema.module.update(model.module, total_step)

            metrics = dict(
                loss_total=loss["loss_total"], mse=loss["mse"], psnr=loss["psnr"]
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
        summary["flow_scaler"] = flow_scaler
        summary["ori_flows"] = ori_flow
        return summary

    def logging(self, summary, scheduler=None, epoch=0, mode="train"):
        if epoch % 10 == 1 or epoch % self.config.experiment.test_imlog_freq == 0:
            self.reconstruct(summary, upsample_ratio=1, epoch=epoch, mode=mode)
            # self.reconstruct(summary, upsample_ratio=3, epoch=epoch, mode=mode)

        self.writer.add_scalar("loss/loss_total", summary["loss_total"], mode, epoch)
        self.writer.add_scalar("loss/mse", summary["mse"], mode, epoch)
        self.writer.add_scalar("loss/psnr", summary["psnr"], mode, epoch)

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
        flow_scaler = summary["flow_scaler"][:8]
        ori_flow = summary["ori_flows"][:8]
        xs = summary["xs"]

        def get_recon_imgs(xs_real, xs_recon, upsample_ratio=1):
            if xs_real.shape[1] == 2:
                real = []
                pred = []
                for i in range(xs_real.shape[0]):
                    for j in range(xs_real.shape[2]):
                        real.append(
                            torch.Tensor(
                                flow_to_image(
                                    xs_real[i, :, j]
                                    .detach()
                                    .permute(1, 2, 0)
                                    .cpu()
                                    .numpy()
                                )
                                / 255.0
                            )
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                        )
                        pred.append(
                            torch.Tensor(
                                flow_to_image(
                                    xs_recon[i, :, j]
                                    .detach()
                                    .permute(1, 2, 0)
                                    .cpu()
                                    .numpy()
                                )
                                / 255.0
                            )
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                        )
                xs_real, xs_recon = torch.cat(real, dim=0), torch.cat(pred, dim=0)
            xs_real = xs_real
            if not upsample_ratio == 1:
                xs_real = torch.nn.functional.interpolate(
                    xs_real, scale_factor=upsample_ratio
                )
            xs_recon = xs_recon
            xs_recon = torch.clamp(xs_recon, 0, 1)
            return xs_real, xs_recon

        model = self.model
        model.eval()

        assert upsample_ratio > 0

        xs_real = xs[:8]

        batch_size = xs_real.shape[0]
        s_shape = xs_real.shape[-2:]
        t_id = 1
        timesteps = (
            0.5
            * t_id
            * torch.ones(batch_size).to(self.device, non_blocking=True).to(torch.float)
        )

        coord_inputs = model.module.sample_coord_input(
            batch_size, s_shape, timesteps, device=xs.device
        )

        input_xs = torch.cat((xs_real[:, :, :1], xs_real[:, :, 2:]), dim=2)
        xs_recon = model(
            input_xs, coord=coord_inputs, ori_flow=ori_flow, timesteps=timesteps
        )

        xs_real = (xs_real[:, :, 1:2] * 2.0 - 1.0) * flow_scaler.unsqueeze(
            -1
        ).unsqueeze(-1).unsqueeze(-1).to(xs_real.device)
        xs_recon = (xs_recon * 2.0 - 1.0) * flow_scaler.unsqueeze(-1).unsqueeze(
            -1
        ).unsqueeze(-1).to(xs_recon.device)
        num_t = xs_real.shape[2]
        xs_real, xs_recon = get_recon_imgs(xs_real, xs_recon, upsample_ratio)
        grid = torch.cat([xs_real, xs_recon], dim=0)
        grid = torchvision.utils.make_grid(grid, nrow=8 * num_t)
        self.writer.add_image(f"reconstruction_x{upsample_ratio}", grid, mode, epoch)
