# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# amt: https://github.com/MCG-NKU/AMT
# motif: https://github.com/sichun233746/MoTIF
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import GIMMVFIConfig
from .modules.coord_sampler import CoordSampler3D
from .modules.hyponet import HypoNet
from .modules.fi_components import *
from .flowformer import initialize_Flowformer
from .modules.fi_utils import normalize_flow, unnormalize_flow, warp, resize
from .raft.corr import BidirCorrBlock
from .modules.softsplat import softsplat


class GIMMVFI_F(nn.Module):
    Config = GIMMVFIConfig

    def __init__(self, config: GIMMVFIConfig):
        super().__init__()
        self.config = config = config.copy()
        self.hyponet_config = config.hyponet
        self.raft_iter = config.raft_iter

        ######### Encoder and Decoder Settings #########
        self.flow_estimator = initialize_Flowformer()
        f_dims = [256, 128]

        skip_channels = f_dims[-1] // 2
        self.num_flows = 3
        self.amt_init_decoder = NewInitDecoder(f_dims[0], skip_channels)
        self.amt_final_decoder = NewMultiFlowDecoder(f_dims[1], skip_channels)

        self.amt_update4_low = self._get_updateblock(f_dims[0] // 2, 2.0)
        self.amt_update4_high = self._get_updateblock(f_dims[0] // 2, None)

        self.amt_comb_block = nn.Sequential(
            nn.Conv2d(3 * self.num_flows, 6 * self.num_flows, 7, 1, 3),
            nn.PReLU(6 * self.num_flows),
            nn.Conv2d(6 * self.num_flows, 3, 7, 1, 3),
        )

        ################ GIMM settings #################
        self.coord_sampler = CoordSampler3D(config.coord_range)

        self.g_filter = torch.nn.Parameter(
            torch.FloatTensor(
                [
                    [1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0],
                    [1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0],
                    [1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0],
                ]
            ).reshape(1, 1, 1, 3, 3),
            requires_grad=False,
        )
        self.fwarp_type = config.fwarp_type

        self.alpha_v = torch.nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        self.alpha_fe = torch.nn.Parameter(torch.FloatTensor([1]), requires_grad=True)

        channel = 32
        in_dim = 2
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_dim, channel // 2, 3, 1, 1, bias=True, groups=1),
            nn.Conv2d(channel // 2, channel, 3, 1, 1, bias=True, groups=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            LateralBlock(channel),
            LateralBlock(channel),
            LateralBlock(channel),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(
                channel, channel // 2, 3, 1, 1, padding_mode="reflect", bias=True
            ),
        )
        channel = 64
        in_dim = 64
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_dim, channel // 2, 3, 1, 1, bias=True, groups=1),
            nn.Conv2d(channel // 2, channel, 3, 1, 1, bias=True, groups=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            LateralBlock(channel),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(
                channel, channel // 2, 3, 1, 1, padding_mode="reflect", bias=True
            ),
        )

        self.hyponet = HypoNet(config.hyponet, add_coord_dim=32)

    def _get_updateblock(self, cdim, scale_factor=None):
        return BasicUpdateBlock(
            cdim=cdim,
            hidden_dim=192,
            flow_dim=64,
            corr_dim=256,
            corr_dim2=192,
            fc_dim=188,
            scale_factor=scale_factor,
            corr_levels=4,
            radius=4,
        )

    def cal_bidirection_flow(self, im0, im1):
        f01, features0, fnet0 = self.flow_estimator(
            im0, im1, return_feat=True, iters=None
        )
        f10, features1, fnet1 = self.flow_estimator(
            im1, im0, return_feat=True, iters=None
        )
        f01 = f01[0]
        f10 = f10[0]
        corr_fn = BidirCorrBlock(fnet0, fnet1, radius=4)
        flow01 = f01.unsqueeze(2)
        flow10 = f10.unsqueeze(2)
        noraml_flows = torch.cat([flow01, -flow10], dim=2)
        noraml_flows, flow_scalers = normalize_flow(noraml_flows)

        ori_flows = torch.cat([flow01, flow10], dim=2)
        return (
            noraml_flows,
            ori_flows,
            flow_scalers,
            features0,
            features1,
            corr_fn,
            torch.cat([f01.unsqueeze(2), f10.unsqueeze(2)], dim=2),
        )

    def predict_flow(self, f, coord, t, flows):
        raft_flow01 = flows[:, :, 0].detach()
        raft_flow10 = flows[:, :, 1].detach()

        # calculate splatting metrics
        weights1, weights2 = self.cal_splatting_weights(raft_flow01, raft_flow10)
        strtype = self.fwarp_type + "-zeroeps"

        # b,c,h,w
        pixel_latent_0 = self.cnn_encoder(f[:, :, 0])
        pixel_latent_1 = self.cnn_encoder(f[:, :, 1])
        pixel_latent = []

        for i, cur_t in enumerate(t):
            cur_t = cur_t.reshape(-1, 1, 1, 1)

            tmp_pixel_latent_0 = softsplat(
                tenIn=pixel_latent_0,
                tenFlow=raft_flow01 * cur_t,
                tenMetric=weights1,
                strMode=strtype,
            )
            tmp_pixel_latent_1 = softsplat(
                tenIn=pixel_latent_1,
                tenFlow=raft_flow10 * (1 - cur_t),
                tenMetric=weights2,
                strMode=strtype,
            )

            tmp_pixel_latent = torch.cat(
                [tmp_pixel_latent_0, tmp_pixel_latent_1], dim=1
            )
            tmp_pixel_latent = tmp_pixel_latent + self.res_conv(
                torch.cat([pixel_latent_0, pixel_latent_1, tmp_pixel_latent], dim=1)
            )
            pixel_latent.append(tmp_pixel_latent.permute(0, 2, 3, 1))

        all_outputs = []
        permute_idx_range = [i for i in range(1, f.ndim - 1)]
        for idx, c in enumerate(coord):
            assert c[0][0, 0, 0, 0, 0] == t[idx][0].squeeze()
            assert isinstance(c, tuple)

            if c[1] is None:
                outputs = self.hyponet(
                    c, modulation_params_dict=None, pixel_latent=pixel_latent[idx]
                ).permute(0, -1, *permute_idx_range)
            else:
                outputs = self.hyponet(
                    c, modulation_params_dict=None, pixel_latent=pixel_latent[idx]
                )
            all_outputs.append(outputs)

        return all_outputs

    def warp_w_mask(self, img0, img1, ft0, ft1, mask, scale=1):
        ft0 = scale * resize(ft0, scale_factor=scale)
        ft1 = scale * resize(ft1, scale_factor=scale)
        mask = resize(mask, scale_factor=scale).sigmoid()
        img0_warp = warp(img0, ft0)
        img1_warp = warp(img1, ft1)
        img_warp = mask * img0_warp + (1 - mask) * img1_warp
        return img_warp

    def frame_synthesize(
        self, img_xs, flow_t, features0, features1, corr_fn, cur_t, full_img=None
    ):
        """
        flow_t: b,2,h,w
        cur_t: b,1,1,1
        """
        batch_size = img_xs.shape[0]
        img0 = 2 * img_xs[:, :, 0] - 1.0
        img1 = 2 * img_xs[:, :, 1] - 1.0

        ##################### update the predicted flow #####################
        ## initialize coordinates for looking up
        lookup_coord = self.flow_estimator.build_coord(img_xs[:, :, 0]).to(
            img_xs[:, :, 0].device
        )

        flow_t0_fullsize = flow_t * (-cur_t)
        flow_t1_fullsize = flow_t * (1.0 - cur_t)

        inv = 1 / 4
        flow_t0_inr4 = inv * resize(flow_t0_fullsize, inv)
        flow_t1_inr4 = inv * resize(flow_t1_fullsize, inv)

        ############################# scale 1/4 #############################
        # i. Initialize feature t at scale 1/4
        flowt0_4, flowt1_4, ft_4_ = self.amt_init_decoder(
            features0[-1],
            features1[-1],
            flow_t0_inr4,
            flow_t1_inr4,
            img0=img0,
            img1=img1,
        )
        mask_4_, ft_4_ = ft_4_[:, :1], ft_4_[:, 1:]
        img_warp_4 = self.warp_w_mask(img0, img1, flowt0_4, flowt1_4, mask_4_, scale=4)
        img_warp_4 = (img_warp_4 + 1.0) / 2
        img_warp_4 = torch.clamp(img_warp_4, 0, 1)

        corr_4, flow_4_lr = self._amt_corr_scale_lookup(
            corr_fn, lookup_coord, flowt0_4, flowt1_4, cur_t, downsample=2
        )

        delta_ft_4_, delta_flow_4 = self.amt_update4_low(ft_4_, flow_4_lr, corr_4)
        delta_flow0_4, delta_flow1_4 = torch.chunk(delta_flow_4, 2, 1)
        flowt0_4 = flowt0_4 + delta_flow0_4
        flowt1_4 = flowt1_4 + delta_flow1_4
        ft_4_ = ft_4_ + delta_ft_4_

        # iii. residue update with lookup corr
        corr_4 = resize(corr_4, scale_factor=2.0)

        flow_4 = torch.cat([flowt0_4, flowt1_4], dim=1)
        delta_ft_4_, delta_flow_4 = self.amt_update4_high(ft_4_, flow_4, corr_4)
        flowt0_4 = flowt0_4 + delta_flow_4[:, :2]
        flowt1_4 = flowt1_4 + delta_flow_4[:, 2:4]
        ft_4_ = ft_4_ + delta_ft_4_

        ############################# scale 1/1 #############################
        flowt0_1, flowt1_1, mask, img_res = self.amt_final_decoder(
            ft_4_,
            features0[0],
            features1[0],
            flowt0_4,
            flowt1_4,
            mask=mask_4_,
            img0=img0,
            img1=img1,
        )

        if full_img is not None:
            img0 = 2 * full_img[:, :, 0] - 1.0
            img1 = 2 * full_img[:, :, 1] - 1.0
            inv = img1.shape[2] / flowt0_1.shape[2]
            flowt0_1 = inv * resize(flowt0_1, scale_factor=inv)
            flowt1_1 = inv * resize(flowt1_1, scale_factor=inv)
            flow_t0_fullsize = inv * resize(flow_t0_fullsize, scale_factor=inv)
            flow_t1_fullsize = inv * resize(flow_t1_fullsize, scale_factor=inv)
            mask = resize(mask, scale_factor=inv)
            img_res = resize(img_res, scale_factor=inv)

        imgt_pred = multi_flow_combine(
            self.amt_comb_block, img0, img1, flowt0_1, flowt1_1, mask, img_res, None
        )
        imgt_pred = torch.clamp(imgt_pred, 0, 1)

        ######################################################################

        flowt0_1 = flowt0_1.reshape(
            batch_size, self.num_flows, 2, img0.shape[-2], img0.shape[-1]
        )
        flowt1_1 = flowt1_1.reshape(
            batch_size, self.num_flows, 2, img0.shape[-2], img0.shape[-1]
        )

        flowt0_pred = [flowt0_1, flowt0_4]
        flowt1_pred = [flowt1_1, flowt1_4]
        other_pred = [img_warp_4]
        return imgt_pred, flowt0_pred, flowt1_pred, other_pred

    def forward(self, img_xs, coord=None, t=None, ds_factor=None):
        assert isinstance(t, list)
        assert isinstance(coord, list)
        assert len(t) == len(coord)
        full_size_img = None
        if ds_factor is not None:
            full_size_img = img_xs.clone()
            img_xs = torch.cat(
                [
                    resize(img_xs[:, :, 0], scale_factor=ds_factor).unsqueeze(2),
                    resize(img_xs[:, :, 1], scale_factor=ds_factor).unsqueeze(2),
                ],
                dim=2,
            )

        (
            normal_flows,
            flows,
            flow_scalers,
            features0,
            features1,
            corr_fn,
            preserved_raft_flows,
        ) = self.cal_bidirection_flow(255 * img_xs[:, :, 0], 255 * img_xs[:, :, 1])
        assert coord is not None

        # List of flows
        normal_inr_flows = self.predict_flow(normal_flows, coord, t, flows)

        ############ Unnormalize the predicted/reconstructed flow ############
        start_idx = 0
        if coord[0][1] is not None:
            # Subsmapled flows for reconstruction supervision in the GIMM module
            # In such case, first two coords in the list are subsampled for supervision up-mentioned
            # Normalized flow_t towards positive t-axis
            assert len(coord) > 2
            flow_t = [
                unnormalize_flow(normal_inr_flows[i], flow_scalers).squeeze()
                for i in range(2, len(coord))
            ]
            start_idx = 2
        else:
            flow_t = [
                unnormalize_flow(normal_inr_flows[i], flow_scalers).squeeze()
                for i in range(len(coord))
            ]

        imgt_preds, flowt0_preds, flowt1_preds, all_others = [], [], [], []

        for idx in range(start_idx, len(coord)):
            cur_flow_t = flow_t[idx - start_idx]
            cur_t = t[idx].reshape(-1, 1, 1, 1)
            if cur_flow_t.ndim != 4:
                cur_flow_t = cur_flow_t.unsqueeze(0)
                assert cur_flow_t.ndim == 4

            imgt_pred, flowt0_pred, flowt1_pred, others = self.frame_synthesize(
                img_xs,
                cur_flow_t,
                features0,
                features1,
                corr_fn,
                cur_t,
                full_img=full_size_img,
            )

            imgt_preds.append(imgt_pred)
            flowt0_preds.append(flowt0_pred)
            flowt1_preds.append(flowt1_pred)
            all_others.append(others)

        return {
            "imgt_pred": imgt_preds,
            "other_pred": all_others,
            "flowt0_pred": flowt0_preds,
            "flowt1_pred": flowt1_preds,
            "raft_flow": preserved_raft_flows,
            "ninrflow": normal_inr_flows,
            "nflow": normal_flows,
            "flowt": flow_t,
        }

    def warp_frame(self, frame, flow):
        return warp(frame, flow)

    def compute_psnr(self, preds, targets, reduction="mean"):
        assert reduction in ["mean", "sum", "none"]
        batch_size = preds.shape[0]
        sample_mses = torch.reshape((preds - targets) ** 2, (batch_size, -1)).mean(
            dim=-1
        )

        if reduction == "mean":
            psnr = (-10 * torch.log10(sample_mses)).mean()
        elif reduction == "sum":
            psnr = (-10 * torch.log10(sample_mses)).sum()
        else:
            psnr = -10 * torch.log10(sample_mses)

        return psnr

    def sample_coord_input(
        self,
        batch_size,
        s_shape,
        t_ids,
        coord_range=None,
        upsample_ratio=1.0,
        device=None,
    ):
        assert device is not None
        assert coord_range is None
        coord_inputs = self.coord_sampler(
            batch_size, s_shape, t_ids, coord_range, upsample_ratio, device
        )
        return coord_inputs

    def cal_splatting_weights(self, raft_flow01, raft_flow10):
        batch_size = raft_flow01.shape[0]
        raft_flows = torch.cat([raft_flow01, raft_flow10], dim=0)

        ## flow variance metric
        sqaure_mean, mean_square = torch.split(
            F.conv3d(
                F.pad(
                    torch.cat([raft_flows**2, raft_flows], 1),
                    (1, 1, 1, 1),
                    mode="reflect",
                ).unsqueeze(1),
                self.g_filter,
            ).squeeze(1),
            2,
            dim=1,
        )
        var = (
            (sqaure_mean - mean_square**2)
            .clamp(1e-9, None)
            .sqrt()
            .mean(1)
            .unsqueeze(1)
        )
        var01 = var[:batch_size]
        var10 = var[batch_size:]

        ## flow warp metirc
        f01_warp = -warp(raft_flow10, raft_flow01)
        f10_warp = -warp(raft_flow01, raft_flow10)
        err01 = (
            torch.nn.functional.l1_loss(
                input=f01_warp, target=raft_flow01, reduction="none"
            )
            .mean(1)
            .unsqueeze(1)
        )
        err02 = (
            torch.nn.functional.l1_loss(
                input=f10_warp, target=raft_flow10, reduction="none"
            )
            .mean(1)
            .unsqueeze(1)
        )

        weights1 = 1 / (1 + err01 * self.alpha_fe) + 1 / (1 + var01 * self.alpha_v)
        weights2 = 1 / (1 + err02 * self.alpha_fe) + 1 / (1 + var10 * self.alpha_v)

        return weights1, weights2

    def _amt_corr_scale_lookup(self, corr_fn, coord, flow0, flow1, embt, downsample=1):
        # convert t -> 0 to 0 -> 1 | convert t -> 1 to 1 -> 0
        # based on linear assumption
        t0_scale = 1.0 / embt
        t1_scale = 1.0 / (1.0 - embt)
        if downsample != 1:
            inv = 1 / downsample
            flow0 = inv * resize(flow0, scale_factor=inv)
            flow1 = inv * resize(flow1, scale_factor=inv)

        corr0, corr1 = corr_fn(coord + flow1 * t1_scale, coord + flow0 * t0_scale)
        corr = torch.cat([corr0, corr1], dim=1)
        flow = torch.cat([flow0, flow1], dim=1)
        return corr, flow
