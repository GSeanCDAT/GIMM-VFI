# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import argparse
import torch

import numpy as np
from tqdm import tqdm

from utils.frame_utils import readFlow
from models import create_model
from utils.utils import set_seed
from utils.setup import single_setup


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model-config", type=str, default="configs/gimm/gimm.yaml"
    )
    parser.add_argument("-l", "--load-path", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true")
    return parser


def parse_args():
    parser = default_parser()
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def process_flow(flowpath):
    flow = readFlow(flowpath)
    flow = torch.from_numpy(flow.copy()).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    return flow


if __name__ == "__main__":
    args, extra_args = parse_args()
    set_seed(args.seed)
    config = single_setup(args, extra_args)
    device = torch.device("cuda")

    model, _ = create_model(config.arch)
    model = model.to(device)

    # Checkpoint loading
    if not args.load_path == "":
        ckpt = torch.load(args.load_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        raise ValueError("--load-path must be specified in evaluation or resume mode")

    model.eval()
    crop_size = None
    path = "data/vimeo90k/vimeo_septuplet"
    with open(path + "/sep_testlist.txt", "r") as f:
        testlist = f.read().splitlines()[:-1]

    psnr_list, epe_list = [], []
    for name in tqdm(testlist):
        for t_id in range(2, 7):
            # oi path info
            flow01_path = (
                "data/vimeo90k/vimeo_septuplet/flow_sequences/" + name + "/im1_im7.flo"
            )
            flow10_path = (
                "data/vimeo90k/vimeo_septuplet/flow_sequences/" + name + "/im7_im1.flo"
            )
            flow_gt_fw_path = (
                "data/vimeo90k/vimeo_septuplet/flow_sequences/"
                + name
                + "/im%d_im7.flo" % t_id
            )
            flow_gt_bw_path = (
                "data/vimeo90k/vimeo_septuplet/flow_sequences/"
                + name
                + "/im%d_im1.flo" % t_id
            )

            # prepare data
            flow_gt_fw = process_flow(flow_gt_fw_path)
            flow_gt_bw = process_flow(flow_gt_bw_path)
            flow_gt = flow_gt_fw - flow_gt_bw
            flow_gt = flow_gt.unsqueeze(2)
            target = flow_gt.to(device, non_blocking=True).detach()

            flow01 = process_flow(flow01_path).unsqueeze(2)
            flow10 = process_flow(flow10_path).unsqueeze(2)
            xs = torch.cat((flow01, -flow10), dim=2).to(device, non_blocking=True)

            def xytshape2coordinate(
                spatial_shape,
                batch_size,
                temporal_shape,
                time_range=(0.0, 0.1),
                coord_range=(-1.0, 1.0),
                upsample_ratio=1,
                device=None,
            ):
                coords = []
                num_t = temporal_shape
                _coords = torch.arange(num_t, device=device) / (num_t - 1)
                _coords = time_range[0] + (time_range[1] - time_range[0]) * _coords
                coords.append(_coords)
                for num_s in spatial_shape:
                    num_s = int(num_s * upsample_ratio)
                    _coords = (0.5 + torch.arange(num_s, device=device)) / num_s
                    _coords = (
                        coord_range[0] + (coord_range[1] - coord_range[0]) * _coords
                    )
                    coords.append(_coords)
                coords = torch.meshgrid(*coords, indexing="ij")
                coords = torch.stack(coords, dim=-1)
                ones_like_shape = (1,) * coords.ndim
                coords = coords.unsqueeze(0).repeat(batch_size, *ones_like_shape)
                # print(coords)
                return coords  # (B,T,H,W,3)

            coord_inputs = xytshape2coordinate(
                (xs.shape[3], xs.shape[4]), 1, 7, time_range=(0.0, 1.0), device=device
            )
            # print(coord_inputs[0,:,0,0,0])
            coord_inputs = coord_inputs[:, t_id - 1 : t_id]

            model.zero_grad()
            scaler = (
                torch.max(torch.abs(xs))
                .clone()
                .to(device, non_blocking=True)
                .detach()
                .reshape(1, 1)
            )

            def normalize(f):
                return (f / scaler + 1.0) / 2.0

            def unnormalize(f):
                return (f * 2.0 - 1.0) * scaler

            with torch.no_grad():
                ori_flow = torch.cat((xs[:, :, :1], -xs[:, :, 1:2]), dim=2)
                xs = normalize(xs)
                cur_t = torch.tensor([t_id / 6]).to(device)
                outputs = model(xs, coord_inputs, ori_flow=ori_flow, timesteps=cur_t)
                target = normalize(target)
                psnr = (
                    model.compute_loss(outputs, target, reduction="sum")["psnr"]
                    .cpu()
                    .numpy()
                )

            target = unnormalize(target)
            epe = (
                unnormalize(outputs)[0, :, 0].detach().cpu()
                - target[:, :, :].squeeze().detach().cpu()
            ) ** 2
            assert epe.ndim == 3
            epe = epe.sum(dim=0).sqrt().mean().cpu().numpy()
            psnr_list.append(psnr)
            epe_list.append(epe)

    print("Avg PSNR: {} EPE: {}".format(np.mean(psnr_list), np.mean(epe_list)))
