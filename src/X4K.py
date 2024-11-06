# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ema-vfi: https://github.com/MCG-NJU/EMA-VFI
# --------------------------------------------------------

import os
import torch
import cv2
import glob
import argparse

from tqdm import tqdm
import numpy as np


from models import create_model
from utils.utils import set_seed, InputPadder
from utils.setup import single_setup
from utils.lpips import calc_lpips


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model-config", type=str, default="configs/gimmvfi_f/gimmvfi_f_arb.yaml"
    )
    parser.add_argument("-p", "--pred_save_path", type=str, default="./eval_output/x4k")
    parser.add_argument("-l", "--load-path", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true")
    return parser


def parse_args():
    parser = default_parser()
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def getXVFI(dir, multiple=8, t_step_size=32):
    """make [I0,I1,It,t,scene_folder]"""
    testPath = []
    t = np.linspace((1 / multiple), (1 - (1 / multiple)), (multiple - 1))
    for type_folder in sorted(glob.glob(os.path.join(dir, "*", ""))):
        for scene_folder in sorted(glob.glob(os.path.join(type_folder, "*", ""))):
            frame_folder = sorted(glob.glob(scene_folder + "*.png"))
            for idx in range(0, len(frame_folder), t_step_size):
                if idx == len(frame_folder) - 1:
                    break
                for mul in range(multiple - 1):
                    I0I1It_paths = []
                    I0I1It_paths.append(frame_folder[idx])
                    I0I1It_paths.append(frame_folder[idx + t_step_size])
                    I0I1It_paths.append(
                        frame_folder[idx + int((t_step_size // multiple) * (mul + 1))]
                    )
                    I0I1It_paths.append(t[mul])
                    testPath.append(I0I1It_paths)

    return testPath


if __name__ == "__main__":
    args, extra_args = parse_args()
    set_seed(args.seed)
    config = single_setup(args, extra_args)
    device = torch.device("cuda")

    os.makedirs(args.pred_save_path, exist_ok=True)

    model, _ = create_model(config.arch)
    model = model.to(device)

    # Checkpoint loading
    if not args.load_path == "":
        ckpt = torch.load(args.load_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        raise ValueError("--load-path must be specified in evaluation mode")
    model.eval()

    path = "./data/x4k/test"
    listFiles = getXVFI(path)

    for strMode in ["XTEST-2k", "XTEST-4k"]:
        psnr_list, lpips_list = [], []
        for d in tqdm(listFiles):
            # prepare data b,c,h,w 0-1
            I0 = (np.array(cv2.imread(d[0])).astype(np.float32) * (1.0 / 255.0))[
                :, :, ::-1
            ]
            I2 = (np.array(cv2.imread(d[1])).astype(np.float32) * (1.0 / 255.0))[
                :, :, ::-1
            ]
            I1 = (np.array(cv2.imread(d[2])).astype(np.float32) * (1.0 / 255.0))[
                :, :, ::-1
            ]
            if strMode == "XTEST-2k":  # downsample
                ds_factor = 0.5
                I0 = cv2.resize(
                    src=I0,
                    dsize=(2048, 1080),
                    fx=0.0,
                    fy=0.0,
                    interpolation=cv2.INTER_AREA,
                )
                I2 = cv2.resize(
                    src=I2,
                    dsize=(2048, 1080),
                    fx=0.0,
                    fy=0.0,
                    interpolation=cv2.INTER_AREA,
                )
                I1 = cv2.resize(
                    src=I1,
                    dsize=(2048, 1080),
                    fx=0.0,
                    fy=0.0,
                    interpolation=cv2.INTER_AREA,
                )
            else:
                ds_factor = 0.25
            I0 = torch.FloatTensor(
                np.ascontiguousarray(I0.transpose(2, 0, 1)[None, :, :, :])
            ).cuda()
            I2 = torch.FloatTensor(
                np.ascontiguousarray(I2.transpose(2, 0, 1)[None, :, :, :])
            ).cuda()
            I1 = torch.FloatTensor(
                np.ascontiguousarray(I1.transpose(2, 0, 1)[None, :, :, :])
            ).cuda()
            padder = InputPadder(I0.shape, 32)
            I0, I2 = padder.pad(I0, I2)
            xs = torch.cat((I0.unsqueeze(2), I2.unsqueeze(2)), dim=2).to(
                device, non_blocking=True
            )

            batch_size = xs.shape[0]
            s_shape = xs.shape[-2:]
            assert d[-1] <= 1
            coord_inputs = [
                (
                    model.sample_coord_input(
                        batch_size,
                        s_shape,
                        [d[-1]],
                        device=xs.device,
                        upsample_ratio=ds_factor,
                    ),
                    None,
                )
            ]
            timesteps = [d[-1] * torch.ones(xs.shape[0]).to(xs.device).to(torch.float)]

            with torch.no_grad():
                all_outputs = model(xs, coord_inputs, t=timesteps, ds_factor=ds_factor)
                I1_pred = all_outputs["imgt_pred"][0]  # 1,c,h,
                I1_pred = padder.unpad(I1_pred)
                I1_pred = (
                    (I1_pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0)
                    .clip(0.0, 255.0)
                    .round()
                    .astype(np.uint8)
                )
                I1_pred = (
                    torch.FloatTensor(I1_pred.transpose(2, 0, 1)[None, :, :, :]).cuda()
                    / 255.0
                )

            def calculate_psnr(img1, img2):
                psnr = -10 * torch.log10(((img1 - img2) * (img1 - img2)).mean())
                return psnr.detach().cpu().numpy()

            lpips_gt = 2 * I1.detach() - 1
            lpips_pred = 2 * I1_pred.detach() - 1
            lpips = calc_lpips(lpips_gt, lpips_pred)[0].detach().cpu().numpy()
            psnr = calculate_psnr(
                I1[0].permute(1, 2, 0), I1_pred[0].permute(1, 2, 0).detach()
            )
            lpips_list.append(lpips)
            psnr_list.append(psnr)
            cur_name = d[2].split("/")[-2] + "_" + d[2].split("/")[-1]
            # cv2.imwrite(os.path.join(gt_save_path, cur_name),
            #             (I1.squeeze().cpu().numpy().transpose(1, 2, 0) * 255.0)[:,:,::-1].astype(np.uint8))
            cv2.imwrite(
                os.path.join(args.pred_save_path, cur_name),
                (I1_pred.squeeze().cpu().numpy().transpose(1, 2, 0) * 255.0)[
                    :, :, ::-1
                ].astype(np.uint8),
            )

        print(f"{strMode}  PSNR: {np.mean(psnr_list)}  LPIPS: {np.mean(lpips_list)}")
