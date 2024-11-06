# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ema-vfi: https://github.com/MCG-NJU/EMA-VFI
# --------------------------------------------------------

import os
import argparse
import torch

import numpy as np
from PIL import Image
from tqdm import tqdm

from models import create_model
from utils.utils import set_seed, InputPadder
from utils.setup import single_setup
from utils.lpips import calc_lpips
import cv2


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model-config", type=str, default="configs/gimmvfi_f/gimmvfi_f_arb.yaml"
    )
    parser.add_argument(
        "-p", "--pred_save_path", type=str, default="./eval_output/snu_film_arb"
    )
    parser.add_argument("-l", "--load-path", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true")
    return parser


def parse_args():
    parser = default_parser()
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def load_image(img_path):
    img = Image.open(img_path)
    img = np.array(img.convert("RGB"))
    img = torch.from_numpy(img.copy()).permute(2, 0, 1) / 255.0
    return img.to(torch.float).unsqueeze(0)


def calculate_psnr(img1, img2):
    psnr = -10 * torch.log10(((img1 - img2) * (img1 - img2)).mean())
    return psnr.detach().cpu().numpy()


if __name__ == "__main__":
    args, extra_args = parse_args()
    set_seed(args.seed)
    config = single_setup(args, extra_args)
    device = torch.device("cuda")

    os.makedirs(args.pred_save_path, exist_ok=True)

    model, model_ema = create_model(config.arch, ema=config.arch.ema is not None)
    model = model.to(device)
    model_ema = model_ema.to(device) if model_ema is not None else None

    # Checkpoint loading
    if not args.load_path == "":
        ckpt = torch.load(args.load_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        raise ValueError("--load-path must be specified in evaluation mode")
    model.eval()

    splits = ["medium", "hard", "extreme"]
    root = "./data/SNU-FILM"
    for split in splits:
        if split == "medium":
            timestep = 4
        elif split == "hard":
            timestep = 8
        elif split == "extreme":
            timestep = 16

        with open(os.path.join(root, f"test-{split}.txt"), "r") as fr:
            file_list = [l.strip().split(" ") for l in fr.readlines()]
        pbar = tqdm(file_list, total=len(file_list))

        psnr_list, lpips_list = [], []
        for name in pbar:
            img_path0 = "./" + name[0]
            img_path2 = "./" + name[2]
            gts = []
            # prepare data b,c,h,w 0-1
            I0 = load_image(img_path0)
            I2 = load_image(img_path2)

            cur_file_names = []
            for t_id in range(1, timestep):
                cur_dir = os.path.join(*name[0].split("/")[:-1])
                cur_name = name[0].split("/")[-1][:-4]
                cur_name = "{:0>{width}}.png".format(
                    int(cur_name) + t_id, width=len(cur_name)
                )
                cur_file_names.append(name[0].split("/")[-2] + "_" + cur_name)
                img_path1 = "./" + os.path.join(cur_dir, cur_name)
                gts.append(load_image(img_path1).to(device, non_blocking=True).detach())

            padder = InputPadder(I0.shape, 32)
            I0, I2 = padder.pad(I0, I2)
            xs = torch.cat((I0.unsqueeze(2), I2.unsqueeze(2)), dim=2).to(
                device, non_blocking=True
            )  # b,c,2,h,w
            batch_size = xs.shape[0]
            s_shape = xs.shape[-2:]
            coord_inputs = [
                (
                    model.sample_coord_input(
                        batch_size,
                        s_shape,
                        [(i + 1) * (1.0 / timestep)],
                        device=xs.device,
                    ),
                    None,
                )
                for i in range(timestep - 1)
            ]

            with torch.no_grad():
                all_outputs = model(
                    xs,
                    coord_inputs,
                    t=[
                        (i + 1)
                        * (1.0 / timestep)
                        * torch.ones(xs.shape[0]).to(xs.device).to(torch.float)
                        for i in range(timestep - 1)
                    ],
                )
                all_outputs = [padder.unpad(im) for im in all_outputs["imgt_pred"]]
            for i in range(timestep - 1):
                lpips_gt = 2 * gts[i].detach() - 1
                lpips_pred = 2 * all_outputs[i].detach() - 1
                lpips = calc_lpips(lpips_gt, lpips_pred)[0].detach().cpu().numpy()
                psnr = calculate_psnr(
                    gts[i].squeeze().permute(1, 2, 0),
                    all_outputs[i].squeeze().permute(1, 2, 0).detach(),
                )
                lpips_list.append(lpips)
                psnr_list.append(psnr)
                # cv2.imwrite(os.path.join(gt_save_path,cur_file_names[i]),
                #             (gts[i].squeeze().cpu().numpy().transpose(1, 2, 0) * 255.0)[:,:,::-1].astype(np.uint8))
                cv2.imwrite(
                    os.path.join(args.pred_save_path, cur_file_names[i]),
                    (all_outputs[i].squeeze().cpu().numpy().transpose(1, 2, 0) * 255.0)[
                        :, :, ::-1
                    ].astype(np.uint8),
                )
            torch.cuda.empty_cache()
            del xs
            del coord_inputs

            avg_psnr = np.mean(psnr_list)
            avg_lpips = np.mean(lpips_list)

            desc_str = f"[SNU-FILM] [{split}] psnr: {avg_psnr:.02f}, lpips: {avg_lpips:.04f}, interpolation_step: {timestep: 02d}"
            pbar.set_description_str(desc_str)
