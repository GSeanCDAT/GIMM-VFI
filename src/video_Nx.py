# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import os
import argparse
import torch
import cv2
import numpy as np

from models import create_model
from utils.utils import set_seed, InputPadder
from utils.setup import single_setup
from utils.flow_viz import flow_to_image
from tqdm import tqdm
from PIL import Image


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model-config", type=str, default="./configs/flow/videodnn.yaml"
    )
    parser.add_argument("--source-path", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--ds-factor", type=int, default=1)
    parser.add_argument("-r", "--result-path", type=str, default="./results.tmp")
    parser.add_argument("-l", "--load-path", type=str, default="")
    parser.add_argument("-p", "--postfix", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true")
    return parser


def parse_args():
    parser = default_parser()
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def load_image(img_path):
    img = Image.open(img_path)
    raw_img = np.array(img.convert("RGB"))
    img = torch.from_numpy(raw_img.copy()).permute(2, 0, 1) / 255.0
    return img.to(torch.float).unsqueeze(0)


def images_to_video(imgs, output_video_path, fps=15):
    height, width, layers = imgs[0].shape
    video = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for img in imgs:
        video.write(img)
    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    args, extra_args = parse_args()
    set_seed(args.seed)
    config = single_setup(args, extra_args)

    device = torch.device("cuda")

    os.makedirs(args.output_path, exist_ok=True)

    model, _ = create_model(config.arch)
    model = model.to(device)

    # Checkpoint loading
    if not args.load_path == "":
        if "ours" in args.load_path:
            ckpt = torch.load(args.load_path, map_location="cpu")

            def convert(param):
                return {
                    k.replace("module.feature_bone", "frame_encoder"): v
                    for k, v in param.items()
                    if "feature_bone" in k
                }

            ckpt = convert(ckpt)
            model.load_state_dict(ckpt, strict=False)
        else:
            ckpt = torch.load(args.load_path, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        ckpt = None
        if args.eval or args.resume:
            raise ValueError(
                "--load-path must be specified in evaluation or resume mode"
            )

    source_path = args.source_path
    output_path = os.path.join(args.output_path, "output.mp4")
    flow_output_path = os.path.join(args.output_path, "flow.mp4")

    img_list = sorted(os.listdir(source_path))
    images = []
    ori_image = []
    flows = []
    start = 0
    end = len(img_list) - 1

    for j in tqdm(range(start, end)):
        img_path0 = os.path.join(source_path, img_list[j])
        img_path2 = os.path.join(source_path, img_list[j + 1])
        # prepare data b,c,h,w
        I0 = load_image(img_path0)
        if j == start:
            images.append(
                (I0.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255.0)[
                    :, :, ::-1
                ].astype(np.uint8)
            )
            ori_image.append(
                (I0.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255.0)[
                    :, :, ::-1
                ].astype(np.uint8)
            )
            images[-1] = cv2.hconcat([ori_image[-1], images[-1]])
        I2 = load_image(img_path2)
        padder = InputPadder(I0.shape, 32)
        I0, I2 = padder.pad(I0, I2)
        xs = torch.cat((I0.unsqueeze(2), I2.unsqueeze(2)), dim=2).to(
            device, non_blocking=True
        )
        model.eval()
        batch_size = xs.shape[0]
        s_shape = xs.shape[-2:]

        model.zero_grad()
        ds_factor = args.ds_factor
        with torch.no_grad():
            coord_inputs = [
                (
                    model.sample_coord_input(
                        batch_size,
                        s_shape,
                        [1 / args.N * i],
                        device=xs.device,
                        upsample_ratio=ds_factor,
                    ),
                    None,
                )
                for i in range(1, args.N)
            ]
            timesteps = [
                i * 1 / args.N * torch.ones(xs.shape[0]).to(xs.device).to(torch.float)
                for i in range(1, args.N)
            ]
            all_outputs = model(xs, coord_inputs, t=timesteps, ds_factor=ds_factor)
            out_frames = [padder.unpad(im) for im in all_outputs["imgt_pred"]]
            out_flowts = [padder.unpad(f) for f in all_outputs["flowt"]]
        flowt_imgs = [
            flow_to_image(
                flowt.squeeze().detach().cpu().permute(1, 2, 0).numpy(),
                convert_to_bgr=True,
            )
            for flowt in out_flowts
        ]
        I1_pred_img = [
            (I1_pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0)[
                :, :, ::-1
            ].astype(np.uint8)
            for I1_pred in out_frames
        ]

        for i in range(args.N - 1):
            images.append(I1_pred_img[i])
            flows.append(flowt_imgs[i])

            images[-1] = cv2.hconcat([ori_image[-1], images[-1]])

        images.append(
            (
                (padder.unpad(I2)).squeeze().detach().cpu().numpy().transpose(1, 2, 0)
                * 255.0
            )[:, :, ::-1].astype(np.uint8)
        )
        ori_image.append(
            (
                (padder.unpad(I2)).squeeze().detach().cpu().numpy().transpose(1, 2, 0)
                * 255.0
            )[:, :, ::-1].astype(np.uint8)
        )
        images[-1] = cv2.hconcat([ori_image[-1], images[-1]])

    images_to_video(images[:-1], output_path, fps=args.N * 2)
    images_to_video(flows, flow_output_path, fps=args.N * 2)
    print(f"=========================Interpolation Finished=========================")

    print(len(images))
