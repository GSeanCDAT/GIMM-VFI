# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import cv2
import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from utils.frame_utils import readFlow

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class fast_vimeo_flow(Dataset):
    def __init__(self, dataset_name, path, expansion, aug=True):
        self.dataset_name = dataset_name
        self.expansion = expansion
        self.if_aug = aug
        self.h = 256
        self.w = 448
        self.data_root = path
        self.image_root = os.path.join(self.data_root, "sequences")
        self.flow_root = os.path.join(self.data_root, "flow_sequences")
        self.load_data()

    def load_data(self):
        train_fn = os.path.join(self.data_root, "tri_trainlist.txt")
        test_fn = os.path.join(self.data_root, "tri_testlist.txt")
        with open(train_fn, "r") as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, "r") as f:
            self.testlist = f.read().splitlines()[:-1]

        if self.dataset_name != "test":
            self.len_tri = len(self.trainlist)
        else:
            self.len_tri = len(self.testlist)

        if self.expansion:
            train_fn = os.path.join(
                self.data_root.replace("vimeo_triplet", "vimeo_septuplet"),
                "sep_trainlist.txt",
            )
            test_fn = os.path.join(
                self.data_root.replace("vimeo_triplet", "vimeo_septuplet"),
                "sep_testlist.txt",
            )
            with open(train_fn, "r") as f:
                self.trainlist = self.trainlist + f.read().splitlines()
            with open(test_fn, "r") as f:
                self.testlist = self.testlist + f.read().splitlines()[:-1]

        if self.dataset_name != "test":
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist

        # self.meta_data = self.meta_data[:100]

    def aug(self, flows, h, w):
        ih, iw = self.h, self.w
        x = random.randint(0, iw - w)
        y = random.randint(0, ih - h)
        for i in range(len(flows)):
            flows[i] = flows[i][y : y + h, x : x + w, :]
        return flows

    def __len__(self):
        return len(self.meta_data)

    def getimg(self, index):
        flowpath = os.path.join(self.flow_root, self.meta_data[index])

        # NOTE: all the flows are normalized into the same moving direction
        flows = [
            readFlow(flowpath + "/im1_im3.flo"),
            (
                (
                    readFlow(flowpath + "/im2_im3.flo")
                    - readFlow(flowpath + "/im2_im1.flo")
                )
            ),
            -readFlow(flowpath + "/im3_im1.flo"),
        ]

        if "train" in self.dataset_name:
            flows = self.aug(flows, 256, 256)

        return np.concatenate(flows, axis=-1)

    def __getitem__(self, index):
        flows = self.getimg(index)

        def normalize_flow(fs):
            flow_scaler = torch.max(torch.abs(torch.cat((fs[:2], fs[4:]), 0)))
            fs = fs / flow_scaler
            # Adapted to [0,1]
            fs = (fs + 1.0) / 2.0
            return fs, flow_scaler

        ori_flows = torch.from_numpy(flows.copy()).permute(2, 0, 1)

        flows, flow_scaler = normalize_flow(ori_flows)

        # xs: 3,2,h,w
        return {
            "xs": torch.cat(
                (
                    flows[:2].unsqueeze(1),
                    flows[2:4].unsqueeze(1),
                    flows[4:].unsqueeze(1),
                ),
                1,
            ).to(torch.float),
            "flow_scaler": flow_scaler,
            "ori_flows": torch.cat(
                (ori_flows[:2].unsqueeze(1), -ori_flows[4:].unsqueeze(1)), 1
            ).to(torch.float),
        }
