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
from PIL import Image


def random_resize(img0, imgt, img1, p=0.1):
    if random.uniform(0, 1) < p:
        img0 = cv2.resize(
            img0, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR
        )
        imgt = cv2.resize(
            imgt, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR
        )
        img1 = cv2.resize(
            img1, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR
        )
    return img0, imgt, img1


def random_crop(img0, imgt, img1, crop_size=(224, 224)):
    h, w = crop_size[0], crop_size[1]
    ih, iw, _ = img0.shape
    x = np.random.randint(0, ih - h + 1)
    y = np.random.randint(0, iw - w + 1)
    img0 = img0[x : x + h, y : y + w, :]
    imgt = imgt[x : x + h, y : y + w, :]
    img1 = img1[x : x + h, y : y + w, :]
    return img0, imgt, img1


def random_reverse_channel(img0, imgt, img1, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, :, ::-1]
        imgt = imgt[:, :, ::-1]
        img1 = img1[:, :, ::-1]
    return img0, imgt, img1


def random_vertical_flip(img0, imgt, img1, p=0.3):
    if random.uniform(0, 1) < p:
        img0 = img0[::-1]
        imgt = imgt[::-1]
        img1 = img1[::-1]

    return img0, imgt, img1


def random_rotate(img0, imgt, img1, p=0.05):
    if random.uniform(0, 1) < p:
        img0 = img0.transpose((1, 0, 2))
        imgt = imgt.transpose((1, 0, 2))
        img1 = img1.transpose((1, 0, 2))
    return img0, imgt, img1


def random_horizontal_flip(img0, imgt, img1, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, ::-1]
        imgt = imgt[:, ::-1]
        img1 = img1[:, ::-1]
    return img0, imgt, img1


def random_reverse_time(img0, imgt, img1, t, p=0.5):
    if random.uniform(0, 1) < p:
        tmp = img1
        img1 = img0
        img0 = tmp
        t = 1 - t
    return img0, imgt, img1, t


class Vimeo_Arbitrary(Dataset):
    def __init__(self, dataset_name, path, aug=True, crop_size=(224, 224)):
        self.dataset_name = dataset_name
        self.if_aug = aug
        self.h = 256
        self.w = 448
        self.crop_size = crop_size
        self.data_root = path
        self.image_root = os.path.join(self.data_root, "sequences")
        self.load_data()

    def load_data(self):
        train_fn = os.path.join(self.data_root, "all_sep.txt")
        test_fn = os.path.join(
            self.data_root.replace("vimeo_septuplet", "vimeo_triplet"),
            "tri_testlist.txt",
        )
        with open(train_fn, "r") as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, "r") as f:
            self.testlist = f.read().splitlines()[:-1]

        if self.dataset_name != "test":
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist
        # self.meta_data = self.meta_data[:100]

    def __len__(self):
        return len(self.meta_data)

    def getimg(self, index):
        if self.dataset_name != "test":
            bgpath = os.path.join(self.image_root, self.meta_data[index])
            idx = sorted(np.random.permutation(7)[:3])
        else:
            bgpath = os.path.join(
                self.image_root.replace("vimeo_septuplet", "vimeo_triplet"),
                self.meta_data[index],
            )
            idx = sorted(np.random.permutation(3)[:3])
        imgs = [
            Image.open(bgpath + "/im%d.png" % (idx[0] + 1)),
            Image.open(bgpath + "/im%d.png" % (idx[1] + 1)),
            Image.open(bgpath + "/im%d.png" % (idx[2] + 1)),
        ]

        t = (idx[1] - idx[0]) / (idx[2] - idx[0])
        img0 = np.array(imgs[0].convert("RGB"))
        gt = np.array(imgs[1].convert("RGB"))
        img1 = np.array(imgs[2].convert("RGB"))
        return img0, gt, img1, t

    def __getitem__(self, index):
        img0, gt, img1, t = self.getimg(index)

        if "train" in self.dataset_name:
            if self.if_aug:
                img0, gt, img1 = random_resize(img0, gt, img1, p=0.1)
                img0, gt, img1 = random_crop(img0, gt, img1, crop_size=self.crop_size)
                img0, gt, img1 = random_reverse_channel(img0, gt, img1, p=0.5)
                img0, gt, img1, t = random_reverse_time(img0, gt, img1, t, p=0.5)
                img0, gt, img1 = random_vertical_flip(img0, gt, img1, p=0.3)
                img0, gt, img1 = random_horizontal_flip(img0, gt, img1, p=0.5)
                img0, gt, img1 = random_rotate(img0, gt, img1, p=0.05)
            else:
                img0, gt, img1 = random_crop(img0, gt, img1, crop_size=(224, 224))
                if random.uniform(0, 1) < 0.5:
                    img0 = img0[:, :, ::-1]
                    img1 = img1[:, :, ::-1]
                    gt = gt[:, :, ::-1]
                if random.uniform(0, 1) < 0.5:
                    img1, img0 = img0, img1
                    t = 1 - t
                if random.uniform(0, 1) < 0.5:
                    img0 = img0[::-1]
                    img1 = img1[::-1]
                    gt = gt[::-1]
                if random.uniform(0, 1) < 0.5:
                    img0 = img0[:, ::-1]
                    img1 = img1[:, ::-1]
                    gt = gt[:, ::-1]

                p = random.uniform(0, 1)
                if p < 0.25:
                    img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                    gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                    img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
                elif p < 0.5:
                    img0 = cv2.rotate(img0, cv2.ROTATE_180)
                    gt = cv2.rotate(gt, cv2.ROTATE_180)
                    img1 = cv2.rotate(img1, cv2.ROTATE_180)
                elif p < 0.75:
                    img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1) / 255.0
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1) / 255.0
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1) / 255.0

        return {
            "xs": torch.cat(
                (img0.unsqueeze(1), img1.unsqueeze(1), gt.unsqueeze(1)), 1
            ).to(
                torch.float
            ),  # c,t,h,w
            "t": (t * torch.ones(1)).to(torch.float),
        }
