# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------


import os
from torch.utils.tensorboard import SummaryWriter


class Writer:
    def __init__(self, result_path):
        self.result_path = result_path

        self.writer_trn = SummaryWriter(os.path.join(result_path, "train"))
        self.writer_val = SummaryWriter(os.path.join(result_path, "valid"))
        self.writer_val_ema = SummaryWriter(os.path.join(result_path, "valid_ema"))

    def _get_writer(self, mode):
        if mode == "train":
            writer = self.writer_trn
        elif mode == "valid":
            writer = self.writer_val
        elif mode == "valid_ema":
            writer = self.writer_val_ema
        else:
            raise ValueError(f"{mode} is not valid..")

        return writer

    def add_scalar(self, tag, scalar, mode, epoch=0):
        writer = self._get_writer(mode)
        writer.add_scalar(tag, scalar, epoch)

    def add_image(self, tag, image, mode, epoch=0):
        writer = self._get_writer(mode)
        writer.add_image(tag, image, epoch)

    def add_text(self, tag, text, mode, epoch=0):
        writer = self._get_writer(mode)
        writer.add_text(tag, text, epoch)

    def add_audio(self, tag, audio, mode, sampling_rate=16000, epoch=0):
        writer = self._get_writer(mode)
        writer.add_audio(tag, audio, epoch, sampling_rate)

    def close(self):
        self.writer_trn.close()
        self.writer_val.close()
        self.writer_val_ema.close()
