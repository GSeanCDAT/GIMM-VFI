# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------

from .trainer_gimm import Trainer as TrainerGIMM
from .trainer_gimmvfi import Trainer as TrainerGIMMVFI


def create_trainer(config):
    if config.arch.type in ["gimm"]:
        return TrainerGIMM
    elif config.arch.type in ["gimmvfi", "gimmvfi_f", "gimmvfi_r"]:
        return TrainerGIMMVFI
    else:
        print(config.arch.type)
        raise ValueError("architecture type not supported")
