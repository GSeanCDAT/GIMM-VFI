# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------

import os
import torch

from .flow_dataset import fast_vimeo_flow
from .vimeo_arb import Vimeo_Arbitrary

SMOKE_TEST = bool(os.environ.get("SMOKE_TEST", 0))


def create_dataset(config, is_eval=False, logger=None):
    if config.dataset.type == "fast_vimeo_flow":
        dataset_trn = fast_vimeo_flow(
            "train", config.dataset.path, config.dataset.expansion, config.dataset.aug
        )
        dataset_val = fast_vimeo_flow(
            "test", config.dataset.path, config.dataset.expansion, config.dataset.aug
        )
    elif config.dataset.type == "vimeo_arb":
        dataset_trn = Vimeo_Arbitrary("train", config.dataset.path, config.dataset.aug)
        dataset_val = Vimeo_Arbitrary("test", config.dataset.path, config.dataset.aug)
    else:
        raise ValueError("%s not supported..." % config.dataset.type)

    if SMOKE_TEST:
        dataset_len = config.experiment.total_batch_size * 2
        dataset_trn = torch.utils.data.Subset(
            dataset_trn, torch.randperm(len(dataset_trn))[:dataset_len]
        )
        dataset_val = torch.utils.data.Subset(
            dataset_val, torch.randperm(len(dataset_val))[:dataset_len]
        )

    if logger is not None:
        logger.info(
            f"#train samples: {len(dataset_trn)}, #valid samples: {len(dataset_val)}"
        )

    return dataset_trn, dataset_val
