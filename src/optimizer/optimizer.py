# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------

import torch


def create_inr_optimizer(model, config):
    optimizer_type = config.type.lower()
    if not config.ft:
        param_dicts = model.parameters()
    else:
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "amt_" in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "amt_" not in n and p.requires_grad
                ],
                "lr": config.init_lr * 0.01,
                "weight_decay": config.weight_decay * 0.01,
            },
        ]
        if len(param_dicts[1]["params"]) == 0:
            print("only amt_part will be trained")
        # assert len(param_dicts[1]['params']) > 0

    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=config.init_lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            param_dicts,
            lr=config.init_lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            param_dicts,
            lr=config.init_lr,
            weight_decay=config.weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"{optimizer_type} invalid..")
    return optimizer


def create_optimizer(model, config):
    arch_type = config.arch.type.lower()
    if (
        "inr" in config.arch.type
        or "dnn" in config.arch.type
        or "gimm" in config.arch.type
    ):
        optimizer = create_inr_optimizer(model, config.optimizer)
    else:
        raise ValueError(f"{arch_type} invalid..")
    return optimizer
