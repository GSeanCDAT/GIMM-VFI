# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------

from .ema import ExponentialMovingAverage
from .generalizable_INR import gimmvfi_f, gimmvfi_r, gimm


def create_model(config, ema=False):
    model_type = config.type.lower()
    if model_type == "gimm":
        model = gimm(config)
        model_ema = gimm(config) if ema else None
    elif model_type == "gimmvfi_f":
        model = gimmvfi_f(config)
        model_ema = gimmvfi_f(config) if ema else None
    elif model_type == "gimmvfi_r":
        model = gimmvfi_r(config)
        model_ema = gimmvfi_r(config) if ema else None
    else:
        raise ValueError(f"{model_type} is invalid..")

    if ema:
        mu = config.ema
        if config.ema_value is not None:
            mu = config.ema_value
        model_ema = ExponentialMovingAverage(model_ema, mu)
        model_ema.eval()
        model_ema.update(model, step=-1)

    return model, model_ema
