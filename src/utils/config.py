# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------

from omegaconf import OmegaConf
from easydict import EasyDict as edict
import yaml

from models.generalizable_INR.configs import GIMMConfig, GIMMVFIConfig
import os.path as osp


def easydict_to_dict(obj):
    if not isinstance(obj, edict):
        return obj
    else:
        return {k: easydict_to_dict(v) for k, v in obj.items()}


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = easydict_to_dict(config)
        config = OmegaConf.create(config)
    return config


def augment_arch_defaults(arch_config):
    if arch_config.type == "gimm":
        arch_defaults = GIMMConfig.create(arch_config)
    elif arch_config.type == "gimmvfi":
        arch_defaults = GIMMVFIConfig.create(arch_config)
    elif arch_config.type == "gimmvfi_f" or arch_config.type == "gimmvfi_r":
        arch_defaults = GIMMVFIConfig.create(arch_config)
    else:
        raise ValueError(f"{arch_config.type} is not implemented for default arguments")

    return OmegaConf.merge(arch_defaults, arch_config)


def augment_optimizer_defaults(optim_config):
    defaults = OmegaConf.create(
        {
            "type": "adamW",
            "max_gn": None,
            "warmup": {
                "mode": "linear",
                "start_from_zero": (True if optim_config.warmup.epoch > 0 else False),
            },
        }
    )
    return OmegaConf.merge(defaults, optim_config)


def augment_defaults(config):
    defaults = OmegaConf.create(
        {
            "arch": augment_arch_defaults(config.arch),
            "dataset": {
                "transforms": {"type": None},
            },
            "optimizer": augment_optimizer_defaults(config.optimizer),
            "experiment": {
                "test_freq": 10,
                "amp": False,
            },
        }
    )

    if "inr" in config.arch.type or "gimm" in config.arch.type:
        subsample_defaults = OmegaConf.create({"type": None, "ratio": 1.0})
        loss_defaults = OmegaConf.create(
            {
                "loss": {
                    "type": "mse",
                    "subsample": subsample_defaults,
                    "coord_noise": None,
                }
            }
        )
        defaults = OmegaConf.merge(defaults, loss_defaults)
    config = OmegaConf.merge(defaults, config)
    return config


def augment_dist_defaults(config, distenv):
    config = config.copy()
    local_batch_size = config.experiment.batch_size
    world_batch_size = distenv.world_size * local_batch_size
    total_batch_size = config.experiment.get("total_batch_size", world_batch_size)

    if total_batch_size % world_batch_size != 0:
        raise ValueError("total batch size must be divisible by world batch size")
    else:
        grad_accm_steps = total_batch_size // world_batch_size

    config.optimizer.grad_accm_steps = grad_accm_steps
    config.experiment.total_batch_size = total_batch_size
    return config


def config_setup(args, distenv, config_path, extra_args=()):
    if not osp.isfile(config_path):
        config_path = args.model_config
    if args.eval:
        config = load_config(config_path)
        config = augment_defaults(config)
        if hasattr(args, "test_batch_size"):
            config.experiment.batch_size = args.test_batch_size
        if not hasattr(config, "seed"):
            config.seed = args.seed

    elif args.resume:
        config = load_config(config_path)
        if distenv.world_size != config.runtime.distenv.world_size:
            raise ValueError("world_size not identical to the resuming config")
        config.runtime = {"args": vars(args), "distenv": distenv}

    else:  # training
        config_path = args.model_config
        config = load_config(config_path)

        extra_config = OmegaConf.from_dotlist(extra_args)
        config = OmegaConf.merge(config, extra_config)

        config = augment_defaults(config)
        config = augment_dist_defaults(config, distenv)

        config.seed = args.seed
        config.runtime = {
            "args": vars(args),
            "extra_config": extra_config,
            "distenv": distenv,
        }

    return config
