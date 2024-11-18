# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------

import ctypes

libgcc_s = ctypes.CDLL("libgcc_s.so.1")

import argparse
import math

import torch
import torch.distributed as dist

import utils.dist as dist_utils

from models import create_model
from trainers import create_trainer
from datasets import create_dataset
from optimizer import create_optimizer, create_scheduler
from utils.utils import set_seed
from utils.profiler import Profiler
from utils.setup import setup


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-config",
        type=str,
        default="./configs/image/imagenette178_low_rank_modulated_transinr.yaml",
    )
    parser.add_argument("-r", "--result-path", type=str, default="./results.tmp")
    parser.add_argument("-l", "--load-path", type=str, default="")
    parser.add_argument("-p", "--postfix", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def add_dist_arguments(parser):
    parser.add_argument(
        "--world_size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="local rank for distributed training"
    )
    parser.add_argument(
        "--node_rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument("--nnodes", default=-1, type=int)
    parser.add_argument("--nproc_per_node", default=-1, type=int)
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=86400,
        help="time limit (s) to wait for other nodes in DDP",
    )
    return parser


def parse_args():
    parser = default_parser()
    parser = add_dist_arguments(parser)
    args, extra_args = parser.parse_known_args()
    return args, extra_args


if __name__ == "__main__":
    args, extra_args = parse_args()
    set_seed(args.seed)
    config, logger, writer = setup(args, extra_args)
    distenv = config.runtime.distenv
    profiler = Profiler(logger)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda", distenv.local_rank)
    torch.cuda.set_device(device)

    dataset_trn, dataset_val = create_dataset(config, is_eval=args.eval, logger=logger)

    model, model_ema = create_model(config.arch, ema=config.arch.ema is not None)
    model = model.to(device)
    model_ema = model_ema.to(device) if model_ema is not None else None

    if distenv.master:
        print(model)
        profiler.get_model_size(model)
        profiler.get_model_size(model, opt="trainable-only")

    # Checkpoint loading
    if not args.load_path == "":
        ckpt = torch.load(args.load_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
        if model_ema is not None:
            if "state_dict_ema" in ckpt.keys():
                ckpt["state_dict_ema"] = {
                    "module." + k: v for k, v in ckpt["state_dict_ema"].items()
                }
                model_ema.load_state_dict(ckpt["state_dict_ema"])
                model_ema.load_state_dict(ckpt["state_dict_ema"])
            else:
                model_ema.load_state_dict(ckpt["state_dict"], strict=False)

        if distenv.master:
            logger.info(f"{args.load_path} model is loaded")
    else:
        ckpt = None
        if args.eval or args.resume:
            raise ValueError(
                "--load-path must be specified in evaluation or resume mode"
            )

    # Optimizer definition
    if args.eval:
        optimizer, scheduler, epoch_st = None, None, None
    else:
        steps_per_epoch = math.ceil(
            len(dataset_trn) / (config.experiment.batch_size * distenv.world_size)
        )
        steps_per_epoch = steps_per_epoch // config.optimizer.grad_accm_steps

        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(
            optimizer,
            config.optimizer.warmup,
            steps_per_epoch,
            config.experiment.epochs,
            distenv,
        )

        if distenv.master:
            print(optimizer)

        if args.resume:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            epoch_st = ckpt["epoch"]

            if distenv.master:
                logger.info(f"Optimizer, scheduler, and epoch is resumed")
                logger.info(f"resuming from {epoch_st}..")
        else:
            epoch_st = 0

    # Usual DDP setting
    static_graph = False
    model = dist_utils.dataparallel_and_sync(
        distenv, model, static_graph=static_graph, find_unused_parameters=False
    )
    if model_ema is not None:
        model_ema = dist_utils.dataparallel_and_sync(
            distenv, model_ema, static_graph=static_graph
        )

    trainer = create_trainer(config)
    trainer = trainer(
        model, model_ema, dataset_trn, dataset_val, config, writer, device, distenv
    )

    if distenv.master:
        logger.info(f"Trainer created. type: {trainer.__class__}")

    if args.eval:
        trainer.config.experiment.subsample_during_eval = False
        trainer.eval(valid=False, verbose=True)
        trainer.eval(valid=True, verbose=True)
        if model_ema is not None:
            trainer.eval(valid=True, ema=True, verbose=True)
    else:
        trainer.run_epoch(optimizer, scheduler, epoch_st)

    dist.barrier()

    if distenv.master:
        writer.close()
