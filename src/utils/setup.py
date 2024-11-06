# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------

from datetime import datetime
import logging
import inspect
import os
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from .writer import Writer
from .config import config_setup
from .dist import initialize as dist_init


def logger_setup(log_path, eval=False):
    log_fname = os.path.join(log_path, "val.log" if eval else "train.log")

    for hdlr in logging.root.handlers:
        logging.root.removeHandler(hdlr)

    SMOKE_TEST = bool(os.environ.get("SMOKE_TEST", 0))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=(logging.DEBUG if SMOKE_TEST else logging.INFO),
        handlers=[logging.FileHandler(log_fname), logging.StreamHandler()],
    )
    main_filename, *_ = inspect.getframeinfo(inspect.currentframe().f_back.f_back)

    logger = logging.getLogger(Path(main_filename).name)
    writer = Writer(log_path)

    return logger, writer


def setup(args, extra_args=()):
    """
    meaning of args.result_path:
        - if args.eval, directory where the model is
        - if args.resume, no meaning
        - otherwise, path to store the logs

    Returns:
        config, logger, writer
    """

    distenv = dist_init(args)

    args.result_path = Path(args.result_path).absolute().as_posix()
    args.model_config = Path(args.model_config).absolute().resolve().as_posix()

    now = datetime.now().strftime("%d%m%Y_%H%M%S")

    if args.eval:
        config_path = Path(args.result_path).joinpath("config.yaml")
        log_path = Path(args.result_path).joinpath("val", now)

    elif args.resume:
        load_path = Path(args.load_path)
        if not load_path.is_file():
            raise ValueError("load_path must be a valid filename")

        config_path = load_path.parent.joinpath("config.yaml").absolute()
        log_path = load_path.parent.parent.joinpath(now)

    else:
        config_path = Path(args.model_config).absolute()
        task_name = config_path.stem
        if args.postfix:
            task_name += f"__{args.postfix}"
        log_path = Path(args.result_path).joinpath(task_name, now)

    config = config_setup(args, distenv, config_path, extra_args=extra_args)
    config.result_path = log_path.absolute().resolve().as_posix()

    if distenv.master:
        if not log_path.exists():
            os.makedirs(log_path)
        logger, writer = logger_setup(log_path)
        logger.info(distenv)
        logger.info(f"log_path: {log_path}")
        logger.info("\n" + OmegaConf.to_yaml(config))
        OmegaConf.save(config, log_path.joinpath("config.yaml"))

        src_dir = Path(os.getcwd()).joinpath("src")
        shutil.copytree(src_dir, log_path.joinpath("src"))
        logger.info(f"source copied to {log_path}/src")
    else:
        logger, writer, log_path = None, None, None

    return config, logger, writer


def single_setup(args, extra_args=(), train=True):
    assert args.eval
    args.model_config = Path(args.model_config).absolute().resolve().as_posix()
    config_path = args.model_config
    config = config_setup(args, None, config_path, extra_args=extra_args)
    return config
