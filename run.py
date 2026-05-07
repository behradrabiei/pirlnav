#!/usr/bin/env python3
import argparse
import random

import numpy as np
import numba
import quaternion
import torch
import habitat

from habitat import logger
from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry

from pirlnav.config import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def execute_exp(config: Config, run_type: str) -> None:
    r"""Run the experiment with a reproducible seed.

    ``config.TASK_CONFIG.SEED`` is used as-is (defaulting to ``100`` via
    habitat-lab's default config). Every launch with the same config is
    therefore bit-deterministic. To vary across runs, pass
    ``TASK_CONFIG.SEED <n>`` on the CLI or set it in the YAML.
    """
    seed = int(config.TASK_CONFIG.SEED)
    logger.info("Using TASK_CONFIG.SEED {}".format(seed))
    config.defrost()
    config.RUN_TYPE = run_type
    config.freeze()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if config.FORCE_TORCH_SINGLE_THREADED and torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()


def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    execute_exp(config, run_type)


if __name__ == "__main__":
    main()
