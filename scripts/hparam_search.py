import dataclasses
import os
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import torch.distributed as dist

from giggleml.models.c_model import CModel
from giggleml.train.train_orchestrator import Finetuner, TrainConfig
from giggleml.utils import path_pickle
from giggleml.utils.print_utils import indent_prints
from giggleml.utils.torch_utils import (
    get_rank,
    get_world_size,
    launch_fabric,
    rprint0,
)


class Cache:
    def __init__(self, path: Path, conf: TrainConfig) -> None:
        self.path: Path = path
        self.conf: TrainConfig = conf

        if os.path.isfile(self.path):
            self.cache: dict[frozenset, Any] = path_pickle.unpickle(self.path)
        else:
            self.cache = dict()

    def do(self, diffs: Iterable[dict[str, Any]]):
        def do_one(diff: dict[str, Any]):
            conf = replace(self.conf, **diff)
            conf_dict = dataclasses.asdict(conf)
            conf_dict["model"] = repr(conf.model)
            conf_dict["cv_ratios"] = frozenset(conf_dict["cv_ratios"].items())
            conf_frozen = frozenset(conf_dict.items())

            if conf_frozen not in self.cache:
                with indent_prints(indent=4):
                    t0 = time()
                    ft = Finetuner(conf)
                    ft.setup()
                    results = ft.run()
                    t1 = time()
                    dist.barrier()
                    rprint0()

                if get_rank() == 0:
                    self.cache[conf_frozen] = (
                        results,
                        t1 - t0,
                        get_world_size(),
                    )
                else:
                    # note, only rank zero will save
                    self.cache[conf_frozen] = None

            # report
            if get_rank() == 0:
                results, duration, world_size = self.cache[conf_frozen]
                train, eval = zip(*results)
                train_loss, train_triplets, train_max_triplets = zip(*train)
                eval_loss, eval_triplets, eval_max_triplets = zip(*eval)

                active_triplets = np.array([train_triplets, eval_triplets])
                max_triplets = np.array([train_max_triplets, eval_max_triplets])
                percent_active_triplets = (
                    active_triplets / max_triplets.round(decimals=3).tolist()
                )

                print(diff)
                print("eval")
                with indent_prints(indent=2):
                    print("eval_triplet_rate:", percent_active_triplets[1])
                    print("eval_loss:", eval_loss)
                print("train")
                with indent_prints(indent=2):
                    print("train_triplet_rate:", percent_active_triplets[0])
                    print("train_loss:", train_loss)
                print("duration:", duration, "world_size:", world_size, "\n")

        return [do_one(diff) for diff in diffs]

    def save(self):
        if get_rank() == 0:
            path_pickle.pickle(self.path, self.cache)


def header(*args, **kwargs):
    rprint0()
    rprint0("-" * 30)
    rprint0(*args, **kwargs)
    rprint0("-" * 30)
    rprint0()


def main():
    launch_fabric()

    conf = TrainConfig(
        "val",
        base_model_dir=Path("modelCkpts", "cmodel_2025-12-05"),
        sprint_steps=10,
        validation_freq=2,
        model=CModel("16k", 512, 128, 2, 2),
        margin=3,
        learning_rate=1e-7,
        pk_ratio=1.5,
        positive_threshold=0.96,
        batch_size=64,
        sampling_rate=1,
    )

    dist.barrier()
    cache = Cache(Path("modelCkpts", "hparam_search.pickle"), conf)
    do = cache.do

    try:
        header("easy")
        do(
            {
                "margin": 0.1,
                "pk_ratio": pkr,
                "sampling_rate": 1,
                "learning_rate": lr,
                "batch_size": 128,
                "sprint_steps": 10,
                "mining_strategy": "semi-hard",
            }
            for lr in [1e-3, 1e-2]
            for pkr in [0.5, 0.25]
        )
        do(
            [
                {
                    "margin": 0.1,
                    "pk_ratio": 0.5,
                    "sampling_rate": 1,
                    "learning_rate": 1e-4,
                    "batch_size": 128,
                    "sprint_steps": 20,
                    "mining_strategy": "semi-hard",
                },
                {
                    "margin": 0.01,
                    "pk_ratio": 0.5,
                    "sampling_rate": 1,
                    "learning_rate": 1e-4,
                    "batch_size": 128,
                    "sprint_steps": 20,
                    "mining_strategy": "semi-hard",
                },
            ]
        )

        header("pkr")
        do(
            {
                "margin": 0.01,
                "pk_ratio": pkr,
                "sampling_rate": 1,
                "learning_rate": 1e-4,
                "batch_size": 128,
                "sprint_steps": 10,
                "mining_strategy": "semi-hard",
            }
            for pkr in [16, 8, 2, 0.5, 0.25]
        )

    finally:
        dist.barrier()
        cache.save()


if __name__ == "__main__":
    main()
