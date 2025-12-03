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

                print("max trips", max_triplets)

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
        base_model_dir=Path("modelCkpts", "cmodel_12022025"),
        total_steps=10,
        validation_freq=2,
        model=CModel("16k", 512, 128, 2, 2),
        margin=3,
        learning_rate=1e-7,
        pk_ratio=1.5,
        positive_threshold=0.96,
        batch_size=128,
        sampling_rate=0.95,
        dex_batch_size=int(1e6),
        dex_sub_workers=0,
    )

    dist.barrier()
    cache = Cache(Path("modelCkpts", "hparam_search.pickle"), conf)
    do = cache.do

    try:
        header("pkr")
        do(
            {
                "pk_ratio": pkr,
                "mining_strategy": "all",
                "margin": 0.1,
            }
            for pkr in [16, 8, 2, 0.5]
        )

        # header("")
        # do(
        #     {
        #         "batch_size": 16,
        #         "density": density,
        #         "pk_ratio": pkr,
        #         "mining_strategy": "all",
        #         "margin": 0.1,
        #     }
        #     for pkr in [8]
        #     for density in [32, 64, 128]
        # )

        # do(
        #     {
        #         "batch_size": batch_size,
        #         "pk_ratio": pk_ratio,
        #         "mining_strategy": "all",
        #         "margin": 0.1,
        #     }
        #     for batch_size in [32, 64]
        #     for pk_ratio in [1 / 16, 1 / 8, 1 / 2, 2, 8, 32]
        # )

        # header("batch, pkr")
        # do(
        #     {
        #         "batch_size": batch_size,
        #         "pk_ratio": pk_ratio,
        #         "margin": 0.1,
        #     }
        #     for batch_size in [32, 64, 128]
        #     for pk_ratio in [1 / 16, 1 / 8, 1 / 2, 2, 8, 32]
        # )

        # rprint0("\n model shape")
        # do(
        #     {
        #         "batch_size": 64,
        #         "model": CModel("16k", *args),
        #     }
        #     for args in [
        #         # quite large
        #         (1024, 128, 2, 2),
        #         (1024, 128, 1, 2),
        #         (1024, 128, 2, 1),
        #         (1024, 128, 1, 1),
        #         # large
        #         (512, 128, 2, 2),
        #         (512, 128, 1, 2),
        #         (512, 128, 2, 1),
        #         (512, 128, 1, 1),
        #         # medium
        #         (256, 128, 2, 2),
        #         (256, 128, 1, 2),
        #         (256, 128, 2, 1),
        #         (256, 128, 1, 1),
        #         # small
        #         (128, 128, 2, 2),
        #         (128, 128, 1, 2),
        #         (128, 128, 2, 1),
        #         (128, 128, 1, 1),
        #     ]
        # )

        # rprint0("\n LR + model shape")
        # do(
        #     {
        #         "batch_size": 64,
        #         "total_steps": 4,
        #         "learning_rate": lr,
        #         "model": CModel("16k", *args),
        #     }
        #     for lr in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        #     for args in [
        #         # quite large
        #         (1024, 128, 2, 2),
        #         (1024, 128, 1, 2),
        #         (1024, 128, 2, 1),
        #         # large
        #         (512, 128, 2, 2),
        #         (512, 128, 1, 2),
        #         (512, 128, 2, 1),
        #     ]
        # )

        # do(
        #     {
        #         "batch_size": batch_size,
        #         "margin": margin,
        #         "pk_ratio": pk_ratio,
        #         "total_steps": 3,
        #         "model": CModel("16k", 512, 128, 1, 2),
        #     }
        #     for batch_size in [32, 64, 128, 256]
        #     for margin in [0.5, 1, 2, 3, 4]
        #     for pk_ratio in [0.25, 0.5, 16, 32, 64, 128, 256]
        #     if pk_ratio < batch_size
        # )
    finally:
        dist.barrier()
        cache.save()


if __name__ == "__main__":
    main()
