import dataclasses
import os
from collections.abc import Iterable
from dataclasses import replace
from functools import cached_property
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
    rprint,
    rprint0,
)


class Cache:
    def __init__(self, conf: TrainConfig) -> None:
        self.path: Path = Path("modelCkpts", "hparam_search.pickle")
        self.conf: TrainConfig = conf

    @cached_property
    def cache(self):
        # INFO: output location

        if os.path.isfile(self.path):
            return path_pickle.unpickle(self.path)
        else:
            return dict()

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
                    losses, active_triplets = zip(*ft.run())
                    t1 = time()
                    dist.barrier()
                    rprint0()

                if get_rank() == 0:
                    percent_active_triplets = (
                        np.array(active_triplets) / conf.corrected_batch_size
                    ).tolist()

                    self.cache[conf_frozen] = {
                        "percent_active_triplets": percent_active_triplets,
                        "active_triplets": active_triplets,
                        "losses": losses,
                        "time": t1 - t0,
                        "world_size": get_world_size(),
                        "config": conf_frozen,
                    }
                else:
                    # only rank zero will save
                    self.cache[conf_frozen] = None

            if get_rank() == 0:
                result = self.cache[conf_frozen].copy()
                del result["config"]
                rprint(diff)
                rprint(result)

        return [do_one(diff) for diff in diffs]

    def save(self):
        if get_rank() == 0:
            path_pickle.pickle(self.path, self.cache)


def main():
    launch_fabric()

    conf = TrainConfig(
        "val",
        total_steps=3,
        validation_freq=1,
        model=CModel("16k"),
        margin=3,
        learning_rate=1e-7,
        pk_ratio=1.5,
        positive_threshold=0.96,
        batch_size=128,
        density=32,
        dex_batch_size=85,
        dex_sub_workers=0,
    )

    cache = Cache(conf)
    do = cache.do

    try:
        # rprint0("\n model shape")
        # do(
        #     {
        #         "batch_size": 32,
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

        rprint0("\n margin, batch, pkr")
        do(
            {
                "batch_size": 64,
                "margin": margin,
                "pk_ratio": pk_ratio,
                "model": CModel("16k", 512, 128, 1, 2),
            }
            for margin in [0.5, 1, 2, 3, 4]
            for pk_ratio in [0.25, 0.5, 16, 32]
        )
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
        cache.save()


if __name__ == "__main__":
    main()
