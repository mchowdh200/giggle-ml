from pathlib import Path

import torch.distributed as dist

from giggleml.models.c_model import CModel
from giggleml.train.train_orchestrator import Finetuner, TrainConfig
from giggleml.utils.torch_utils import launch_fabric


def main():
    launch_fabric()

    conf = TrainConfig(
        "train",
        base_model_dir=Path("modelCkpts", "cmodel_2025-12-07"),
        sprint_steps=7000,
        max_batches=7000,
        validation_freq=100,
        model=CModel("16k", 512, 128, 2, 2),
        mining_strategy="all",
        margin=0.01,
        learning_rate=2e-4,
        pk_ratio=0.5,
        positive_threshold=0.96,
        batch_size=128,
        sampling_rate=0.95,
    )

    dist.barrier()
    ft = Finetuner(conf)
    ft.setup()
    ft.run()


if __name__ == "__main__":
    main()
