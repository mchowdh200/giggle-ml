from giggleml.train.train_orchestrator import Finetuner, TrainConfig
from giggleml.utils.time_this import time_this
from giggleml.utils.torch_utils import get_rank, rprint


def main():
    conf = TrainConfig(
        "val",
        total_steps=3,
        validation_freq=3,
        model_size="16k",
        margin=3,
        learning_rate=1e-7,
        pk_ratio=1.5,
        positive_threshold=0.96,
        batch_size=128,
        density=32,
        dex_batch_size=85,
        dex_sub_workers=0,
    )

    with time_this("training"):
        ft = Finetuner(conf)
        ft.setup()
        loss = ft.run()

    if get_rank() == 0:
        rprint()
        rprint("final validation loss:", loss)

    # 2 gpus, 5m
    # conf = TrainConfig(
    #     "val",
    #     total_steps=6,
    #     validation_freq=3,
    #     model_size="16k",
    #     margin=3,
    #     learning_rate=1e-7,
    #     pk_ratio=1.5,
    #     positive_threshold=0.96,
    #     batch_size=128,
    #     density=30,
    #     dex_batch_size=85,
    #     dex_sub_workers=8,
    # )

    # 2 gpus, 3m
    # conf = TrainConfig(
    #     ...
    #     dex_sub_workers=2,
    # )

    # 2 gpus, 116s
    # conf = TrainConfig(
    #     ...
    #     dex_sub_workers=0,
    # )


if __name__ == "__main__":
    main()
