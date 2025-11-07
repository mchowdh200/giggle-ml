import torch

from giggleml.train.train_orchestrator import Finetuner, TrainConfig
from giggleml.utils.torch_utils import rprint


def main():
    conf = TrainConfig(
        "val",
        3,
        3,
        model_size="16k",
        margin=3,
        learning_rate=1e-7,
        batch_size=16,
        pk_ratio=1.5,
        positive_threshold=0.96,
        dex_batch_size=32,
        dex_sub_workers=2,
    )

    torch.autograd.set_detect_anomaly(True)
    ft = Finetuner(conf)
    ft.setup()
    loss = ft.run()

    rprint()
    rprint("final validation loss:", loss)


if __name__ == "__main__":
    main()
