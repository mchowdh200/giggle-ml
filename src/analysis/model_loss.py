from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
from lightning_fabric import Fabric
from matplotlib import pyplot as plt

from giggleml.train.train_orchestrator import StepResult


def loss_plot(ckpt_dir: Path):
    paths = list(ckpt_dir.iterdir())
    names = [x.stem for x in paths]
    ckpt_time = lambda name: datetime.strptime(name, "%Y-%m-%d_%H-%M-%S").timestamp()
    times = [ckpt_time(x) for x in names]
    steps = np.argsort(times)

    def get_loss(ckpt):
        state = {"step_stats": tuple()}
        fabric = Fabric()
        fabric.load(ckpt, state)
        results = cast(tuple[StepResult, StepResult], state["step_stats"])
        return results[0].loss, results[1].loss

    loss = np.array([get_loss(x) for x in paths])
    train_loss = loss[:, 0]
    eval_loss = loss[:, 1]

    # cut out the first and last 10 steps because
    # they were done with mine-all strategy which has
    # a loss non-comparable to the rest.
    unsort = np.argsort(steps)
    sub_unsort = unsort
    ticks = steps[sub_unsort] * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(ticks, train_loss[sub_unsort], label="Training Loss")
    ax.plot(ticks, eval_loss[sub_unsort], alpha=0.5, label="Eval Loss")
    ax.set_title("Loss")
    ax.set_xlabel("Step")
    ax.legend()

    ax.axvline(3500, color="black")

    return fig


if __name__ == "__main__":
    fig = loss_plot(Path("modelCkpts/cmodel_2025-12-07/ckpts_mix-mine"))
    fig.savefig(Path("experiments/loss_cmodel_2025-12-07_mix-mine.png"))
