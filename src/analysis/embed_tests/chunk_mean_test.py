from collections.abc import Callable, Iterable, Sequence
from random import random

import numpy as np
from matplotlib import pyplot as plt

from giggleml.data_wrangling.interval_dataset import MemoryIntervalDataset
from giggleml.embed_gen.embed_pipeline import DirectPipeline
from giggleml.interval_transforms import ChunkMax, Split
from giggleml.models.caduceus import Caduceus
from giggleml.utils.types import GenomicInterval, PathLike

type Transform = Callable[[GenomicInterval], Iterable[GenomicInterval]]


def _chunk_mean_test(
    ax,
    fasta: PathLike,
    origin_size: int,
    inputs: Sequence[tuple[float, Transform]],
    trials: int = 50,
):
    def trial(transform: Transform) -> float:
        start = int(random() * 2e8)
        origin_interval = ("chr1", start, start + origin_size)
        chunks = transform(origin_interval)
        dataset = MemoryIntervalDataset([origin_interval, *chunks], fasta)
        embeds = (
            DirectPipeline(Caduceus("131k"), 32, sub_workers=4).embed(dataset).cpu()
        )
        chunk_mean = embeds[1:].mean(dim=0)
        diff_from_original = (embeds[0] - chunk_mean).norm().item()
        return diff_from_original

    def do_trials(fn: Transform):
        data = np.array([trial(fn) for _ in range(trials)])
        mean = data.mean().item()
        # multiply two due to matplotlib
        err = 2 * 1.96 * data.std() / np.sqrt(len(data))
        return mean, err

    ticks, transforms = zip(*inputs)
    points, error = zip(*[do_trials(fn) for fn in transforms])
    ax.errorbar(
        ticks,
        points,
        yerr=error,
        fmt="o-",
        capsize=3,
        capthick=1,
    )
    return ax


def chunk_combo_plot(fasta: PathLike, base_interval_size: int, trials: int):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)

    ax = axes[0]
    step = base_interval_size // 10
    ticks = [i for i in range(step, base_interval_size + step, step)]
    tests = [(tick, ChunkMax(tick)) for tick in ticks]
    _chunk_mean_test(ax, fasta, base_interval_size, tests, trials)
    ax.invert_xaxis()
    ax.set_xlabel("Biggest Chunk")
    ax.set_title("Chunk up to Value")

    ax = axes[1]
    counts = [1, 2, 3, 5, 8, 13]
    ticks = [round(base_interval_size / i) for i in counts]
    tests = [(size, Split(count)) for count, size in zip(counts, ticks)]
    _chunk_mean_test(ax, fasta, base_interval_size, tests, trials)
    ax.set_xticks(ticks=ticks, labels=counts)
    ax.invert_xaxis()
    ax.set_xlabel("Chunk Amount")
    ax.set_title("Symmetric Chunking")

    fig.suptitle(
        f"Embedding Error due to Chunking Strategy\nInterval size: {base_interval_size}"
    )
    fig.supylabel("Mean L2-dist to original")
    return fig


def main():
    fig = chunk_combo_plot("data/hg/hg19.fa", 1000, 30)
    fig.savefig("experiments/chunk_combo-cad.png", dpi=300)
    fig.show()


if __name__ == "__main__":
    main()
