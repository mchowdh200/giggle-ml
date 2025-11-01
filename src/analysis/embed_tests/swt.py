import os
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from random import random

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from giggleml.data_wrangling.interval_dataset import MemoryIntervalDataset
from giggleml.embed_gen.embed_pipeline import DirectPipeline
from giggleml.models.hyena_dna import HyenaDNA
from giggleml.utils.types import GenomicInterval, PathLike


def _mean(items: Iterable[float]) -> float:
    items_list = list(items)
    return sum(items_list) / len(items_list) if items_list else 0.0


def _err(items: Iterable[float]) -> float:
    items_list = list(items)
    return (
        # *2 due to matplotlib
        2 * 1.96 * np.std(items_list) / np.sqrt(len(items_list)) if items_list else 0.0
    )


def _samples(ticks: list[int], origin_size: int, others_size: int | None = None):
    others_size = others_size or origin_size
    chr = "chr1"
    midpoint = int(random() * 1e8)
    yield (chr, midpoint - origin_size // 2, midpoint + origin_size // 2)  # origin

    for tick in ticks:
        yield (
            chr,
            midpoint + tick - others_size // 2,
            midpoint + tick + others_size // 2,
        )


def _dist(x: NDArray, y: NDArray) -> float:
    # cosine_similarity
    # return np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y)

    # euclidean
    return np.linalg.norm(y - x).item()


def overlap_plot(
    ax,
    ticks: list[int],
    interval_size: tuple[int, int],
    trials: int = 50,
):
    def _write_bed(origin: GenomicInterval, others: Sequence[GenomicInterval]):
        with tempfile.NamedTemporaryFile("w", delete=False) as file:
            origin_terms = [str(x) for x in origin]
            origin_str = "\t".join(origin_terms)

            for interval in others:
                terms = [str(x) for x in interval]
                file.write(f"{origin_str}\t{'\t'.join(terms)}\n")
            file.close()
            return file.name

    def _trial():
        bed = None

        try:
            samples = list(_samples(ticks, *interval_size))
            origin = samples[0]
            others = samples[1:]
            bed = _write_bed(origin, others)

            result = subprocess.run(
                ["bedtools", "overlap", "-i", bed, "-cols", "2,3,5,6"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout

            return [int(line.split()[6]) for line in result.splitlines()]
        finally:
            if bed:
                os.remove(bed)

    samples = [_trial() for _ in range(trials)]
    averages = [_mean(y) for y in zip(*samples)]
    errs = [_err(y) for y in zip(*samples)]

    ax.errorbar(
        ticks,
        averages,
        yerr=errs,
        fmt="o-",
        capsize=3,
        capthick=1,
    )

    ax.set_xlabel("Distance (bp)")
    ax.set_ylabel("(Bedtools) Overlap")
    ax.set_title("(Bedtools) Overlap")
    ax.grid(True, alpha=0.3)
    return ax


def swt(
    ax,
    fastas: list[PathLike],
    ticks: list[int],
    interval_size: tuple[int, int],
    trials: int = 50,
    log_xscale: bool = False,
):
    def _trial(fasta: Path):
        samples = list(_samples(ticks, *interval_size))
        dataset = MemoryIntervalDataset(samples, fasta)
        embeds = DirectPipeline(HyenaDNA("16k"), 5, 10).embed(dataset).detach().numpy()
        diffs = [_dist(item, embeds[0]) for item in embeds]
        return diffs[1:]  # skip the extra item we added

    def _results(fasta: Path) -> tuple[list[int], list[float], list[float]]:
        samples = [_trial(fasta) for _ in range(trials)]
        averages = [_mean(y) for y in zip(*samples)]
        stderrs = [_err(y) for y in zip(*samples)]
        return ticks, averages, stderrs

    data = [_results(Path(fasta)) for fasta in fastas]

    for fasta, (ticks, averages, errs) in zip(fastas, data):
        ax.errorbar(
            ticks,
            averages,
            yerr=errs,
            fmt="o-",
            label=Path(fasta).name,
            capsize=3,
            capthick=1,
        )

    if log_xscale:
        ax.set_xscale("log")
    ax.set_xlabel("Distance (bp)")
    ax.set_ylabel("Mean L2-dist to Original")
    ax.set_title("Embedding Error vs Displacement")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def all_plot(
    fastas: list[PathLike],
    origin_size: int,
    other_sizes: tuple[int, ...],
    trials: int = 50,
):
    fig, axes = plt.subplots(nrows=2, ncols=len(other_sizes), figsize=(12, 10))

    for i, size in enumerate(other_sizes):
        window = (size + origin_size) // 2
        ticks = list(range(-window, window + 1, window // 10))

        # top row
        ax = overlap_plot(axes[0, i], ticks, (origin_size, size), trials)
        ax.set_title(f"{origin_size} by {size}")

        # bottom row
        ax = swt(axes[1, i], fastas, ticks, (origin_size, size), trials)
        ax.set_title(f"{origin_size} by {size}")
        ax.set_ylim(-1, 6)
        ax.invert_yaxis()

    fig.suptitle("Sliding Window Test")
    plt.tight_layout()
    return fig


def main():
    # combo plot
    fig = all_plot(
        ["data/hg/hg19.fa", "data/hg/synthetic.fa"],
        1000,
        (10, 100, 1000, 10000),
        trials=50,
    )
    fig.savefig("experiments/swt_combo.png", dpi=300)
    fig.show()

    # long
    fig, ax = plt.subplots(figsize=(8, 5))
    long_ticks = [2**x for x in range(28)]
    swt(
        ax,
        ["data/hg/hg19.fa", "data/hg/synthetic.fa"],
        long_ticks,
        (1000, 1000),
        trials=50,
        log_xscale=True,
    )
    fig.savefig("experiments/swt_long_range.png", dpi=300)
    fig.show()


if __name__ == "__main__":
    main()
