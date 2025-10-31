import argparse
import os
from collections.abc import Iterable
from pathlib import Path
from random import random

import matplotlib.pyplot as plt
import numpy as np

from giggleml.data_wrangling.interval_dataset import MemoryIntervalDataset
from giggleml.embed_gen.embed_pipeline import DirectPipeline
from giggleml.models.hyena_dna import HyenaDNA


def swt(
    fastas: list[os.PathLike],
    out_path: os.PathLike,
    ticks: list[int],
    interval_size: int = 1000,
    trials: int = 50,
    log_xscale: bool = True,
):
    def _trial(fasta: Path):
        def _samples():
            chr = "chr1"
            start = int(random() * 1e8)
            yield (chr, start, start + interval_size)  # origin

            for tick in ticks:
                yield (chr, start + tick, start + tick + interval_size)

        samples = list(_samples())
        dataset = MemoryIntervalDataset(samples, fasta)
        embeds = DirectPipeline(HyenaDNA("1k"), 5, 10).embed(dataset).detach().numpy()
        diffs = [np.linalg.norm(item - embeds[0]) for item in embeds]
        return diffs[1:]  # skip the extra item we added

    def _mean(items: Iterable[float]) -> float:
        items_list = list(items)
        return sum(items_list) / len(items_list) if items_list else 0.0

    def _err(items: Iterable[float]) -> float:
        items_list = list(items)
        return (
            # *2 due to matplotlib
            2 * 1.96 * np.std(items_list) / np.sqrt(len(items_list))
            if items_list
            else 0.0
        )

    def _results(fasta: Path) -> tuple[list[int], list[float], list[float]]:
        samples = [_trial(fasta) for _ in range(trials)]
        averages = [_mean(y) for y in zip(*samples)]
        stderrs = [_err(y) for y in zip(*samples)]
        return ticks, averages, stderrs

    out_path = Path(out_path)
    data = [_results(Path(fasta)) for fasta in fastas]

    plt.figure(figsize=(10, 6))

    for fasta, (ticks, averages, errs) in zip(fastas, data):
        plt.errorbar(
            ticks,
            averages,
            yerr=errs,
            fmt="o-",
            label=Path(fasta).name,
            capsize=3,
            capthick=1,
        )

    if log_xscale:
        plt.xscale("log")
    plt.xlabel("Distance (bp)")
    plt.ylabel("Average Embedding Difference")
    plt.title("Embedding Similarity vs Distance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def long_range(
    fastas: list[os.PathLike],
    out_path: os.PathLike,
    trials: int = 50,
):
    ticks = [2**x for x in range(28)]
    swt(fastas, out_path, ticks, trials=trials)


def short_range(
    fastas: list[os.PathLike],
    out_path: os.PathLike,
    trials: int = 50,
):
    interval_size = 100
    ticks = [
        x for x in range(-interval_size * 2, interval_size * 2 + 1, interval_size // 10)
    ]
    swt(
        fastas,
        out_path,
        ticks,
        interval_size=interval_size,
        trials=trials,
        log_xscale=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Sliding Window Test")
    parser.add_argument("fastas", nargs="+", help="Path(s) to FASTA files to analyze")
    parser.add_argument(
        "-o", "--output", required=True, help="Output path for the plot"
    )
    parser.add_argument(
        "-t",
        "--trials",
        type=int,
        default=50,
        help="Number of trials to run (default: 50)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["short", "long"],
        default="short",
        help="Analysis mode: 'short' for short-range or 'long' for long-range (default: short)",
    )

    args = parser.parse_args()

    if args.mode == "short":
        short_range(args.fastas, args.output, args.trials)
    else:
        long_range(args.fastas, args.output, args.trials)

    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
