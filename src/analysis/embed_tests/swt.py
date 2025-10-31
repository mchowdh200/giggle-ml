import argparse
import os
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from random import random

import matplotlib.pyplot as plt
import numpy as np

from giggleml.data_wrangling.interval_dataset import MemoryIntervalDataset
from giggleml.embed_gen.embed_pipeline import DirectPipeline
from giggleml.models.hyena_dna import HyenaDNA
from giggleml.utils.types import GenomicInterval


def _mean(items: Iterable[float]) -> float:
    items_list = list(items)
    return sum(items_list) / len(items_list) if items_list else 0.0


def _err(items: Iterable[float]) -> float:
    items_list = list(items)
    return (
        # *2 due to matplotlib
        2 * 1.96 * np.std(items_list) / np.sqrt(len(items_list)) if items_list else 0.0
    )


def _samples(ticks: list[int], interval_size: int):
    chr = "chr1"
    start = int(random() * 1e8)
    yield (chr, start, start + interval_size)  # origin

    for tick in ticks:
        yield (chr, start + tick, start + tick + interval_size)


def overlap_plot(
    ticks: list[int],
    interval_size: int = 1000,
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
            samples = list(_samples(ticks, interval_size))
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

    plt.figure(figsize=(10, 6))

    plt.errorbar(
        ticks,
        averages,
        yerr=errs,
        fmt="o-",
        capsize=3,
        capthick=1,
    )

    plt.xlabel("Distance (bp)")
    plt.ylabel("Overlap")
    plt.title("(Bedtools) overlap")
    plt.grid(True, alpha=0.3)
    return plt


def swt(
    fastas: list[os.PathLike],
    ticks: list[int],
    interval_size: int = 1000,
    trials: int = 50,
    log_xscale: bool = True,
):
    def _trial(fasta: Path):
        samples = list(_samples(ticks, interval_size))
        dataset = MemoryIntervalDataset(samples, fasta)
        embeds = DirectPipeline(HyenaDNA("1k"), 5, 10).embed(dataset).detach().numpy()
        diffs = [np.linalg.norm(item - embeds[0]) for item in embeds]
        return diffs[1:]  # skip the extra item we added

    def _results(fasta: Path) -> tuple[list[int], list[float], list[float]]:
        samples = [_trial(fasta) for _ in range(trials)]
        averages = [_mean(y) for y in zip(*samples)]
        stderrs = [_err(y) for y in zip(*samples)]
        return ticks, averages, stderrs

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
    plt.ylabel("Average Embedding Error")
    plt.title("Embedding Similarity vs Distance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt


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
        choices=["short", "long", "overlap"],
        default="short",
        help="Analysis mode: 'short' for short-range or 'long' for long-range (default: short)",
    )

    args = parser.parse_args()

    interval_size = 100
    long_ticks = [2**x for x in range(28)]
    short_ticks = [
        x for x in range(-interval_size * 2, interval_size * 2 + 1, interval_size // 10)
    ]

    if args.mode == "short":
        plt = swt(
            args.fastas,
            short_ticks,
            interval_size,
            args.trials,
            log_xscale=False,
        )
        plt.title("Short range")
    elif args.mode == "long":
        plt = swt(
            args.fastas,
            long_ticks,
            interval_size,
            args.trials,
        )
        plt.title("Long range")
    elif args.mode == "overlap":
        plt = overlap_plot(short_ticks, interval_size, args.trials)
    else:
        raise RuntimeError("unknown mode")

    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
