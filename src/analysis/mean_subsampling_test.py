"""
"How do embeddings of interval sets vary if we sample randomly?"
"""

import contextlib
import enum
import os
import random
from collections.abc import Sequence
from math import floor

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from analysis.utils import conf_int95
from giggleml.data_wrangling.interval_dataset import IntervalDataset, MemoryIntervalDataset
from giggleml.embed_gen.embed_pipeline import EmbedPipeline


def mean_subsampling_test(pipeline: EmbedPipeline, beds: Sequence[IntervalDataset]) -> Figure:
    np.random.seed()

    matplotlib.use("agg")
    fig, ax = plt.subplots()
    base_embeds = list()

    with contextlib.suppress(FileNotFoundError):
        os.remove("meanTest.tmp.npy")

    for bed in beds:
        out = f"meanTest.tmp.npy"
        embed = pipeline.embed(bed, out)
        mean = np.mean(embed.data, axis=0)
        embed.delete()
        base_embeds.append(mean)

    points = list()
    error = list()
    sampling_rates = [k * 0.01 for k in range(100, 0, -10)]

    for rate_idx, rate in enumerate(sampling_rates):
        dists = list()

        for i, bed in enumerate(beds):
            contents = list(iter(bed))
            size = floor(len(bed) * rate)
            contents = random.sample(contents, size)
            fa = bed.associated_fasta_path
            subsampled = MemoryIntervalDataset(contents, fa)

            embed = pipeline.embed(subsampled, "meanTest.tmp.npy")
            mean = np.mean(embed.data, axis=0)
            embed.delete()

            dist = np.linalg.norm(mean - base_embeds[i]).item()
            dists.append(dist)

        mean_dist = np.mean(dists)
        points.append(mean_dist)

        err = conf_int95(dists)
        error.append(err)
        print(f"{rate_idx+1}/{len(sampling_rates)}\n")

    ax.plot(sampling_rates, points)
    labels = [f"{round(100*rate)}%" for rate in sampling_rates]
    ax.set_xticks(sampling_rates, labels)

    # Results are unreliable if too little samples
    if len(beds) >= 5:
        # Confidence intervals
        plt.errorbar(
            sampling_rates,
            points,
            yerr=error,
            fmt="o",
            color="black",
            ecolor="lightgray",
            elinewidth=2,
            capsize=0,
        )

    ax.set_title("Subsampling Error")
    ax.set_ylabel("L2 distance from original")
    ax.set_xlabel("subsampling rate")
    return fig
