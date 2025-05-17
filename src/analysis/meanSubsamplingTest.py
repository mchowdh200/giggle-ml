"""
"How do embeddings of interval sets vary if we sample randomly?"
"""

import enum
import random
from collections.abc import Sequence
from math import floor

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from analysis.utils import confInt95
from giggleml.dataWrangling.intervalDataset import IntervalDataset, MemoryIntervalDataset
from giggleml.embedGen.embedPipeline import EmbedPipeline

# TODO: 5/16/2025 needs testing


def meanSubsamplingTest(pipeline: EmbedPipeline, beds: Sequence[IntervalDataset]) -> Figure:
    fig, ax = plt.subplots()
    baseEmbeds = list()

    for bed in beds:
        out = f"meanTest.tmp.npy"
        embed = pipeline.embed(bed, out)
        mean = np.mean(embed.data, axis=0)
        embed.delete()
        baseEmbeds.append(mean)

    labels = list()
    points = list()
    error = list()
    samplingRates = [(1.5) ** (-k) for k in range(10)]

    for rate in samplingRates:
        dists = list()

        for i, bed in enumerate(beds):
            contents = np.array(list(iter(bed)))
            size = floor(len(bed) * rate)
            contents = np.random.choice(contents, size).tolist()
            fa = bed.associatedFastaPath
            subsampled = MemoryIntervalDataset(contents, fa)

            embed = pipeline.embed(subsampled, "meanTest.tmp.npy")
            mean = np.mean(embed.data, axis=0)
            embed.delete()

            dist = np.linalg.norm(mean - baseEmbeds[i]).item()
            dists.append(dist)

        labels.append(f"{rate*100}%")

        meanDist = np.mean(dists)
        points.append(meanDist)

        err = confInt95(dists)
        error.append(err)

    ax.plot(labels, points)

    # Confidence intervals
    plt.errorbar(
        labels,
        points,
        yerr=error,
        fmt="o",
        color="black",
        ecolor="lightgray",
        elinewidth=2,
        capsize=0,
    )

    ax.set_title("subsampling error")
    ax.set_ylabel("L2 distance from original")
    ax.set_label("sampling percentage")
    return fig
