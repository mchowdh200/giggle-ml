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

from analysis.utils import confInt95
from giggleml.dataWrangling.intervalDataset import IntervalDataset, MemoryIntervalDataset
from giggleml.embedGen.embedPipeline import EmbedPipeline


def meanSubsamplingTest(pipeline: EmbedPipeline, beds: Sequence[IntervalDataset]) -> Figure:
    np.random.seed()

    matplotlib.use("agg")
    fig, ax = plt.subplots()
    baseEmbeds = list()

    with contextlib.suppress(FileNotFoundError):
        os.remove("meanTest.tmp.npy")

    for bed in beds:
        out = f"meanTest.tmp.npy"
        embed = pipeline.embed(bed, out)
        mean = np.mean(embed.data, axis=0)
        embed.delete()
        baseEmbeds.append(mean)

    points = list()
    error = list()
    samplingRates = [k * 0.01 for k in range(100, 0, -10)]

    for rateIdx, rate in enumerate(samplingRates):
        dists = list()

        for i, bed in enumerate(beds):
            contents = list(iter(bed))
            size = floor(len(bed) * rate)
            contents = random.sample(contents, size)
            fa = bed.associatedFastaPath
            subsampled = MemoryIntervalDataset(contents, fa)

            embed = pipeline.embed(subsampled, "meanTest.tmp.npy")
            mean = np.mean(embed.data, axis=0)
            embed.delete()

            dist = np.linalg.norm(mean - baseEmbeds[i]).item()
            dists.append(dist)

        meanDist = np.mean(dists)
        points.append(meanDist)

        err = confInt95(dists)
        error.append(err)
        print(f"{rateIdx+1}/{len(samplingRates)}\n")

    ax.plot(samplingRates, points)
    labels = [f"{round(100*rate)}%" for rate in samplingRates]
    ax.set_xticks(samplingRates, labels)

    # Results are unreliable if too little samples
    if len(beds) >= 5:
        # Confidence intervals
        plt.errorbar(
            samplingRates,
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
