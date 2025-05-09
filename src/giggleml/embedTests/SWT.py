import os
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from giggleml.embedGen.embedPipeline import EmbedPipeline
from giggleml.intervalTransformer import IntervalTransformer, Slide
from giggleml.utils.types import MmapF32


def swt(
    pipeline: EmbedPipeline,
    intervals,
    outFig: str,
    stepCount: int = 11,
    considerLimit: int = 128,
):
    print("Creating embeddings...")
    # round to nearest multiple
    considerLimit = stepCount * (considerLimit // stepCount)
    dataset = IntervalTransformer(intervals, Slide(stepCount), considerLimit).newDataset
    embedOutPath = "./swtEmbeds.tmp.npy"
    embeds: MmapF32 = pipeline.embed(dataset, embedOutPath).data
    print(" - Success")

    # Analyze results

    bins: list[Any] = [None] * stepCount

    for i in range(len(dataset)):
        binIdx = i % stepCount
        if bins[binIdx] is None:
            bins[binIdx] = []

        rootIdx = i // stepCount * stepCount  # 0th item of bin
        rootEmbed = embeds[rootIdx]
        thisEmbed = embeds[i]

        # TODO: is this the best metric?
        # dist = cosine_similarity(rootEmbed, thisEmbed)
        dist = np.linalg.norm(rootEmbed - thisEmbed)
        bins[binIdx].append(dist)

    avgs: list[Any] = [None] * stepCount
    error: list[Any] = [None] * stepCount

    for i, bin in enumerate(bins):
        avgs[i] = np.mean(bin)
        std = np.std(bin)
        n = len(bin)
        # 1.96 is the z-score for 95% confidence
        error[i] = 1.96 * std / np.sqrt(n)

    labels: list[Any] = [None] * stepCount
    gapFactor = 1 / (stepCount - 1)

    for i in range(len(avgs)):
        labels[i] = 100 - round(i * gapFactor * 100)
        print(f"Bin {labels[i]}%: {avgs[i]}")

    plt.clf()
    plt.title("Sliding Window Test")
    plt.xlabel("Overlap %")
    plt.ylabel("Euclidean Distance")

    # Line graph
    plt.plot(labels, avgs)
    plt.xticks(labels)
    # Error bars
    plt.errorbar(
        labels,
        avgs,
        yerr=error,
        fmt="o",
        color="black",
        ecolor="lightgray",
        elinewidth=2,
        capsize=0,
    )

    plt.show()
    Path(outFig).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outFig, dpi=300)
    os.remove(embedOutPath)
    os.remove(embedOutPath + ".meta")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
