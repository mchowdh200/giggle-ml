import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from analysis.utils import confInt95
from giggleml.dataWrangling.intervalDataset import IntervalDataset
from giggleml.embedGen.embedPipeline import EmbedPipeline
from giggleml.intervalTransformer import IntervalTransformer, Slide
from giggleml.utils.types import MmapF32


def swt(
    pipeline: EmbedPipeline,
    inputs: Sequence[tuple[IntervalDataset, str]],
    outFig: str,
    stepCount: int = 11,
    considerLimit: int = 128,
):
    print("Creating embeddings...")
    # round to nearest multiple
    considerLimit = stepCount * (considerLimit // stepCount)

    bins = defaultdict[int, list[list[float]]](list)

    for inputIdx, (intervals, _) in enumerate(inputs):
        interTran = IntervalTransformer(
            intervals, Slide(stepCount, strideNumber=10), considerLimit
        )
        newDataset = interTran.newDataset
        embedOutPath = "./swtEmbeds.tmp.npy"
        embeds: MmapF32 = pipeline.embed(newDataset, embedOutPath).data
        print(" - Success")

        # Analyze results

        for i in range(len(embeds)):
            j = (i // stepCount) * stepCount  # nearest multiple

            newEmbed = embeds[i]
            originalEmbed = embeds[j]
            embeddingDistance = np.linalg.norm(newEmbed - originalEmbed).item()

            newInterval = newDataset[i]
            originalInterval = newDataset[j]
            originalSize = originalInterval[2] - originalInterval[1]
            # relative gap wrt size
            gap = 10 * round(10 * (abs(newInterval[1] - originalInterval[1]) / originalSize))

            if len(bins[gap]) <= inputIdx:
                bins[gap].append(list())

            bins[gap][inputIdx].append(embeddingDistance)

        os.remove(embedOutPath)
        os.remove(embedOutPath + ".meta")

    # Line graph

    plt.clf()
    plt.title("Sliding Window Test")
    plt.xlabel("Relative Distance")
    plt.ylabel("Euclidean Distance")

    labels = list(bins.keys())
    plt.xticks(labels, [f"{int(t)}%" if i % 2 == 0 else "" for i, t in enumerate(labels)])

    for idx in range(len(inputs)):
        avgs = [np.mean(results[idx]) for results in bins.values()]
        error = [confInt95(results[idx]) for results in bins.values()]

        inputName = inputs[idx][1]
        plt.plot(labels, avgs, label=inputName)

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

    plt.legend()
    plt.show()
    Path(outFig).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outFig, dpi=300)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
