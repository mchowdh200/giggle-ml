from collections.abc import Iterable, Sequence

import numpy as np
from matplotlib import pyplot as plt

from giggleml.dataWrangling.intervalDataset import IntervalDataset
from giggleml.embedGen.embedPipeline import EmbedPipeline
from giggleml.intervalTransformer import IntervalTransformer
from giggleml.intervalTransforms import Tiling


def trial(pipeline: EmbedPipeline, bed: IntervalDataset, tileSizes: Iterable[int]):
    baselineEmbedObj = pipeline.embed(bed, "tilingTest1.tmp.npy")
    baseline = baselineEmbedObj.data.mean(axis=0)
    baselineEmbedObj.delete()

    deltas = list()

    for tileSize in tileSizes:
        tiler = Tiling(tileSize)
        tran = IntervalTransformer(bed, tiler)

        tiledEmbedObj = pipeline.embed(tran.newDataset, "tilingTest2.tmp.npy")
        tiledEmbed = tiledEmbedObj.data.mean(axis=0)
        tiledEmbedObj.delete()

        delta = np.linalg.norm(tiledEmbed - baseline)
        deltas.append(delta)

    return deltas


def tilingTest(pipeline: EmbedPipeline, beds: Sequence[IntervalDataset]):
    intervalSizes: list[int] = [interval[2] - interval[1] for bed in beds for interval in bed]
    meanSize = round(np.mean(intervalSizes).item())
    tileSizes = [size for size in range(meanSize // 2, meanSize * 2, meanSize // 10)]
    diffs = np.array([trial(pipeline, bed, tileSizes) for bed in beds])

    points = np.mean(diffs, axis=0)  # shape (len(tileSizes),)
    fig, ax = plt.subplots()

    ax.plot(tileSizes, points)
    ax.set_title("Error Due to Tiling")
    ax.set_ylabel("L2 distance from original")
    ax.set_xlabel("Tiling size")

    ax.axvline(x=meanSize, linestyle="--", color="gray")
    ax.text(meanSize + 1.1, max(points) * 0.9, "mean interval size", fontsize=12, color="gray")

    return fig
