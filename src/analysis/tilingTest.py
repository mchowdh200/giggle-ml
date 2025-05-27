import contextlib
import enum
import os
from collections.abc import Iterable, Sequence

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from analysis.utils import confInt95
from giggleml.dataWrangling.intervalDataset import IntervalDataset
from giggleml.embedGen.embedPipeline import EmbedPipeline
from giggleml.intervalTransformer import IntervalTransformer
from giggleml.intervalTransforms import Tiling
from giggleml.utils.printTo import printTo


def trial(pipeline: EmbedPipeline, bed: IntervalDataset, tileSizes: Sequence[int], octaves: int):
    baselineEmbedObj = pipeline.embed(bed, "tilingTest1.tmp.npy")
    baseline = baselineEmbedObj.data.mean(axis=0)
    baselineEmbedObj.delete()

    deltas = list()

    for i, tileSize in enumerate(tileSizes):
        tiler = Tiling(tileSize, octaves)
        tran = IntervalTransformer(bed, tiler)

        newBed = tran.newDataset
        tiledEmbedObj = pipeline.embed(newBed, "tilingTest2.tmp.npy")
        tiledEmbed = (tiledEmbedObj.data * tiler.weights(newBed)[:, None]).mean(axis=0)
        tiledEmbedObj.delete()

        delta = np.linalg.norm(tiledEmbed - baseline)
        deltas.append(delta)
        print(f" - {i+1}/{len(tileSizes)}")

    return deltas


def tilingTest(pipeline: EmbedPipeline, beds: Sequence[IntervalDataset], octaves: int = 1):
    with contextlib.suppress(FileNotFoundError):
        os.remove("tilingTest1.tmp.npy")
        os.remove("tilingTest2.tmp.npy")
        os.remove("tilingTest1.tmp.npy.meta")
        os.remove("tilingTest2.tmp.npy.meta")

    intervalSizes: list[int] = [interval[2] - interval[1] for bed in beds for interval in bed]
    meanSize = round(np.mean(intervalSizes).item())
    tileSizes = [size for size in range(meanSize // 10, round(meanSize * 1.2), meanSize // 10)]

    diffsList = list()

    for i, bed in enumerate(beds):
        diffsList.append(trial(pipeline, bed, tileSizes, octaves))
        print(f"{i+1}/{len(beds)}\n")

    diffs = np.array(diffsList, np.float32)
    points = np.mean(diffs, axis=0)  # shape (len(tileSizes),)
    err = [confInt95(diffs[:, i]) for i in range(diffs.shape[1])]

    matplotlib.use("agg")
    fig, ax = plt.subplots()

    ax.plot(tileSizes, points)
    plt.errorbar(
        tileSizes,
        points,
        yerr=err,
        fmt="o",
        color="black",
        ecolor="lightgray",
        elinewidth=2,
        capsize=0,
    )

    ax.set_title("Error Due to Tiling")
    ax.set_ylabel("L2 distance from original")
    ax.set_xlabel(f"Tiling size ({octaves} octaves)")
    ax.axvline(x=meanSize, linestyle="--", color="gray")
    ax.text(meanSize * 0.75, max(points) * 0.3, "mean interval size", fontsize=12, color="gray")

    return fig
