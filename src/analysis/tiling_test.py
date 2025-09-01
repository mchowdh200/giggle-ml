import contextlib
import enum
import os
from collections.abc import Iterable, Sequence

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from analysis.utils import conf_int95
from giggleml.data_wrangling.interval_dataset import IntervalDataset
from giggleml.embed_gen.embed_pipeline import EmbedPipeline
from giggleml.interval_transformer import IntervalTransformer
from giggleml.interval_transforms import Tiling
from giggleml.utils.print_to import print_to


def trial(pipeline: EmbedPipeline, bed: IntervalDataset, tile_sizes: Sequence[int], octaves: int):
    baseline_embed_obj = pipeline.embed(bed, "tilingTest1.tmp.npy")
    baseline = baseline_embed_obj.data.mean(axis=0)
    baseline_embed_obj.delete()

    deltas = list()

    for i, tile_size in enumerate(tile_sizes):
        tiler = Tiling(tile_size, octaves)
        tran = IntervalTransformer(bed, tiler)

        new_bed = tran.new_dataset
        tiled_embed_obj = pipeline.embed(new_bed, "tilingTest2.tmp.npy")
        tiled_embed = (tiled_embed_obj.data * tiler.weights(new_bed)[:, None]).mean(axis=0)
        tiled_embed_obj.delete()

        delta = np.linalg.norm(tiled_embed - baseline)
        deltas.append(delta)
        print(f" - {i+1}/{len(tile_sizes)}")

    return deltas


def tiling_test(pipeline: EmbedPipeline, beds: Sequence[IntervalDataset], octaves: int = 1):
    with contextlib.suppress(FileNotFoundError):
        os.remove("tilingTest1.tmp.npy")
        os.remove("tilingTest2.tmp.npy")
        os.remove("tilingTest1.tmp.npy.meta")
        os.remove("tilingTest2.tmp.npy.meta")

    interval_sizes: list[int] = [interval[2] - interval[1] for bed in beds for interval in bed]
    mean_size = round(np.mean(interval_sizes).item())
    tile_sizes = [size for size in range(mean_size // 10, round(mean_size * 1.2), mean_size // 10)]

    diffs_list = list()

    for i, bed in enumerate(beds):
        diffs_list.append(trial(pipeline, bed, tile_sizes, octaves))
        print(f"{i+1}/{len(beds)}\n")

    diffs = np.array(diffs_list, np.float32)
    points = np.mean(diffs, axis=0)  # shape (len(tileSizes),)
    err = [conf_int95(diffs[:, i]) for i in range(diffs.shape[1])]

    matplotlib.use("agg")
    fig, ax = plt.subplots()

    ax.plot(tile_sizes, points)
    plt.errorbar(
        tile_sizes,
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
    ax.axvline(x=mean_size, linestyle="--", color="gray")
    ax.text(mean_size * 0.75, max(points) * 0.3, "mean interval size", fontsize=12, color="gray")

    return fig
