import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from analysis.utils import conf_int95
from giggleml.data_wrangling.interval_dataset import IntervalDataset
from giggleml.embed_gen.embed_pipeline import EmbedPipeline
from giggleml.interval_transformer import IntervalTransformer
from giggleml.interval_transforms import Slide
from giggleml.utils.types import MmapF32


def swt(
    pipeline: EmbedPipeline,
    inputs: Sequence[tuple[IntervalDataset, str]],
    out_fig: str,
    step_count: int = 11,
    consider_limit: int = 128,
):
    print("Creating embeddings...")
    # round to nearest multiple
    consider_limit = step_count * (consider_limit // step_count)

    bins = defaultdict[int, list[list[float]]](list)

    for input_idx, (intervals, _) in enumerate(inputs):
        inter_tran = IntervalTransformer(
            intervals, Slide(step_count, stride_number=10), consider_limit
        )
        new_dataset = inter_tran.new_dataset
        embed_out_path = "./swtEmbeds.tmp.npy"
        embeds: MmapF32 = pipeline.embed(new_dataset, embed_out_path).data
        print(" - Success")

        # Analyze results

        for i in range(len(embeds)):
            j = (i // step_count) * step_count  # nearest multiple

            new_embed = embeds[i]
            original_embed = embeds[j]
            embedding_distance = np.linalg.norm(new_embed - original_embed).item()

            new_interval = new_dataset[i]
            original_interval = new_dataset[j]
            original_size = original_interval[2] - original_interval[1]
            # relative gap wrt size
            gap = 10 * round(10 * (abs(new_interval[1] - original_interval[1]) / original_size))

            if len(bins[gap]) <= input_idx:
                bins[gap].append(list())

            bins[gap][input_idx].append(embedding_distance)

        os.remove(embed_out_path)
        os.remove(embed_out_path + ".meta")

    # Line graph

    plt.clf()
    plt.title("Sliding Window Test")
    plt.xlabel("Relative Distance")
    plt.ylabel("Euclidean Distance")

    labels = list(bins.keys())
    plt.xticks(labels, [f"{int(t)}%" if i % 2 == 0 else "" for i, t in enumerate(labels)])

    for idx in range(len(inputs)):
        avgs = [np.mean(results[idx]) for results in bins.values()]
        error = [conf_int95(results[idx]) for results in bins.values()]

        input_name = inputs[idx][1]
        plt.plot(labels, avgs, label=input_name)

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
    Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=300)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
