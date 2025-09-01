import datetime
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from giggleml.interval_transformer import IntervalTransformer
from giggleml.utils.interval_arithmetic import intersect
from giggleml.utils.types import GenomicInterval, MmapF32


class SimpleKNN:
    def __init__(self, embeddings: np.ndarray):
        """
        Initializes the VectorDB_IP with a set of embeddings.

        Args:
            embeddings (np.ndarray): A 2D NumPy array where each row represents
                                       an embedding vector.
        """
        self.embeddings = embeddings

    def search(self, query: np.ndarray, k: int) -> tuple[None, list[np.ndarray]]:
        """
        Performs a k-nearest neighbors search for multiple query embeddings
        based on inner product.

        Args:
            query (np.ndarray): A 2D NumPy array where each row is a
                                            query embedding vector.
            k (int): The number of nearest neighbors to retrieve for each query.

        Returns:
            Tuple[None, List[np.ndarray]]: A tuple containing:
                - None (as per the usage in modernGiggle).
                - knn (List[np.ndarray]): A list of lists, where each inner list
                  contains the indices of the k-nearest neighbors for the
                  corresponding query embedding in query_embeddings.
        """
        knn_indices = []
        for query_embedding in query:
            similarities = np.inner(self.embeddings, query_embedding)
            top_k_indices = np.argsort(similarities)[::-1][:k]
            knn_indices.append(top_k_indices)

        # redundant None just to match this file's usage of faiss.IndexFlatIP
        return None, knn_indices


def overlap_degree(x: GenomicInterval, y: GenomicInterval) -> float:
    """
    100% (1) means the smaller interval is fully contained in the larger interval.
    """

    z = intersect(x, y)

    if z is None:
        return 0

    z_size = z[2] - z[1]
    x_size = x[2] - x[1]
    y_size = y[2] - y[1]
    ref_size = min(x_size, y_size)
    return z_size / ref_size


def intersects(x: GenomicInterval, y: GenomicInterval):
    return overlap_degree(x, y) > 0


def build_vector_db(embeds: MmapF32):
    # dim = embeds.shape[1]
    # vdb = faiss.IndexFlatIP(dim)
    normal = embeds / np.linalg.norm(embeds, axis=1).reshape(-1, 1)
    # vdb.add(normal)
    # return vdb
    return SimpleKNN(normal)


def modern_giggle(
    sample_it: IntervalTransformer,
    sample_embeds: MmapF32,
    query_it: IntervalTransformer,
    query_embeds: MmapF32,
    k: int,
):

    results = defaultdict(set)

    print("Building vector database")
    vdb = build_vector_db(sample_embeds)
    _, knn = vdb.search(query_embeds, k)
    print(" - completed KNN search")

    for query_id, neighbors in enumerate(knn):
        query_base = query_it.old_dataset[query_it.backward_idx(query_id)]
        hits = []

        for neighbor_id in neighbors:
            hit_base = sample_it.old_dataset[sample_it.backward_idx(neighbor_id)]

            if intersects(hit_base, query_base):
                hits.append(hit_base)

        results[query_base].update(hits)

    return results


def parse_legacy_giggle(path: str) -> dict[GenomicInterval, list[GenomicInterval]]:
    ancient_results = dict()
    with open(path, "r") as f:
        profile = list[GenomicInterval]()
        head: GenomicInterval | None = None

        for line in f:
            chr, start, end, *_ = line.strip().split()
            start = int(start)
            end = int(end)

            if chr[0] == "#":
                ancient_results[head] = profile
                profile = []
                chr = chr[2:]
                head = (chr, start, end)
                continue

            profile.append((chr, start, end))

        ancient_results[head] = profile
        ancient_results.pop(None)

    return ancient_results


def analyze_results(modern_results, ancient_results, overlap_plot, ge_overlap_plot):
    for query in modern_results.keys():
        if query not in ancient_results:
            raise RuntimeError("Extra query in modern", query)

    for query in ancient_results.keys():
        if query not in modern_results:
            raise RuntimeError("Missing query in modern", query)

    # core stats

    highest_depth_per_query = max(map(len, ancient_results.values()))
    print("Highest depth per query (ground truth)", highest_depth_per_query)

    print("Modern non zero hits", sum(filter(lambda x: x != 0, map(len, modern_results.values()))))

    hit_count_ancient = sum(map(len, ancient_results.values()))
    hit_count_modern = sum(map(len, modern_results.values()))
    print("Hit Count")
    print(" - ground truth", hit_count_ancient)
    print(" - modern", hit_count_modern)

    # used to make a histogram at 10% intervals
    hits = [0] * 11
    totals = [0] * 11

    for query in ancient_results.keys():
        modern_profile = set(modern_results[query])
        ancient_profile = set(ancient_results[query])

        overlap = len(ancient_profile.intersection(modern_profile))
        assert overlap >= min(len(modern_profile), len(ancient_profile))

        for real_hit in ancient_profile:
            overlap = overlap_degree(real_hit, query)
            discrete_overlap = round(overlap * 10)
            totals[discrete_overlap] += 1

            if real_hit in modern_profile:
                hits[discrete_overlap] += 1

    recall = sum(hits) / hit_count_ancient
    print("recall", recall)

    # recall by overlap

    hit_prob = []
    for i in range(11):
        if totals[i] == 0:
            raise RuntimeError("A total is zero, cannot compute average")
        hit_prob.append(hits[i] / totals[i])

    ticks = np.arange(0, 1.1, 0.1)

    plt.figure()
    plt.bar(ticks, hit_prob, width=0.1)

    plt.xticks(ticks)
    plt.xlim(0, 1)
    plt.yticks(ticks)

    plt.axhline(y=0.5, linestyle="-", color="b")
    plt.xlabel("Overlap")
    plt.ylabel("Recall")
    plt.title("Recall by overlap")
    plt.savefig(overlap_plot, dpi=300)

    # recall by >= overlap

    running_hits = np.array([0] * 11)
    running_totals = np.array([0] * 11)

    for i in range(len(hits)):
        running_hits[i] = sum(hits[i:])
        running_totals[i] = sum(totals[i:])

        if running_totals[i] == 0:
            print("A total is zero, cannot compute average")

    hit_prob_sum = running_hits / running_totals

    plt.figure()
    plt.bar(ticks, hit_prob_sum, width=0.1)

    plt.xticks(ticks)
    plt.xlim(0, 1)
    plt.yticks(ticks)

    plt.axhline(y=0.5, linestyle="-", color="b")
    plt.xlabel(">= Overlap")
    plt.ylabel("Recall")
    plt.title("Recall by at least overlap")
    plt.savefig(ge_overlap_plot, dpi=300)

    return hit_count_modern, hit_count_ancient


def giggle_benchmark(
    query_bed: IntervalTransformer,
    sample_bed: IntervalTransformer,
    query_embeds: MmapF32,
    sample_embeds: MmapF32,
    legacy_results_path: str,
    overlap_plot_path: str,
    ge_overlap_plot_path: str,
    k: int,
) -> tuple[int, int]:
    """
    In addition to savign figures, two values are returned: hitCount,
    groundTruth. These correspond to the amount of results returned by the
    modern/legacy system respectively.
    """

    print("Performing modern giggle")
    modern_results = modern_giggle(sample_bed, sample_embeds, query_bed, query_embeds, k)
    print("Parsing ancient giggle")
    ancient_results = parse_legacy_giggle(legacy_results_path)

    Path(overlap_plot_path).parent.mkdir(parents=True, exist_ok=True)
    Path(ge_overlap_plot_path).parent.mkdir(parents=True, exist_ok=True)
    return analyze_results(modern_results, ancient_results, overlap_plot_path, ge_overlap_plot_path)
