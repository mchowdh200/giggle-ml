import datetime
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from giggleml.intervalTransformer import IntervalTransformer
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


def intersection(x: GenomicInterval, y: GenomicInterval) -> GenomicInterval | None:
    ch1, start1, end1 = x
    ch2, start2, end2 = y

    if ch1 != ch2:
        return None

    start = max(start1, start2)
    end = min(end1, end2)

    if start >= end:
        return None

    return (ch1, start, end)


def overlapDegree(x: GenomicInterval, y: GenomicInterval) -> float:
    """
    100% (1) means the smaller interval is fully contained in the larger interval.
    """

    z = intersection(x, y)

    if z is None:
        return 0

    zSize = z[2] - z[1]
    xSize = x[2] - x[1]
    ySize = y[2] - y[1]
    refSize = min(xSize, ySize)
    return zSize / refSize


def intersects(x: GenomicInterval, y: GenomicInterval):
    return overlapDegree(x, y) > 0


def buildVectorDB(embeds: MmapF32):
    # dim = embeds.shape[1]
    # vdb = faiss.IndexFlatIP(dim)
    normal = embeds / np.linalg.norm(embeds, axis=1).reshape(-1, 1)
    # vdb.add(normal)
    # return vdb
    return SimpleKNN(normal)


def modernGiggle(
    sampleIT: IntervalTransformer,
    sampleEmbeds: MmapF32,
    queryIT: IntervalTransformer,
    queryEmbeds: MmapF32,
    k: int,
):

    results = defaultdict(set)

    print("Building vector database")
    vdb = buildVectorDB(sampleEmbeds)
    _, knn = vdb.search(queryEmbeds, k)
    print(" - completed KNN search")

    for queryId, neighbors in enumerate(knn):
        queryBase = queryIT.oldDataset[queryIT.backwardIdx(queryId)]
        hits = []

        for neighborId in neighbors:
            hitBase = sampleIT.oldDataset[sampleIT.backwardIdx(neighborId)]

            if intersects(hitBase, queryBase):
                hits.append(hitBase)

        results[queryBase].update(hits)

    return results


def parseLegacyGiggle(path: str) -> dict[GenomicInterval, list[GenomicInterval]]:
    ancientResults = dict()
    with open(path, "r") as f:
        profile = list[GenomicInterval]()
        head: GenomicInterval | None = None

        for line in f:
            chr, start, end, *_ = line.strip().split()
            start = int(start)
            end = int(end)

            if chr[0] == "#":
                ancientResults[head] = profile
                profile = []
                chr = chr[2:]
                head = (chr, start, end)
                continue

            profile.append((chr, start, end))

        ancientResults[head] = profile
        ancientResults.pop(None)

    return ancientResults


def analyzeResults(modernResults, ancientResults, overlapPlot, geOverlapPlot):
    for query in modernResults.keys():
        if query not in ancientResults:
            raise RuntimeError("Extra query in modern", query)

    for query in ancientResults.keys():
        if query not in modernResults:
            raise RuntimeError("Missing query in modern", query)

    # core stats

    highestDepthPerQuery = max(map(len, ancientResults.values()))
    print("Highest depth per query (ground truth)", highestDepthPerQuery)

    print("Modern non zero hits", sum(filter(lambda x: x != 0, map(len, modernResults.values()))))

    hitCountAncient = sum(map(len, ancientResults.values()))
    hitCountModern = sum(map(len, modernResults.values()))
    print("Hit Count")
    print(" - ground truth", hitCountAncient)
    print(" - modern", hitCountModern)

    # used to make a histogram at 10% intervals
    hits = [0] * 11
    totals = [0] * 11

    for query in ancientResults.keys():
        modernProfile = set(modernResults[query])
        ancientProfile = set(ancientResults[query])

        overlap = len(ancientProfile.intersection(modernProfile))
        assert overlap >= min(len(modernProfile), len(ancientProfile))

        for realHit in ancientProfile:
            overlap = overlapDegree(realHit, query)
            discreteOverlap = round(overlap * 10)
            totals[discreteOverlap] += 1

            if realHit in modernProfile:
                hits[discreteOverlap] += 1

    recall = sum(hits) / hitCountAncient
    print("recall", recall)

    # recall by overlap

    hitProb = []
    for i in range(11):
        if totals[i] == 0:
            raise RuntimeError("A total is zero, cannot compute average")
        hitProb.append(hits[i] / totals[i])

    ticks = np.arange(0, 1.1, 0.1)

    plt.figure()
    plt.bar(ticks, hitProb, width=0.1)

    plt.xticks(ticks)
    plt.xlim(0, 1)
    plt.yticks(ticks)

    plt.axhline(y=0.5, linestyle="-", color="b")
    plt.xlabel("Overlap")
    plt.ylabel("Recall")
    plt.title("Recall by overlap")
    plt.savefig(overlapPlot, dpi=300)

    # recall by >= overlap

    runningHits = np.array([0] * 11)
    runningTotals = np.array([0] * 11)

    for i in range(len(hits)):
        runningHits[i] = sum(hits[i:])
        runningTotals[i] = sum(totals[i:])

        if runningTotals[i] == 0:
            print("A total is zero, cannot compute average")

    hitProbSum = runningHits / runningTotals

    plt.figure()
    plt.bar(ticks, hitProbSum, width=0.1)

    plt.xticks(ticks)
    plt.xlim(0, 1)
    plt.yticks(ticks)

    plt.axhline(y=0.5, linestyle="-", color="b")
    plt.xlabel(">= Overlap")
    plt.ylabel("Recall")
    plt.title("Recall by at least overlap")
    plt.savefig(geOverlapPlot, dpi=300)

    return hitCountModern, hitCountAncient


def giggleBenchmark(
    queryBed: IntervalTransformer,
    sampleBed: IntervalTransformer,
    queryEmbeds: MmapF32,
    sampleEmbeds: MmapF32,
    legacyResultsPath: str,
    overlapPlotPath: str,
    geOverlapPlotPath: str,
    k: int,
) -> tuple[int, int]:
    """
    In addition to savign figures, two values are returned: hitCount,
    groundTruth. These correspond to the amount of results returned by the
    modern/legacy system respectively.
    """

    print("Performing modern giggle")
    modernResults = modernGiggle(sampleBed, sampleEmbeds, queryBed, queryEmbeds, k)
    print("Parsing ancient giggle")
    ancientResults = parseLegacyGiggle(legacyResultsPath)

    Path(overlapPlotPath).parent.mkdir(parents=True, exist_ok=True)
    Path(geOverlapPlotPath).parent.mkdir(parents=True, exist_ok=True)
    return analyzeResults(modernResults, ancientResults, overlapPlotPath, geOverlapPlotPath)
