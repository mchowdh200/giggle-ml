import datetime
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import faiss
import matplotlib.pyplot as plt
import numpy as np

import intervalTransforms as Transform
from data_wrangling.seq_datasets import BedDataset
from data_wrangling.seq_datasets import FastaDataset
from data_wrangling.seq_datasets import TokenizedDataset
from data_wrangling.transform_dataset import TransformDataset
from embed_gen.inference_batch import BatchInferHyenaDNA
from utils.strToInfSystem import getInfSystem


# TODO: extract intersection utilities


def intersection(x, y):
    ch1, start1, end1 = x
    ch2, start2, end2 = y

    if ch1 != ch2:
        return None

    start = max(start1, start2)
    end = min(end1, end2)

    if start >= end:
        return None

    return (ch1, start, end)


def overlapDegree(x, y):
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


def intersects(x, y):
    return overlapDegree(x, y) > 0


def build_vecdb_shards(intervals, embeds):
    dim = embeds.shape[1]
    shard = faiss.IndexFlatIP(dim)

    for embed in embeds:
        embed = embed / np.linalg.norm(embed)
        embed = embed.reshape(1, -1)
        shard.add(embed)

    return shard


def modernGiggle(shard,
                 sampleIntervals: TransformDataset,
                 sampleEmbeds,
                 queryIntervals: TransformDataset,
                 queryEmbeds,
                 k):

    results = defaultdict(set)

    _, knn = shard.search(queryEmbeds, k)
    print(" - completed KNN search")

    for queryId, neighbors in enumerate(knn):
        queryBase = queryIntervals.baseItem(queryId)
        hits = []

        for neighborId in neighbors:
            hitBase = sampleIntervals.baseItem(neighborId)

            if intersects(hitBase, queryBase):
                hits.append(hitBase)

        results[queryBase].update(hits)

    return results


def parseLegacyGiggle(path):
    ancientResults = dict()
    with open(path, "r") as f:
        profile = []
        head = None

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


def analyzeResults(modernResults, ancientResults, overlapPlot, geOverlapPlot, k):
    # checks

    for query in modernResults.keys():
        if query not in ancientResults:
            print('Extra query in modern', query)
            return

    for query in ancientResults.keys():
        if query not in modernResults:
            print('Missing query in modern', query)
            return

    # core stats

    highestDepthPerQuery = max(map(len, ancientResults.values()))
    print('Highest depth per query (ground truth)', highestDepthPerQuery)

    print('Modern non zero hits', sum(
        filter(lambda x: x != 0,
               map(len, modernResults.values()))))

    hitCountAncient = sum(map(len, ancientResults.values()))
    hitCountModern = sum(map(len, modernResults.values()))
    print("Hit Count")
    print(' - ground truth', hitCountAncient)
    print(' - modern', hitCountModern)

    # used to make a histogram at 10% intervals
    hits = [0]*11
    totals = [0]*11

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
    print('recall', recall)

    # recall by overlap

    hitProb = []
    for i in range(11):
        if totals[i] == 0:
            print("A total is zero, cannot compute average")
            return
        hitProb.append(hits[i] / totals[i])

    ticks = np.arange(0, 1.1, 0.1)

    plt.figure()
    plt.bar(ticks, hitProb, width=0.1)

    plt.xticks(ticks)
    plt.xlim(0, 1)
    plt.yticks(ticks)

    plt.axhline(y=0.5, linestyle='-', color='b')
    plt.xlabel("Overlap")
    plt.ylabel("Recall")
    plt.title("Recall by overlap")
    plt.savefig(overlapPlot, dpi=300)

    # recall by >= overlap

    runningHits = np.array([0]*11)
    runningTotals = np.array([0]*11)

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

    plt.axhline(y=0.5, linestyle='-', color='b')
    plt.xlabel(">= Overlap")
    plt.ylabel("Recall")
    plt.title("Recall by at least overlap")
    plt.savefig(GEOverlapPlot, dpi=300)


def giggleBenchmark(
        queryIntervalsPath,
        sampleIntervalsPath,
        queryEmbedsPath,
        sampleEmbedsPath,
        legacyGiggleResults,
        overlapPlotPath,
        geOverlapPlotPath,
        k = 100,
        querySwellFactor = 1,
        queryChunkAmnt = 1,
        sampleSwellFactor = 1,
        sampleChunkAmnt = 1,
        queryTranslation = 0,
):
    conf = snakemake.config.batchInference
    bufferSize = conf.bufferSize
    inputsInMemory = conf.inputsInMemory

    sampleIntervals = TransformDataset(
        backingDataset=BedDataset(
            sampleIntervalsPath,
            inMemory=inputsInMemory,
            bufferSize=bufferSize),
        transforms=[
            Transform.Swell(swellFactor=sampleSwellFactor),
            Transform.Chunk(chunkAmnt=sampleChunkAmnt)
        ])

    queryIntervals = TransformDataset(
        backingDataset=BedDataset(
            queryIntervalsPath,
            inMemory=inputsInMemory,
            bufferSize=bufferSize),
        transforms=[
            Transform.Translate(offset=queryTranslation),
            Transform.Swell(swellFactor=querySwellFactor),
            Transform.Chunk(chunkAmnt=queryChunkAmnt)
        ])

    print("Length (sample, query):", len(sampleIntervals), len(queryIntervals))
    dim = getInfSystem().embedDim

    sampleEmbeds = np.memmap(sampleEmbedsPath, dtype=np.float32, mode="r")
    sampleEmbeds = sampleEmbeds.reshape(-1, dim)
    queryEmbeds = np.memmap(queryEmbedsPath, dtype=np.float32, mode="r")
    queryEmbeds = queryEmbeds.reshape(-1, dim)

    # Perform analysis
    print("Building vector database shards")
    vdbShards = build_vecdb_shards(sampleIntervals, sampleEmbeds)

    print("Performing modern giggle")
    modernResults = modernGiggle(vdbShards,
                                 sampleIntervals,
                                 sampleEmbeds,
                                 queryIntervals,
                                 queryEmbeds,
                                 k)

    print("Parsing ancient giggle")
    ancientResults = parseLegacyGiggle(legacyGiggleResults)
    querySeqLens = list(map(lambda x: x[2] - x[1], queryIntervals))
    print('Query Sequence Lengths (after swell & chunk)', set(querySeqLens))
    analyzeResults(modernResults, ancientResults, overlapPlotPath, geOverlapPlotPath, k)