import faiss
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from data_wrangling.swell_and_chunk_dataset import SwellChunkDataset
from data_wrangling.seq_datasets import BedDataset
from data_wrangling.seq_datasets import TokenizedDataset
from data_wrangling.seq_datasets import FastaDataset
from data_wrangling.truncated_dataset import TruncatedDataset
from gpu_embeds.inference_batch import BatchInferHyenaDNA
from types import SimpleNamespace


# TODO: extract to utils
def intersects(x, y):
    # Unpack the intervals
    ch1, start1, end1 = x
    ch2, start2, end2 = y

    if ch1 != ch2:
        return False
    return max(start1, start2) <= min(end1, end2)


def by_chromosome(intervals):
    byChrm = defaultdict(list)
    byChrm[intervals[0][0]] = range(len(intervals))

    for i, interval in enumerate(intervals):
        chrm, *_ = interval
        byChrm[chrm].append(i)

        # TODO: chaotic ordering will break when multiple shards/chromosomes
        if len(byChrm) > 1:
            raise ValueError("multiple chromosomes not yet supported")
    return byChrm


def build_vecdb_shards(intervals, embeds):
    dim = embeds.shape[1]
    shards = defaultdict(lambda: faiss.IndexFlatL2(dim))
    byChrm = by_chromosome(intervals)

    for chrm, indices in byChrm.items():
        print(" - Building shard for", chrm)
        shard = shards[chrm]
        correspondingEmbeds = embeds[indices]
        shard.add(correspondingEmbeds)

    return shards


def modern_giggle(vecdbShards,
                  sampleIntervals: SwellChunkDataset,
                  sampleEmbeds,
                  queryIntervals: SwellChunkDataset,
                  queryEmbeds,
                  k):

    results = dict()
    byChrm = by_chromosome(queryIntervals)

    for chrm, shard in vecdbShards.items():
        batchQueryIds = byChrm[chrm]

        bachQueryEmbeds = queryEmbeds[batchQueryIds]
        print(f" - {len(batchQueryIds)} queries on shard", chrm)

        knn = zip(batchQueryIds, shard.search(bachQueryEmbeds, k)[1])
        print("\t- completed KNN search")

        for queryId, neighbors in knn:
            backingInterval = queryIntervals.baseItem(queryId)
            intersections = set()

            for neighborId in neighbors:
                neighborInterval = sampleIntervals.baseItem(neighborId)

                if intersects(backingInterval, neighborInterval):
                    intersections.add(neighborInterval)

            results[backingInterval] = list(intersections)

    return results


def parse_ancient_giggle(path):
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


def analyze_results(modernResults, ancientResults, outDir, k):
    highestDepthPerQuery = max(map(len, ancientResults.values()))
    print('Highest depth per query (ground truth)', highestDepthPerQuery)

    hits = 0

    # histogram at 10% intervals
    hitCountByOverlap = defaultdict(int)
    missCountByOverlap = defaultdict(int)

    for query in modernResults.keys():
        if query not in ancientResults:
            print('Extra query in modern', query)
            continue

    for query, ancientProfile in ancientResults.items():
        if query not in modernResults:
            print('Missing query in modern', query)
            continue

        modernProfile = modernResults[query]
        setModern = set(modernProfile)
        setAncient = set(ancientProfile)
        overlap = len(setAncient.intersection(setModern))
        if overlap < min(len(modernProfile), len(ancientProfile)):
            print('Mismatch',
                  query,
                  overlap,
                  len(modernProfile),
                  len(ancientProfile))
            print(' - modern', modernProfile)
            print(' - ancient', ancientProfile)
        hits += overlap

        for miss in setAncient:
            start = max(miss[1], query[1])
            end = min(miss[2], query[2])
            intervalOverlap = end - start
            intervalOverlap /= (miss[2] - miss[1])  # normalized
            # round to nearest 10% -- NOT truncating to 10%
            intervalOverlap = round(intervalOverlap * 10) * 10

            if miss not in setModern:
                missCountByOverlap[intervalOverlap] += 1
            else:
                hitCountByOverlap[intervalOverlap] += 1

    # core stats

    print('Modern non zero hits', sum(
        filter(lambda x: x != 0,
               map(len, modernResults.values()))))

    hitCountAncient = sum(map(len, ancientResults.values()))
    hitCountModern = sum(map(len, modernResults.values()))
    print("Hit Count")
    print(' - ground truth', hitCountAncient)
    print(' - modern', hitCountModern)

    queryCountAncient = len(ancientResults)
    queryCountModern = len(modernResults)
    print('Query counts: ancient, modern:',
          queryCountAncient, queryCountModern)

    print('Intersect probability overall',
          hitCountModern / (queryCountModern * k))

    recall = hits / hitCountAncient
    print('recall', recall)

    hitProbByOverlap = dict()
    for overlap, hitCount in hitCountByOverlap.items():
        missCount = missCountByOverlap[overlap]
        total = hitCount + missCount
        hitProbByOverlap[overlap] = hitCount / total

    # plotting

    # recall by overlap plot (exact overlap band)
    plt.figure()
    plt.bar(hitProbByOverlap.keys(), hitProbByOverlap.values())
    plt.xticks(list(hitProbByOverlap.keys()))
    plt.yticks(list(map(lambda x: x / 10, range(0, 11))))
    plt.title(f"Hit Probability at Overlap, k={k}")
    plt.axhline(y=0.5, color='g', linestyle='-')
    plt.savefig(outDir + "/hitProbAtOverlap.png")

    # recall by ATLEAST OVERLAP plot

    aggregateHitCountByOverlap = defaultdict(int)
    aggregateMissCountByOverlap = defaultdict(int)

    for overlap in reversed(sorted(hitProbByOverlap.keys())):
        prevHits = aggregateHitCountByOverlap[overlap + 10]
        thisHits = hitCountByOverlap[overlap]
        aggregateHitCountByOverlap[overlap] = prevHits + thisHits

        prevMisses = aggregateMissCountByOverlap[overlap + 10]
        thisMisses = missCountByOverlap[overlap]
        aggregateMissCountByOverlap[overlap] = prevMisses + thisMisses

    del aggregateHitCountByOverlap[110]
    del aggregateMissCountByOverlap[110]

    hitProbByAtLeastOverlap = dict()
    for overlap, hitCount in aggregateHitCountByOverlap.items():
        missCount = aggregateMissCountByOverlap[overlap]
        total = hitCount + missCount
        hitProbByAtLeastOverlap[overlap] = hitCount / total

    plt.figure()
    plt.bar(hitProbByAtLeastOverlap.keys(), hitProbByAtLeastOverlap.values())
    plt.xticks(list(hitProbByOverlap.keys()))
    plt.yticks(list(map(lambda x: x / 10, range(0, 11))))
    plt.title(f"Recall by Overlap Cutoff, k={k}")
    plt.axhline(y=0.5, color='g', linestyle='-')
    plt.savefig(outDir + "/recallByOverlapCutoff.png")

    return recall


def main():
    # INFO: Config

    limit = None
    batchSize = 1000
    workers = 2
    bufferSize = 10
    inputsInMemory = True

    padToLength = 100
    k = 100

    querySwellFactor = 2
    queryChunkAmnt = 3
    sampleSwellFactor = 0
    sampleChunkAmnt = 1

    paths = SimpleNamespace(
        fasta="./data/hg38.fa",
        queryBed="./data/giggleBench/query.bed",
        sampleBed="./data/giggleBench/sample.bed",

        queryEmbeds="./data/giggleBench/embeds/swell_query/" + "/query.npy",
        sampleEmbeds="./data/giggleBench/embeds/straight" + "/sample.npy",

        experimentalAnalysis="./experiments/giggleBench/swell_query",
        ancientResults="./data/giggleBench/gresults.gbed")

    # write config information to info.md
    with open(paths.experimentalAnalysis + "/info.md", "w") as f:
        f.write(f"k: {k}\n")
        f.write(f"querySwellFactor: {querySwellFactor}\n")
        f.write(f"queryChunkAmnt: {queryChunkAmnt}\n")
        f.write(f"sampleSwellFactor: {sampleSwellFactor}\n")
        f.write(f"sampleChunkAmnt: {sampleChunkAmnt}\n")

    sampleIntervals = SwellChunkDataset(
        BedDataset(
            paths.sampleBed,
            inMemory=inputsInMemory,
            limit=limit,
            bufferSize=bufferSize),
        sampleSwellFactor,
        sampleChunkAmnt)

    queryIntervals = SwellChunkDataset(
        BedDataset(
            paths.queryBed,
            inMemory=inputsInMemory,
            limit=limit,
            bufferSize=bufferSize),
        querySwellFactor,
        queryChunkAmnt)

    # Make embeddings

    infSystem = BatchInferHyenaDNA()
    dim = infSystem.embedDim

    # sampleTokens = TokenizedDataset(
    #     FastaDataset(paths.fasta, sampleIntervals),
    #     padToLength=padToLength)
    #
    # queryTokens = TokenizedDataset(
    #     FastaDataset(paths.fasta, queryIntervals),
    #     padToLength=padToLength)
    #
    # infSystem.batchInfer(sampleTokens, paths.sampleEmbeds, batchSize, workers)
    # infSystem.batchInfer(queryTokens, paths.queryEmbeds, batchSize, workers)

    sampleEmbeds = np.memmap(paths.sampleEmbeds, dtype=np.float32, mode="r")
    sampleEmbeds = sampleEmbeds.reshape(-1, dim)
    queryEmbeds = np.memmap(paths.queryEmbeds, dtype=np.float32, mode="r")
    queryEmbeds = queryEmbeds.reshape(-1, dim)

    # Perform analysis

    print("Building vector database shards")
    vdbShards = build_vecdb_shards(sampleIntervals, sampleEmbeds)

    print("Performing modern giggle")
    modernResults = modern_giggle(vdbShards,
                                  sampleIntervals,
                                  sampleEmbeds,
                                  queryIntervals,
                                  queryEmbeds,
                                  k)

    print("Parsing ancient giggle")
    ancientResults = parse_ancient_giggle(paths.ancientResults)

    print('k =', k)
    querySeqLens = list(map(lambda x: x[2] - x[1], queryIntervals))
    print('Query Sequence Lengths (after swell & chunk)', set(querySeqLens))
    analyze_results(modernResults, ancientResults,
                    paths.experimentalAnalysis, k)


if __name__ == "__main__":
    main()
