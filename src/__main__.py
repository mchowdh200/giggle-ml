from gpu_embeds.inference_batch import BatchInferHyenaDNA
from gpu_embeds.from_genomic_benchmarks import main as genom_main
import numpy as np
from types import SimpleNamespace
from data_wrangling.seq_datasets import BedDataset, FastaDataset, TokenizedDataset
import statistical_tests.tests as tests
import faiss
from collections import defaultdict
from matplotlib import pyplot as plt
from utils.bed_utils import get_intervals


def getInfSystem():
    # HyenaDNA
    return BatchInferHyenaDNA()

    # Region2Vec
    # return R2VBatchInf()


def make_embeds(limit, batchSize, paths, workers, bufferSize, inMemory=True):
    print("Preparing bed dataset...")
    bedDs = BedDataset(paths.bed, limit=limit,
                       inMemory=inMemory, bufferSize=bufferSize)
    # print("Preparing seq dataset...")
    print("Running inference...")
    inf = getInfSystem()

    # HyenaDNA
    fastaDs = FastaDataset(paths.fasta, bedDs)
    seqDs = TokenizedDataset(fastaDs)
    inf.batchInfer(seqDs, paths.embeds, batchSize, workers)

    # Region2Vec
    # inf.batchInfer(bedDs, paths.embeds, batchSize, workers)


def advanced_tests(intervals, embeds, infSystem, limit):
    fastaDs = FastaDataset(paths.fasta, intervals)
    tokDs = TokenizedDataset(fastaDs)

    # TODO: only god knows why the reverse order causes a crash
    tests.swt(paths.embeds, fastaDs, infSystem, considerLimit=limit)
    print()
    tests.rct(embeds, tokDs)


def run_tests(paths, limit):
    intervals = BedDataset(paths.bed, limit=limit, inMemory=True)
    infSystem = getInfSystem()

    embeds = np.memmap(paths.embeds, dtype=np.float32, mode="r")
    embedDim = infSystem.embedDim
    embeds = embeds.reshape(-1, embedDim)

    print()
    advanced_tests(intervals, embeds, infSystem, limit)
    print()
    tests.ctt(embeds)
    print()
    tests.gdst(embeds, intervals)
    print()
    tests.npt(embeds, intervals)


def build_vecdb(paths):
    dim = getInfSystem().embedDim
    # vdb = faiss.IndexHNSWFlat(dim, faiss.METRIC_INNER_PRODUCT)
    vdb = faiss.IndexFlatL2(dim)

    embeds = np.memmap(paths.embeds, dtype=np.float32, mode="r")
    embeds = embeds.reshape(-1, dim)

    vdb.add(embeds)

    return vdb


def intersection_scan(vecdb,
                      fullIntervals,
                      fullEmbeds,
                      testIntervals,
                      testEmbeds,
                      ancientResultsPath,
                      k=30):
    print('k =', k)

    testEmbeds = np.memmap(testEmbeds, dtype=np.float32, mode="r") \
        .reshape(-1, 256)
    fullEmbeds = np.memmap(fullEmbeds, dtype=np.float32, mode="r") \
        .reshape(-1, 256)

    testIntervals = get_intervals(testIntervals)
    fullIntervals = get_intervals(fullIntervals)

    def intersects(x, y):
        # Unpack the intervals
        ch1, start1, end1 = x
        ch2, start2, end2 = y

        if ch1 != ch2:
            return False
        return max(start1, start2) <= min(end1, end2)

    # dists, ids
    _, ids = vecdb.search(testEmbeds, k)

    nativeResults = dict()

    for i, neighborIds in enumerate(ids):
        testInt = testIntervals[i]
        neighborInts = [fullIntervals[id] for id in neighborIds]

        intersectingNeighbors = list(filter(
            lambda x: intersects(x, testInt), neighborInts))

        nativeResults[testInt] = intersectingNeighbors

    querySeqLens = list(map(lambda x: x[2] - x[1], testIntervals))
    print('Query Sequence Lengths', set(querySeqLens))

    # parse ancient results

    ancientResults = dict()
    with open(ancientResultsPath, "r") as f:
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

    highestDepthPerQuery = max(map(len, ancientResults.values()))
    print('Highest depth per query (ground truth)', highestDepthPerQuery)

    # perform analysis

    hits = 0

    # histogram at 10% intervals
    hitCountByOverlap = defaultdict(int)
    missCountByOverlap = defaultdict(int)

    for query, ancientProfile in ancientResults.items():
        if query not in nativeResults:
            print('Mismatch key', query)
            continue

        nativeProfile = nativeResults[query]
        setNative = set(nativeProfile)
        setAncient = set(ancientProfile)
        overlap = len(setAncient.intersection(setNative))
        if overlap < min(len(nativeProfile), len(ancientProfile)):
            print('Mismatch',
                  query,
                  overlap,
                  len(nativeProfile),
                  len(ancientProfile))
            print(' - native', nativeProfile)
            print(' - ancient', ancientProfile)
        hits += overlap

        for miss in setAncient:
            start = max(miss[1], query[1])
            end = min(miss[2], query[2])
            intervalOverlap = end - start
            intervalOverlap /= (miss[2] - miss[1])  # normalized
            # round to nearest 10% -- NOT truncating to 10%
            intervalOverlap = round(intervalOverlap * 10) * 10

            if miss not in setNative:
                missCountByOverlap[intervalOverlap] += 1
            else:
                hitCountByOverlap[intervalOverlap] += 1

    hitProbByOverlap = dict()
    for overlap, hitCount in hitCountByOverlap.items():
        missCount = missCountByOverlap[overlap]
        total = hitCount + missCount
        hitProbByOverlap[overlap] = hitCount / total

    # recall by overlap plot (exact overlap band)
    plt.figure()
    plt.bar(hitProbByOverlap.keys(), hitProbByOverlap.values())
    plt.xticks(list(hitProbByOverlap.keys()))
    plt.yticks(list(map(lambda x: x / 10, range(0, 11))))
    plt.title(f"Hit Probability at Overlap, k={k}")
    plt.axhline(y=0.5, color='g', linestyle='-')
    plt.savefig("./experiments/giggleBench/hitProbAtOverlap.png")

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
    plt.savefig("./experiments/giggleBench/recallByOverlapCutoff.png")

    print('Native non zero hits', sum(
        filter(lambda x: x != 0,
               map(len, nativeResults.values()))))

    hitCountAncient = sum(map(len, ancientResults.values()))
    hitCountNative = sum(map(len, nativeResults.values()))
    print("Hit Count")
    print(' - ground truth', hitCountAncient)
    print(' - native', hitCountNative)

    queryCountAncient = len(ancientResults)
    queryCountNative = len(nativeResults)
    print('Query count: ancient, native', queryCountAncient, queryCountNative)

    print('Intersect probability overall',
          hitCountNative / (queryCountNative * k))

    recall = hits / hitCountAncient
    print('recall', recall)

    return recall


def main():
    print("This is main")

    # INFO: Config
    limit = None
    batchSize = 200
    workers = 2
    bufferSize = 100
    inputsInMemory = False

    # paths = SimpleNamespace(
    #     fasta="./data/hg38.fa",
    #     bed="./data/hg38_trf.bed",
    #     embeds="./data/embeddings.npy")
    # paths = SimpleNamespace(
    #     fasta="./data/synthetic/seqs.fa",
    #     bed="./data/synthetic/universe_0.bed",
    #     embeds="./data/synthetic/embeds.npy")

    # Giggle Misc
    paths = SimpleNamespace(
        fasta="./data/hg38.fa",
        bed="./data/giggleBench/sample.bed",
        embeds="./data/giggleBench/embeds_sample.npy")

    testPaths = SimpleNamespace(
        fasta="./data/hg38.fa",
        bed="./data/giggleBench/query.bed",
        embeds="./data/giggleBench/embeds_query.npy")

    # make_embeds(limit, batchSize, paths, workers, bufferSize, inputsInMemory)
    # make_embeds(limit, batchSize, testPaths,
    #             workers, bufferSize, inputsInMemory)

    vdb = build_vecdb(paths)
    giggleResultsPath = "./data/giggleBench/gresults.gbed"
    intersection_scan(vdb, paths.bed, paths.embeds,
                      testPaths.bed, testPaths.embeds,
                      giggleResultsPath)
    # run_tests(paths, limit)
    # genom_main(limit, batchSize, paths.embeds)


if __name__ == "__main__":
    main()
