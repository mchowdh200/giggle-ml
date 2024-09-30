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
from giggle import build_vecdb, intersection_scan


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

    vdb = build_vecdb(paths.embeds, getInfSystem().embedDim)
    giggleResultsPath = "./data/giggleBench/gresults.gbed"
    intersection_scan(vdb, paths.bed, paths.embeds,
                      testPaths.bed, testPaths.embeds,
                      giggleResultsPath)
    # run_tests(paths, limit)
    # genom_main(limit, batchSize, paths.embeds)


if __name__ == "__main__":
    main()
