from gpu_embeds.inference_batch import BatchInferHyenaDNA
from gpu_embeds.from_genomic_benchmarks import main as genom_main
import numpy as np
from types import SimpleNamespace
from data_wrangling.seq_datasets import BedDataset, FastaDataset, TokenizedDataset
import statistical_tests.tests as tests
from gpu_embeds.region2vec_wrapper import R2VBatchInf


def get_intervals(bedPath):
    with open(bedPath) as f:
        intervals = []
        for line in f:
            columns = line.strip().split()
            chromosome, start, end, *_ = columns
            start = int(start)
            end = int(end)
            intervals.append((chromosome, start, end))
    return intervals


def getInfSystem():
    # HyenaDNA
    # return BatchInferHyenaDNA()

    # Region2Vec
    return R2VBatchInf()


def embeds(limit, batchSize, paths, workers, bufferSize):
    print("Preparing bed dataset...")
    bedDs = BedDataset(paths.bed, limit=limit,
                       inMemory=True, bufferSize=bufferSize)
    # print("Preparing seq dataset...")
    print("Running inference...")
    inf = getInfSystem()

    # HyenaDNA
    # fastaDs = FastaDataset(paths.fasta, bedDs)
    # seqDs = TokenizedDataset(fastaDs)
    # inf.batchInfer(seqDs, paths.embeds, batchSize, workers)

    # Region2Vec
    inf.batchInfer(bedDs, paths.embeds, batchSize, workers)


def doSWT(intervals, infSystem, limit):
    fastaDs = FastaDataset(paths.fasta, intervals)
    tokDs = TokenizedDataset(fastaDs)
    tests.swt(paths.embeds, fastaDs, infSystem, considerLimit=limit)
    print()


def run_tests(paths, limit):
    intervals = BedDataset(paths.bed, limit=limit, inMemory=True)
    infSystem = getInfSystem()

    embeds = np.memmap(paths.embeds, dtype=np.float32, mode="r")
    embedDim = infSystem.embedDim
    embeds = embeds.reshape(-1, embedDim)

    # doSWT(intervals, infSystem, limit)
    tests.ctt(embeds)
    print()
    tests.gdst(embeds, intervals)
    print()
    tests.npt(embeds, intervals)
    print()
    # tests.rct(embeds, tokDs)
    # print()


if __name__ == "__main__":
    print("This is main")

    # INFO: Config
    limit = 440
    batchSize = 11
    workers = 1
    bufferSize = 100

    # paths = SimpleNamespace(
    #     fasta="./data/hg38.fa",
    #     bed="./data/hg38_trf.bed",
    #     embeds="./data/embeddings.npy")
    paths = SimpleNamespace(
        fasta="./data/synthetic/seqs.fa",
        bed="./data/synthetic/universe_0.bed",
        embeds="./data/synthetic/embeds.npy")

    embeds(limit, batchSize, paths, workers, bufferSize)
    run_tests(paths, limit)
    # genom_main(limit, batchSize, paths.embeds)
