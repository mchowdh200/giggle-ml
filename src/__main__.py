from gpu_embeds.inference_batch import batchInfer
from gpu_embeds.from_genomic_benchmarks import main as genom_main
import numpy as np
from types import SimpleNamespace
from gpu_embeds.from_fasta import BedDataset, SeqDataset
import statistical_tests.tests as tests


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


def embeds(limit, batchSize, paths, workers, bufferSize):
    print("Preparing bed dataset...")
    bedDs = BedDataset(paths.bed, limit=limit,
                       inMemory=False, bufferSize=bufferSize)
    print("Preparing fasta dataset...")
    fastaDs = SeqDataset(paths.fasta, bedDs)
    print("Running inference...")
    batchInfer(fastaDs, paths.embeds, batchSize, workers)


def run_tests(limit, paths):
    bedDs = BedDataset(paths.bed, limit=limit, inMemory=True)
    limit = min(limit, len(bedDs))
    intervals = [bedDs[i] for i in range(limit)]

    embeds = np.memmap(paths.embeds + ".0", dtype=np.float32, mode="r")
    embeds = embeds.reshape(-1, 256)

    print()
    tests.ctt(embeds)
    tests.gdst(intervals, embeds)
    # tests.npt(intervals, embeds)
    tests.cct(intervals, embeds)


if __name__ == "__main__":
    print("This is main")

    # INFO: Config
    limit = 1000
    batchSize = 5
    workers = 2
    bufferSize = 200

    # paths = SimpleNamespace(
    #     fasta="./data/hg38.fa",
    #     bed="./data/hg38_trf.bed",
    #     embeds="./data/embeddings.npy")
    paths = SimpleNamespace(
        fasta="./data/synthetic/seqs.fa",
        bed="./data/synthetic/universe_0.bed",
        embeds="./data/synthetic/embeds")

    # embeds(limit, batchSize, paths, workers, bufferSize)
    run_tests(limit, paths)
    # genom_main(limit, batchSize, paths.embeds)
