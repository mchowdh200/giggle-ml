import os

import pyfastx

from data_wrangling.seq_datasets import BedDataset, FastaDataset, TokenizedDataset
from gpu_embeds.inference_batch import BatchInferHyenaDNA


def main():
    workers = 2
    batchSize = 2048
    bufferSize = batchSize * 2
    inputsInMemory = True
    seqMaxLen = 1000

    dirRoot = "data"
    fastaPath = dirRoot + "/hg19.fa"
    intervalDir = dirRoot + "/myod"
    embedsDir = dirRoot + "/myod"

    infSystem = BatchInferHyenaDNA()
    names = os.listdir(intervalDir)
    datasets = list()
    outPaths = list()

    # use Fastx to read sequences into memory for sharing between workers
    print("Loading fasta file into memory...")
    fastaIdx = pyfastx.Fastx(fastaPath)
    seqs = {name: seq for name, seq, *_ in fastaIdx}

    print("Preparing datasets...")
    for i, name in enumerate(names):
        embedFile = os.path.join(embedsDir, name + ".npy")
        bedFile = os.path.join(intervalDir, name)
        outPaths.append(embedFile)

        bedDataset = BedDataset(
            bedFile, inputsInMemory, bufferSize=bufferSize, maxLen=seqMaxLen
        )
        fastaDataset = FastaDataset(seqs, bedDataset)
        tokDataset = TokenizedDataset(fastaDataset, padToLength=seqMaxLen)

        datasets.append(tokDataset)

    print("Starting inference on", len(datasets), "bed files.")
    infSystem.batchInfer(datasets, outPaths, batchSize, workers)


if __name__ == "__main__":
    main()

