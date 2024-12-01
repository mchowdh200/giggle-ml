import os

from data_wrangling.seq_datasets import TokenizedDataset, FastaDataset, BedDataset
from gpu_embeds.inference_batch import BatchInferHyenaDNA


def main():
    workers = 8
    batchSize = int(5e3)
    bufferSize = batchSize * 2
    inputsInMemory = True
    seqMaxLen = 1000

    fastaPath = "./data/hg19.fa"
    intervalDir = "./data/roadmap_epigenomics/roadmap_sort"
    embedsDir = "./data/roadmap_epigenomics/embeds"

    infSystem = BatchInferHyenaDNA()
    ids = os.listdir(intervalDir)
    sourceDatasets = list()
    outPaths = list()

    print("Starting inference on", len(ids), "bed files.")
    for name in ids:
        embedFile = os.path.join(embedsDir, name + ".npy")
        outPaths.append(embedFile)

        bedFile = os.path.join(intervalDir, name)

        dataset = TokenizedDataset(
            FastaDataset(
                fastaPath,
                BedDataset(
                    bedFile,
                    inputsInMemory,
                    bufferSize=bufferSize,
                    maxLen=seqMaxLen
                )
            ),
            padToLength=seqMaxLen
        )

        sourceDatasets.append(dataset)

    infSystem.batchInfer(sourceDatasets, outPaths, batchSize, workers)


if __name__ == '__main__':
    main()