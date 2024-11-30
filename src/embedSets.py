import os
from data_wrangling.seq_datasets import TokenizedDataset, FastaDataset, BedDataset
from gpu_embeds.inference_batch import BatchInferHyenaDNA


def main():
    limit = None
    workers = 2
    batchSize = int(10e3)
    bufferSize = batchSize * 2
    inputsInMemory = True
    seqMaxLen = 500

    fastaPath = "./data/hg19.fa"
    intervalDir = "./data/roadmap_epigenomics/roadmap_sort"
    embedsDir = "./data/roadmap_epigenomics/embeds"

    infSystem = BatchInferHyenaDNA()
    bedFiles = os.listdir(intervalDir)

    print("Starting inference on", len(bedFiles), "bed files.")
    for i, name in enumerate(bedFiles):
        if limit and i >= limit:
            break

        embedFile = os.path.join(embedsDir, name + ".npy")

        if os.path.exists(embedFile):
            continue

        bedFile = os.path.join(intervalDir, name)
        bedDs = BedDataset(
            bedFile,
            inputsInMemory,
            bufferSize=bufferSize,
            maxLen=seqMaxLen
        )

        fastaDs = FastaDataset(
            fastaPath,
            bedDs
        )

        dataset = TokenizedDataset(
            fastaDs,
            padToLength=seqMaxLen
        )

        infSystem.batchInfer(dataset, embedFile, batchSize, workers)
        print(f"\t== ", i, "/", len(bedFiles))


if __name__ == '__main__':
    main()