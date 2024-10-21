import numpy as np
from data_wrangling.seq_datasets import BedDataset
from data_wrangling.transform_dataset import TransformDataset
from gpu_embeds.inference_batch import BatchInferHyenaDNA
from types import SimpleNamespace
import umap
import umap.plot
from matplotlib import pyplot as plt


def main():
    k = 90

    paths = SimpleNamespace(
        fasta="./data/hg38.fa",
        sampleBed="./data/giggleBench/sample.bed",
        sampleEmbeds="./data/giggleBench/embeds/straight/sample.npy",
        outDir="embedAnalysis/umap")

    infSystem = BatchInferHyenaDNA()
    dim = infSystem.embedDim

    sampleEmbeds = np.memmap(paths.sampleEmbeds, dtype=np.float32, mode="r")
    sampleEmbeds = sampleEmbeds.reshape(-1, dim)

    mapper = umap.UMAP().fit(sampleEmbeds)
    umap.plot.points(mapper)
    plt.savefig(f"{paths.outDir}/umap.png", dpi=300)


if __name__ == "__main__":
    main()
