import numpy as np
from gpu_embeds.inference_batch import BatchInferHyenaDNA
from types import SimpleNamespace
from matplotlib import pyplot as plt
import faiss


def main():
    k = 100
    trials = 1000
    renormalize = True

    paths = SimpleNamespace(
        fasta="./data/hg38.fa",
        sampleBed="./data/giggleBench/sample.bed",
        sampleEmbeds="./data/giggleBench/embeds/straight/sample.npy",
        outDir="embedAnalysis/umap")

    infSystem = BatchInferHyenaDNA()
    dim = infSystem.embedDim

    embeds = np.memmap(paths.sampleEmbeds, dtype=np.float32, mode="r")
    embeds = embeds.reshape(-1, dim)

    vdb = None
    if renormalize:
        vdb = faiss.IndexFlatIP(dim)
    else:
        vdb = faiss.IndexFlatL2(dim)

    for embed in embeds:
        if renormalize:
            embed = embed / np.linalg.norm(embed)
        embed = embed.reshape(1, -1)
        vdb.add(embed)

    resultsTruth = []
    resultsBaseline = []

    for _ in range(trials):
        query1 = np.random.randint(0, len(embeds))
        query2 = np.random.randint(0, len(embeds))
        truthEmbeds = embeds[[query1, query2]].reshape(2, -1)

        embedMerge = np.mean(truthEmbeds, axis=0)
        if renormalize:
            embedMerge = embedMerge / np.linalg.norm(embedMerge)
        embedMerge = embedMerge.reshape(1, -1)

        queryBaseline = np.random.randint(0, len(embeds))
        embedBaseline = embeds[queryBaseline].reshape(1, -1)

        # search

        _, knnTruth = vdb.search(truthEmbeds, k)
        knnTruth = knnTruth.flatten()

        _, knnMerge = vdb.search(embedMerge, k * 2)
        knnMerge = knnMerge.flatten()

        _, knnBaseline = vdb.search(embedBaseline, k * 2)
        knnBaseline = knnBaseline.flatten()

        recallTruth = len(set(knnTruth) & set(knnMerge)) / len(knnTruth)
        recallBaseline = len(set(knnTruth) & set(
            knnBaseline)) / len(knnBaseline)

        resultsTruth.append(recallTruth)
        resultsBaseline.append(recallBaseline)

    recallTruth = round(np.mean(resultsTruth), 3)
    recallBaseline = round(np.mean(resultsBaseline), 3)

    print("Recall:", recallTruth)
    print("Random Recall:", recallBaseline)
    print("% Improvement:", recallTruth / recallBaseline - 1)


if __name__ == "__main__":
    main()
