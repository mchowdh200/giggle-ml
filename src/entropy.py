import numpy as np
import os
from utils.basic_logger import BasicLogger
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.special import softmax
from scipy.stats import entropy
from gpu_embeds.inference_batch import BatchInferHyenaDNA
from types import SimpleNamespace
from data_wrangling.seq_datasets import BedDataset
from data_wrangling.seq_datasets import TokenizedDataset
from data_wrangling.seq_datasets import FastaDataset


def entropy_by_distance(embeds: np.ndarray, aggregateEmbed: np.ndarray) -> float:
    """
    Compute entropy based on distances from aggregate embedding.

    Args:
        embeds: Array of shape (n_samples, dim)
        aggregateEmbed: Array of shape (dim,)

    Returns:
        Normalized entropy value between 0 and 1
    """
    # Compute cosine similarities
    similarities = np.array([1 - cosine(embed, aggregateEmbed)
                             for embed in embeds])

    # Apply softmax to get probability distribution
    probs = softmax(similarities)

    # Compute entropy and normalize by log(n) (maximum possible entropy)
    return entropy(probs) / np.log(len(embeds))


def entropy_by_components(embeds: np.ndarray,
                          aggregateEmbed: np.ndarray,
                          n_bins: int = 10) -> float:
    """
    Compute entropy based on component-wise distributions.

    Args:
        embeds: Array of shape (n_samples, dim)
        aggregateEmbed: Array of shape (dim,)
        n_bins: Number of bins for histogram

    Returns:
        Normalized entropy value between 0 and 1
    """
    dim = embeds.shape[1]
    probs = []

    # For each dimension
    for dim in range(dim):
        # Get distribution of values for this dimension
        hist, histEdges = np.histogram(embeds[:, dim], bins=n_bins,
                                       density=True)

        # Find which bin the aggregate embedding's component falls into
        idx = np.digitize(aggregateEmbed[dim], histEdges) - 1

        # Edge case: if the aggregate value is beyond the distribution,
        # consider its probability zero
        if idx == n_bins:
            probs.append(0)
            continue

        # Get probability for this component
        probs.append(hist[idx])

    # Convert to numpy array and apply softmax
    probs = softmax(probs)

    # Compute entropy and normalize by log(dim)
    return entropy(probs) / np.log(dim)


if __name__ == "__main__":
    intervalSetLimit = 100
    intervalLimit = 1000

    pad = 500
    batchSize = 500
    workers = 2
    makeNewEmbeds = False

    paths = SimpleNamespace(
        fasta="./data/hg19.fa",
        samples="./data/roadmap_epigenomics/roadmap_sort/",
        embeds="./data/roadmap_epigenomics/embeds/",
        out="./experiments/entropy_analysis/roadmap_epigenomics.md")

    info = BasicLogger()
    info.print("pad =", pad)
    info.print("intervalLimit =", intervalLimit)

    infSystem = BatchInferHyenaDNA()
    dim = infSystem.embedDim
    embeds = list()

    for sampleName in os.listdir(paths.samples)[:intervalSetLimit]:
        embedPath = os.path.join(paths.embeds, sampleName + ".npy")

        if makeNewEmbeds:
            samplePath = os.path.join(paths.samples, sampleName)

            dataset = TokenizedDataset(
                padToLength=pad,
                fastaDataset=FastaDataset(
                    fastaPath=paths.fasta,
                    bedDataset=BedDataset(
                        bedPath=samplePath,
                        limit=intervalLimit,
                        inMemory=True,
                        bufferSize=30)))

            infSystem.batchInfer(dataset, embedPath, batchSize, workers)

    for embedName in os.listdir(paths.embeds)[:intervalSetLimit]:
        embedPath = os.path.join(paths.embeds, embedName)
        memMap = np.memmap(embedPath, dtype=np.float32, mode="r")
        memMap = memMap.reshape(-1, dim)
        embeds.append(memMap)

    meanEmebds = list()

    for embedSeries in embeds:
        meanEmebds.append(embedSeries.mean(axis=0))

    entropy1 = list()
    entropy2 = list()

    for i in range(len(embeds)):
        entropy1.append(entropy_by_distance(embeds[i], meanEmebds[i]))
        entropy2.append(entropy_by_components(embeds[i], meanEmebds[i]))

    info.print("Entropies computed for", len(entropy1), "interval sets")

    entropy1 = pd.Series(entropy1)
    entropy2 = pd.Series(entropy2)

    info.print("# Entropy statistics")
    info.print("## By distance")
    info.print(entropy1.describe())
    info.print()

    info.print("## Component-wise")
    info.print(entropy2.describe())
    info.print()

    info.print("# Sample entropies")
    info.print("## By distance")
    info.print(entropy1[:5])
    info.print("## Component-wise")
    info.print(entropy2[:5])

    print(info)
    info.save(paths.out)
