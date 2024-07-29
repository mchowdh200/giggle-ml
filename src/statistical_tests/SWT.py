from data_wrangling.sliding_window_dataset import SlidingWindowDataset
from data_wrangling.truncated_dataset import TruncatedDataset
from data_wrangling.seq_datasets import TokenizedDataset

from gpu_embeds.inference_batch import batchInfer
import numpy as np
from matplotlib import pyplot as plt


def swt(embedOutPath, fastaDataset, gapFactor=.1, considerLimit=128):
    swDataset = SlidingWindowDataset(fastaDataset, gapFactor)
    binAmnt = swDataset.spreeCount
    # Round to multiple of binAmnt
    considerLimit = binAmnt * (considerLimit // binAmnt)
    shortDataset = TokenizedDataset(
        TruncatedDataset(swDataset, considerLimit))

    print('Creating embeddings...')
    # TODO: configurable batch size
    batchInfer(shortDataset, embedOutPath, batchSize=binAmnt)
    embeds = np.memmap(embedOutPath, dtype='float32', mode='r')
    embedDim = 256
    embeds = embeds.reshape((-1, embedDim))
    print(' - Success')

    # Analyze results

    bins = [None] * binAmnt

    for i in range(len(shortDataset)):
        binIdx = i % binAmnt
        if bins[binIdx] is None:
            bins[binIdx] = []

        rootIdx = i // binAmnt * binAmnt  # 0th item of bin
        rootEmbed = embeds[rootIdx]
        thisEmbed = embeds[i]

        # TODO: is this the best metric?
        dist = cosine_similarity(rootEmbed, thisEmbed)
        # dist = np.linalg.norm(rootEmbed - thisEmbed)
        bins[binIdx].append(dist)

    avgs = [np.mean(bin) for bin in bins]
    labels = [None] * binAmnt

    for i in range(len(avgs)):
        labels[i] = 100 - round(i * gapFactor * 100)
        print(f'Bin {labels[i]}%: {avgs[i]}')

    plt.plot(labels, avgs)
    plt.title('Intersection Similarity')
    plt.xlabel('Overlap %')
    plt.ylabel('Cosine similarity')
    plt.xticks(labels)
    plt.show()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
