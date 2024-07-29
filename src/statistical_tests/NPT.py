import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import cKDTree


def npt(embeds, intervals, k=100, testPointRate=.1):
    """
    Neighborhood Preserving Test (NPT):

    - Evaluates how well the genomic neighborhood of regions is preserved in embedding space
    - Calculates overlap between k-nearest neighbors in genome space vs embedding space
    - Requires genomic coordinates but not training data

    High Level:
        1. Sample regions
        2. For each sampled region, find K-nearest neighbors in the
            (total, not sampled) genome space and embedding space.
        3. Calculate overlap ratio between these two sets of neighbors
        4. Average overlap ratios to get Neighborhood Preserving Ratio (NPR)
        5. Compare NPR to that of random embeddings to get significance (SNPR)
        6. Typically calculated for various K values
    """

    querySize = int(testPointRate * len(embeds))
    sampleIndices = np.random.choice(len(embeds), querySize, replace=False)

    queryEmbeds = [None] * querySize
    queryIntervals = [None] * querySize

    intervals = list(map(lambda x: (x[1], x[2]), intervals))

    for i, idx in enumerate(sampleIndices):
        queryEmbeds[i] = embeds[idx]
        # TODO: separate kdTree for each chromosome
        queryIntervals[i] = intervals[idx]

    # Build KD-trees for the full datasets
    embedTree = cKDTree(embeds)
    intervalTree = cKDTree(intervals)

    # Find K-nearest neighbors in both spaces
    npr = get_npr(embedTree, intervalTree,
                  queryEmbeds, queryIntervals, k)

    # Calculate NPR for random embeddings
    randomEmbeds = np.random.rand(*embeds.shape)
    randomTree = cKDTree(randomEmbeds)
    nprRandom = get_npr(randomTree, intervalTree,
                        queryEmbeds, queryIntervals, k)

    # Calculate SNPR
    snpr = np.log10(npr / nprRandom)

    print(f"NPR: {npr:.4f}")
    print(f"Random NPR: {nprRandom:.4f}")
    print(f"SNPR: {snpr:.4f}")

    return {
        'NPR': npr,
        'Random NPR': nprRandom,
        'SNPR': snpr
    }


def get_npr(tree1, tree2, queries1, queries2, k):
    """
    Calculate the NPR between two KD-trees
    """
    # print(queries1, queries2)
    _, neighbors1 = tree1.query(queries1, k=k+1)
    # print(neighbors1)
    _, neighbors2 = tree2.query(queries2, k=k+1)
    # print(neighbors2)
    overlap = [set(n1[1:]) & set(n2[1:])
               for n1, n2 in zip(neighbors1, neighbors2)]
    # print(overlap)
    percentOverlap = [len(o) / k for o in overlap]
    # print()
    # print()
    return np.mean(percentOverlap)
