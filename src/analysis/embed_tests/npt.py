import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import cKDTree


def npt(embeds, intervals, k=100, test_point_rate=.1):
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

    consider_limit = min(len(embeds), len(intervals))
    query_size = int(test_point_rate * consider_limit)
    sample_indices = np.random.choice(consider_limit, query_size, replace=False)

    query_embeds = [None] * query_size
    query_intervals = [None] * query_size

    intervals = list(map(lambda x: (x[1], x[2]), intervals))

    for i, idx in enumerate(sample_indices):
        query_embeds[i] = embeds[idx]
        # TODO: separate kdTree for each chromosome
        query_intervals[i] = intervals[idx]

    # Build KD-trees for the full datasets
    embed_tree = cKDTree(embeds)
    interval_tree = cKDTree(intervals)

    # Find K-nearest neighbors in both spaces
    npr = get_npr(embed_tree, interval_tree,
                  query_embeds, query_intervals, k)

    # Calculate NPR for random embeddings
    random_embeds = np.random.rand(*embeds.shape)
    random_tree = cKDTree(random_embeds)
    npr_random = get_npr(random_tree, interval_tree,
                        query_embeds, query_intervals, k)

    # Calculate SNPR
    snpr = np.log10(npr / npr_random)

    print(f"NPR: {npr:.4f}")
    print(f"Random NPR: {npr_random:.4f}")
    print(f"SNPR: {snpr:.4f}")

    return {
        'NPR': npr,
        'Random NPR': npr_random,
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
    percent_overlap = [len(o) / k for o in overlap]
    # print()
    # print()
    return np.mean(percent_overlap)
