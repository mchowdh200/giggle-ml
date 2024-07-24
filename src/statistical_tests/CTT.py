import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def ctt(embeds, testPointRate=.1, considerationLimit=10000):
    """
    Cluster Tendency Test (CTT):
        Derived from the Hopkins Statistic

    - Evaluates how well region embeddings can form clusters
    - Higher score indicates greater tendency to cluster
    - Only requires the embeddings themselves, not training data

    High Level:
        (After initial subsampling)
        1. Subsample (again) for test points.
        2. For each, get **single** nearest neighbor.
        3. Sum these distances and distances to a random(ly generated, not sampled) point
        4. Call the neighbor distances D_T and random D_R, CTT = D_R / (D_R + D_T)

    Notes:
        Initial Subsampling:
        "To reduce the computational complexity, we first sample N_S = min(10^4, N) region embeddings from the original N embeddings."

        **Single** nearest neighbor to test point:
        The paper does not justify this decision, but it was likely for performance reasons.
    """

    # TODO: scipy::euclidean_distances room for optimization?
    # And currently naive KNN.

    # Sample embeds if necessary
    if len(embeds) > considerationLimit:
        indices = np.random.choice(
            len(embeds), considerationLimit, replace=False)
        embeds = embeds[indices]

    # Further subsample for test points
    testPointAmnt = int(len(embeds) * testPointRate)
    testPointIndices = np.random.choice(
        len(embeds), testPointAmnt, replace=False)
    testPoints = embeds[testPointIndices]

    # Calculate distances for test embeddings
    # TODO: consider optimization with KNN (kd/ball?) trees
    distTests = euclidean_distances(testPoints, embeds)  # matrix: all combos
    # Account for self-dist
    distTests[np.arange(len(testPoints)), testPointIndices] = np.inf
    d_t = np.sum(np.min(distTests, axis=1))  # Get NN & sum dists at once

    # Generate random points
    embedMin = np.min(embeds, axis=0)
    embedMax = np.max(embeds, axis=0)
    embedDim = embeds.shape[1]
    # TODO: adjust min max based on percetile to account
    # for distribution shape
    randomPoints = np.random.uniform(
        embedMin, embedMax, (len(testPoints), embedDim))

    # Calculate distances for random points
    distsRandom = euclidean_distances(randomPoints, embeds)
    d_r = np.sum(np.min(distsRandom, axis=1))

    # Calculate CTT score
    score = d_r / (d_r + d_t)
    print(f"CTT: {score}")
    return score
