import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import linregress


def gdst(embeds, intervals, considerLimit=None):
    """
    Genome Distance Scaling Test (GDST):

    - Analyzes how embedding distances correlate with genome distances between regions
    - Positive score indicates embeddings capture biological tendency of nearby regions to have similar functions
    - Requires genomic coordinates but not training data

    High Level:
        1. Randomly sample pairs of regions
        2. Calculate embedding distance (ED) and genome distance (GD) for each pair
        3. Fit a linear curve to (GD, ED) points
        4. GDST score is the slope of this fitted line

    Notes:
        - Uses cosine distance for embedding distance to avoid magnitude effects
        - Genome distance is calculated as base pair distance on same chromosome, infinity for different chromosomes
    """

    pointCount = min(len(embeds), len(intervals))

    byChrm = dict()
    for i, (chrm, _, _) in enumerate(intervals):
        if chrm not in byChrm:
            byChrm[chrm] = []
        byChrm[chrm].append(i)

    if considerLimit is None:
        considerLimit = 0

        for chrm in byChrm:
            chrmIndices = byChrm[chrm]
            size = len(chrmIndices)
            considerLimit += size * (size + 1) // 4

    print(
        f"Considering {considerLimit} random comparisons for {pointCount} points.")

    pairs = []
    while len(pairs) < considerLimit:
        chrm = np.random.choice(list(byChrm.keys()))
        chrmIndices = byChrm[chrm]
        pair = np.random.choice(chrmIndices, 2, replace=False)
        pairs.append(pair)

    # Calculate embedding distances (ED)
    # embedDists = np.array([cosine(embeds[i], embeds[j]) for i, j in pairs])
    embedDists = np.array([euclidian(embeds[i], embeds[j]) for i, j in pairs])

    # Calculate genome distances (GD)
    genomDists = np.array([genome_distance(intervals[i], intervals[j])
                           for i, j in pairs])

    # Compute correlation
    correl = np.corrcoef(genomDists, embedDists)[0, 1]

    # Filter for finite distances
    finiteMask = np.isfinite(genomDists)
    assert finiteMask.sum() == len(genomDists), "All genome distances should be finite."

    # Fit linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        genomDists, embedDists)

    print(f"GDST Score (slope): {slope:.6f}")
    print(f"Correlation: {correl:.6f}")

    return {
        'Slope': slope,
        'Correlation': correl
    }


def same_chromosome(i1, i2, intervals):
    chr1, _, _ = intervals[i1]
    chr2, _, _ = intervals[i2]
    return chr1 == chr2


def genome_distance(interval1, interval2):
    """
    Calculate the genome distance between two intervals.
    Assumes intervals are in the format (chromosome, start, end).
    """
    chr1, start1, end1 = interval1
    chr2, start2, end2 = interval2

    if chr1 != chr2:
        return np.inf
    else:
        dist = max(start2 - end1, start1 - end2, 0)
        # TODO: artifically scaling genom dist to compensate for cosine distance
        # Using an arbitrary constant, geniml uses 1e10
        dist = dist / 1e6
        return dist


def euclidian(v1, v2):
    return np.linalg.norm(v1 - v2)
