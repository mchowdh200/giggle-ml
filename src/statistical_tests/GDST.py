import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import linregress


def gdst(embeds, intervals, considerLimit=100000):
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

    considerLimit = min(considerLimit, len(embeds) * (len(embeds) - 1) // 2)
    print(f"Using {considerLimit} pairs of regions")

    # Sample pairs of regions
    indices = np.random.choice(len(embeds), (considerLimit, 2))

    # Calculate embedding distances (ED)
    embedDists = np.array([cosine(embeds[i], embeds[j]) for i, j in indices])

    # Calculate genome distances (GD)
    genomDists = np.array([genome_distance(intervals[i], intervals[j])
                           for i, j in indices])

    # Compute correlation
    correl = np.corrcoef(genomDists, embedDists)[0, 1]

    # Filter for finite distances
    finite_mask = np.isfinite(genomDists)
    embedDists = embedDists[finite_mask]
    genomDists = genomDists[finite_mask]

    # Fit linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        genomDists, embedDists)

    print(f"GDST Score (slope): {slope:.6f}")
    print(f"Correlation: {correl:.6f}")

    return {
        'Slope': slope,
        'Correlation': correl
    }


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
        return max(start2 - end1, start1 - end2, 0)
