#Neighborhood Preserving Test

import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse

def calculate_overlap_ratio(BK, TK):
    overlap = np.intersect1d(BK, TK)
    return len(overlap) / len(BK)

def calculate_qNPR(B, Q, K):
    qNPR_sum = 0
    for i in range(len(Q)):
        query_embedding = Q[i]

        # Calculate K-nearest neighbors in genome space
        nbrs_genome = NearestNeighbors(n_neighbors=K, metric='minkowski').fit(B)
        distances_genome, indices_genome = nbrs_genome.kneighbors([query_embedding])

        # Calculate K-nearest neighbors in embeddings space
        nbrs_embeddings = NearestNeighbors(n_neighbors=K, metric='cosine').fit(Q)
        distances_embeddings, indices_embeddings = nbrs_embeddings.kneighbors([query_embedding])

        # Calculate overlap ratio
        overlap_ratio = calculate_overlap_ratio(indices_genome[0], indices_embeddings[0])
        qNPR_sum += overlap_ratio

    return qNPR_sum / len(Q)

def calculate_rNPR(Q, K):
    np.random.seed(42)
    random_embeddings = np.random.uniform(low=-0.5, high=0.5, size=Q.shape)
    return calculate_qNPR(random_embeddings, Q, K)

def calculate_SNPR(qNPR, rNPR):
    return np.log10(qNPR / rNPR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate SNPR for given data.")
    parser.add_argument("-k", type=int, help="Number of nearest neighbors (K).", required=True)
    parser.add_argument("-b", type=str, help="File containing data points in genome space (B).", required=True)
    parser.add_argument("-q", type=str, help="File containing data points in embedding space (Q).", required=True)
    args = parser.parse_args()

    try:
        B = np.loadtxt(args.b)
        Q = np.loadtxt(args.q)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    K = args.k

    qNPR = calculate_qNPR(B, Q, K)
    rNPR = calculate_rNPR(Q, K)
    SNPR = calculate_SNPR(qNPR, rNPR)

    print("qNPR:", qNPR)
    print("rNPR:", rNPR)
    print("SNPR:", SNPR)
