#Neighborhood Preserving Test

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
import argparse

def genome_distance(r, r_tilde, high_number=1e9):
    c, s, e = r
    c_tilde, s_tilde, e_tilde = r_tilde

    s, e, s_tilde, e_tilde = int(s), int(e), int(s_tilde), int(e_tilde)

    if c == c_tilde:
        return max(s - e_tilde, s_tilde - e, 0)
    else:
	return high_number

def prepare_bed_data(regions):
    return np.array([[chromosome, int(start), int(end)] for chromosome, start, end in regions])

def calculate_qNPR(B, Q, K, high_number=1e9):
    Q = np.array(Q)
    overlaps = []

    # Create a NearestNeighbors object for embedding space using cosine distance
    nbrs_embedding = NearestNeighbors(n_neighbors=K, metric='cosine').fit(Q)

    for i in range(len(Q)):
        # Find K-nearest neighbors in genome space
        distances_genome = []
        for j in range(len(B)):
            if i != j:
                distances_genome.append((genome_distance(B[i], B[j], high_number), j))
        distances_genome.sort(key=lambda x: x[0])
        indices_genome = [idx for _, idx in distances_genome[:K]]

        # Find K-nearest neighbors in embedding space
        distances_embedding, indices_embedding = nbrs_embedding.kneighbors([Q[i]])

        # Calculate overlap ratio
        overlap = len(set(indices_genome).intersection(set(indices_embedding[0]))) / K
        overlaps.append(overlap)

    qNPR = np.mean(overlaps)
    return qNPR

def create_pairs(regions, embeddings):
    num_regions = len(regions)
    if num_regions > len(regions) // 2:
        num_regions = len(regions) // 2

    region_pairs = [(regions[i], regions[i + 1]) for i in range(0, 2 * num_regions, 2)]
    embedding_pairs = [(i, i + 1) for i in range(0, 2 * num_regions, 2)]

    return region_pairs, embedding_pairs

def main():
    parser = argparse.ArgumentParser(description='Neighborhood Preserving Test')
    parser.add_argument('-k', type=int, required=True, help='Number of nearest neighbors')
    parser.add_argument('-b', type=str, required=True, help='Path to BED file')
    parser.add_argument('-q', type=str, required=True, help='Path to embedding file')

    args = parser.parse_args()

    K = args.k
    bed_file = args.b
    query_file = args.q

    regions = []
    with open(bed_file) as f:
        for line in f:
            columns = line.strip().split()
            chromosome, start, end = columns[0], int(columns[1]), int(columns[2])
            regions.append((chromosome, start, end))

    B = prepare_bed_data(regions)
    Q = np.load(query_file)

    # Ensure the number of regions and embeddings match
    region_pairs, embedding_pairs = create_pairs(regions, Q)

    qNPR = calculate_qNPR(B, Q, K)
    print(f"qNPR: {qNPR}")

if __name__ == "__main__":
    main()

