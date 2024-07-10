import numpy as np
from scipy.spatial.distance import cdist

def squared_euclidean_distance(point1, point2):
    return np.sum((point1 - point2) ** 2)

# Function to calculate DT
def calculate_DT(Q, NT):
    DT = 0
    for q_star in Q:
        q_hat = min(Q[np.where(~np.all(Q == q_star, axis=1))], key=lambda x: squared_euclidean_distance(x, q_star))
        DT += squared_euclidean_distance(q_star, q_hat)
    return DT

# Function to calculate DR
def calculate_DR(Q, NT):
    DR = 0
    for _ in range(NT):
        # Generate a random point uniformly distributed within the range of embeddings in Q
        u = np.random.uniform(Q.min(axis=0), Q.max(axis=0))
        # Find the nearest neighbor to u in Q
        q_tilde = min(Q, key=lambda x: squared_euclidean_distance(x, u))
        # Calculate squared Euclidean distance between u and q_tilde
        DR += squared_euclidean_distance(u, q_tilde)
    return DR

def main(file_path, num_iterations=10):
    try:
        N = np.load(file_path)
        NS = min(10000, len(N))  # Limit NS to 10000 or the size of N if it's smaller
        NT = NS // 10  # NT is 10% of NS

        Q = N[np.random.choice(N.shape[0], NS, replace=False)]

        DT_sum = 0
        DR_sum = 0
        for _ in range(num_iterations):
            T = Q[np.random.choice(Q.shape[0], NT, replace=False)]
            DT_sum += calculate_DT(T, NT)
            DR_sum += calculate_DR(Q, NT)

        print("Sum of squared Euclidean distances between test points and their nearest neighbors (DT):", DT_sum)
        print("Sum of squared Euclidean distances between random points and their nearest neighbors in Q (DR):", DR_sum)
        CTT = DR_sum / (DR_sum + DT_sum)
        print("Cluster Tendency Test (CTT) Score:", CTT)
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")

# Example usage:
if __name__ == "__main__":
    file_path = 'path_to_embeddings.npy'
    num_iterations = 10
    main(file_path, num_iterations)
