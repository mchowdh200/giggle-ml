import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Function to read intervals from a file
def read_intervals_from_file(file_path):
    interval_set = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split()
            chromosome = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            interval = (chromosome, start, end)
            interval_set.append(interval)
    return interval_set

# Function to read embeddings from a file
def read_embeddings_from_file(embedding_file):
    embeddings = np.load(embedding_file)
    return embeddings

# Function to check intersection of intervals using Monte Carlo simulation
def check_intersection(interval_set, num_points=8, num_trials=1):
    intersections = 0
    for _ in range(num_trials):
        for i in range(len(interval_set)):
            interval1 = interval_set[i]
            points_interval1 = np.random.uniform(interval1[1], interval1[2], num_points)
            for j in range(i + 1, len(interval_set)):
                interval2 = interval_set[j]
                if interval1[0] == interval2[0]:  # Ensure they are on the same chromosome
                    for point in points_interval1:
                        if interval2[1] <= point <= interval2[2]:
                            intersections += 1
                            break
    # Update the denominator to account for the reduced number of comparisons
    probability = intersections / (num_trials * len(interval_set) * (len(interval_set) - 1) / 2)
    return probability

# Function to check cosine similarity of embeddings using Monte Carlo simulation
def check_similarity(embeddings, threshold=0.1, num_trials=1):
    intersections = 0
    for _ in range(num_trials):
        for i in range(len(embeddings)):
            embedding1 = embeddings[i]
            for j in range(i + 1, len(embeddings)):
                embedding2 = embeddings[j]
                similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                if similarity > threshold:
                    intersections += 1
                    break  # Move to the next embedding1
    # Calculate the probability based on the number of comparisons
    n = len(embeddings)
    probability = intersections / (num_trials * n * (n - 1) / 2)  # Divided by 2 to avoid double counting
    return probability

def main(interval_file, embedding_file, num_points=8, num_trials_intervals=1, threshold=0.1, num_trials_embeddings=10):
    # Read interval sets from files
    interval_set = read_intervals_from_file(interval_file)

    # Read embeddings from files
    embeddings = read_embeddings_from_file(embedding_file)

    # Run the Monte Carlo simulations
    probability_intervals = check_intersection(interval_set, num_points=num_points, num_trials=num_trials_intervals)
    probability_embeddings = check_similarity(embeddings, threshold=threshold, num_trials=num_trials_embeddings)

    # Print the probabilities
    print("Probability of intersection (intervals):", probability_intervals)
    print("Probability of intersection (embeddings):", probability_embeddings)

# Example usage:
if __name__ == '__main__':
    interval_file = 'test.bed'
    embedding_file = 'test.npy'
    main(interval_file, embedding_file)
