import argparse
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
def read_embeddings_from_file(file_path):
    with open(file_path, 'r') as file:
        embeddings = []
        for line in file:
            embedding = np.array(list(map(float, line.strip().split())))
            embeddings.append(embedding)
    return np.array(embeddings)

# Function to check intersection of intervals using Monte Carlo simulation
def check_intersection(interval_set1, interval_set2, num_points=8, num_trials=1):
    intersections = 0
    for _ in range(num_trials):
        for interval1 in interval_set1:
            points_interval1 = np.random.uniform(interval1[1], interval1[2], num_points)
            for interval2 in interval_set2:
                for point in points_interval1:
                    if interval2[1] <= point <= interval2[2] and interval1[0] == interval2[0]:
                        intersections += 1
                        break
    probability = intersections / (num_trials * len(interval_set1) * len(interval_set2))
    return probability

# Function to check cosine similarity of embeddings using Monte Carlo simulation
def check_similarity(embeddings1, embeddings2, threshold=0.1, num_trials=10):
    intersections = 0
    for _ in range(num_trials):
        random_embedding = embeddings1[np.random.randint(len(embeddings1))]
        for embedding in embeddings2:
            similarity = cosine_similarity([random_embedding], [embedding])[0][0]
            if similarity > threshold:
                intersections += 1
                break
    probability = intersections / num_trials
    return probability

def main():
    parser = argparse.ArgumentParser(description='Monte Carlo simulations for interval and embedding intersections')
    parser.add_argument('--interval_file1', required=True, help='Path to the first interval file')
    parser.add_argument('--interval_file2', required=True, help='Path to the second interval file')
    parser.add_argument('--embedding_file1', required=True, help='Path to the first embedding file')
    parser.add_argument('--embedding_file2', required=True, help='Path to the second embedding file')
    parser.add_argument('--num_points', type=int, default=8, help='Number of points for interval intersection')
    parser.add_argument('--num_trials', type=int, default=1, help='Number of trials for interval intersection')
    parser.add_argument('--threshold', type=float, default=0.1, help='Cosine similarity threshold for embedding intersection')
    parser.add_argument('--num_trials_embeddings', type=int, default=10, help='Number of trials for embedding intersection')
    
    args = parser.parse_args()

    # Read interval sets from files
    interval_set1 = read_intervals_from_file(args.interval_file1)
    interval_set2 = read_intervals_from_file(args.interval_file2)

    # Read embeddings from files
    embeddings1 = read_embeddings_from_file(args.embedding_file1)
    embeddings2 = read_embeddings_from_file(args.embedding_file2)

    # Run the Monte Carlo simulations
    probability_intervals = check_intersection(interval_set1, interval_set2, num_points=args.num_points, num_trials=args.num_trials)
    probability_embeddings = check_similarity(embeddings1, embeddings2, threshold=args.threshold, num_trials=args.num_trials_embeddings)

    # Print the probabilities
    print("Probability of intersection (intervals):", probability_intervals)
    print("Probability of intersection (embeddings):", probability_embeddings)

if __name__ == '__main__':
    main()
