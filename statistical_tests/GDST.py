import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics.pairwise import cosine_distances

def load_embeddings(file_path):
    """
    Load region embeddings from a file.

    Args:
    file_path (str): Path to the file containing region embeddings.

    Returns:
    numpy.ndarray: Array of region embeddings.
    """
    try:
        embeddings = np.load(file_path)
        return embeddings
    except FileNotFoundError:
        print("Error: Embeddings file not found.")
        exit(1)

def load_regions(file_path):
    """
    Load region information from a file.

    Args:
    file_path (str): Path to the file containing region information.

    Returns:
    list: List of tuples, each containing chromosome name, start position, and end position.
    """
    regions = []
    try:
        with open(file_path) as f:
            for line in f:
                columns = line.strip().split()
                chromosome, start, end = columns[0], int(columns[1]), int(columns[2])
                regions.append((chromosome, int(start), int(end)))
    except FileNotFoundError:
        print("Error: Regions file not found.")
        exit(1)

    return regions

def generate_region_pairs(num_regions, regions):
    """
    Generate pairs of random regions.

    Args:

    num_regions (int): Number of region pairs to generate.
    regions (list): List of tuples containing chromosome name, start position, and end position.

    Returns:
    list: List of tuples, each containing two distinct region tuples.
    """

    if num_regions > len(regions) // 2:
        num_regions = len(regions) //2
    region_pairs = [(regions[i], regions[i+1]) for i in range(0, 2*num_regions, 2)]
    embedding_pairs = [(i, i+1) for i in range(0, 2*num_regions, 2)]


    return region_pairs, embedding_pairs

def calculate_gd(region_pairs, high_number=1e9):
    """
    Calculate Genome Distance (GD) between pairs of regions.

    Args:
    region_pairs (list): List of region pairs, each containing two region tuples.
    high_number (float): A very high number to assign to incomparable regions.

    Returns:
    numpy.ndarray: Array of GD values.
    """
    gd_values = []
    for pair in region_pairs:
        r, r_tilde = pair
        c, s, e = r
        c_tilde, s_tilde, e_tilde = r_tilde

        if c == c_tilde:
            gd_values.append(max(s - e_tilde, s_tilde - e, 0))
        else:
            gd_values.append(high_number)

    return np.array(gd_values)

def calculate_ed(embedding_pairs, embeddings):
    """
    Calculate Euclidean Distance (ED) between pairs of region embeddings.

    Args:
    region_pairs (list): List of region pairs, each containing two region tuples.
    embeddings (numpy.ndarray): Array of region embeddings.

    Returns:
    numpy.ndarray: Array of ED values.
    """
    ed_values = []
    for pair in embedding_pairs:
        idx1, idx2 = pair
        r_embedding, r_tilde_embedding = embeddings[idx1], embeddings[idx2]
        ed = cosine_distances([r_embedding], [r_tilde_embedding])[0][0]
        if np.isnan(ed) or np.isinf(ed):  # Check if ED is NaN or infinity
            ed = 0  # Set invalid ED values to 0
        ed_values.append(ed)

    return np.array(ed_values)

def linear_curve_fit(x, m, b):
    """
    Linear curve fitting function.

    Args:
    x (numpy.ndarray): Array of x-values.
    m (float): Slope of the curve.
    b (float): Intercept of the curve.

    Returns:
    numpy.ndarray: Array of y-values.
    """
    return m * x + b

def calculate_gdst_score(ed_values, gd_values):
    """
    Calculate the Genome Distance Scaling Test (GDST) score.

    Args:
    ed_values (numpy.ndarray): Array of Euclidean Distance (ED) values.
    gd_values (numpy.ndarray): Array of Genome Distance (GD) values.

    Returns:
    float: GDST score.
    """
    # Perform linear curve fitting
    popt, _ = curve_fit(linear_curve_fit, ed_values, gd_values)

    # Extract slope from fitted curve
    gdst_score = popt[0]
    return gdst_score

def main(embeddings_file, regions_file, num_regions=100):
    # Load embeddings and regions
    embeddings = load_embeddings(embeddings_file)
    regions = load_regions(regions_file)

    # Generate region pairs and embedding pairs
    region_pairs, embedding_pairs = generate_region_pairs(num_regions, regions)

    # Calculate GD and ED values
    gd_values = calculate_gd(region_pairs)
    ed_values = calculate_ed(embedding_pairs, embeddings)

    # Calculate GDST score
    gdst_score = calculate_gdst_score(ed_values, gd_values)
    print("GDST score:", gdst_score)

# Example usage:
if __name__ == "__main__":
    embeddings_file = 'path_to_embeddings.npy'
    regions_file = 'path_to_regions.txt'
    num_regions = 100
    main(embeddings_file, regions_file, num_regions)
