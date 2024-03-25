import argparse
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
        embeddings = np.loadtxt(file_path)
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
                chromosome, start, end = line.strip().split()
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
    np.random.shuffle(regions)
    region_pairs = [(regions[i], regions[i+1]) for i in range(0, len(regions), 2)]

    return region_pairs

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

def calculate_ed(region_pairs, embeddings):
    """
    Calculate Euclidean Distance (ED) between pairs of region embeddings.

    Args:
    region_pairs (list): List of region pairs, each containing two region tuples.
    embeddings (numpy.ndarray): Array of region embeddings.

    Returns:
    numpy.ndarray: Array of ED values.
    """
    ed_values = []
    for pair in region_pairs:
        r_embedding, r_tilde_embedding = embeddings[np.random.choice(embeddings.shape[0], 2, replace=False)]
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Region Distance Analysis")
    parser.add_argument("--num_regions", type=int, default=100, help="Number of region pairs to generate")
    parser.add_argument("--embeddings_file", type=str, help="Path to the file containing region embeddings")
    parser.add_argument("--regions_file", type=str, help="Path to the file containing region information")
    args = parser.parse_args()

    if args.embeddings_file:
        embeddings = load_embeddings(args.embeddings_file)
    else:
        print("Error: Please provide the path to the embeddings file.")
        exit(1)

    if args.regions_file:
        regions = load_regions(args.regions_file)
    else:
        print("Error: Please provide the path to the regions file.")
        exit(1)

    region_pairs = generate_region_pairs(args.num_regions, regions)
    print("Region pairs:", region_pairs)

    gd_values = calculate_gd(region_pairs)
    print("GD values:", gd_values)

    ed_values = calculate_ed(region_pairs, embeddings)
    print("ED values:", ed_values)

    gdst_score = calculate_gdst_score(ed_values, gd_values)
    print("GDST score:", gdst_score)
