from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import numpy as np
import argparse


def main(B, Q, num_folds=2):
    """
        Calculate RCT score for given data.
        Parameters:
        B: NpArray, Binary embeddings (labels).
        Q: NpArray, Query embeddings (inputs).
    """

    kf = KFold(n_splits=num_folds)
    r2_scores = []

    for train_index, test_index in kf.split(B):
        X_train, X_test = Q[train_index], Q[test_index]
        y_train, y_test = B[train_index], B[test_index]

        # Define and train the neural network
        model = MLPRegressor(hidden_layer_sizes=(100,), random_state=42)
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate R2 score
        r2 = r2_score(y_test, y_pred)

        r2_scores.append(r2)

        # Calculate the average R2 score across all folds
        rct_score = np.mean(r2_scores)
    return rct_score


if __name__ == "__main__":
    # example usage
    parser = argparse.ArgumentParser(
        description="Calculate RCT score for given data.")
    parser.add_argument(
        "-b", type=str, help="File containing binary embeddings (labels).", required=True)
    parser.add_argument(
        "-q", type=str, help="File containing query embeddings (inputs).", required=True)
    parser.add_argument(
        "-f", type=int, help="Number of folds for K-fold cross-validation.", default=2)
    args = parser.parse_args()

    try:
        B = np.loadtxt(args.b)
        Q = np.loadtxt(args.q)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    num_folds = args.f

    rct_score = main(B, Q, num_folds)
    print("RCT score:", rct_score)
