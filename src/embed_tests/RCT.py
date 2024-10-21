import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score


def rct(embeds, seqs, folds=5):
    """
    Reconstruction Test (RCT):

    - Evaluates how much of the original training information is preserved in the embeddings
    - Uses a regression task to predict original binary data from embeddings
    - Requires access to both embeddings and original sequence nucleotides

    High Level:
        1. Perform K-fold cross-validation
        2. For each fold:
            a. Train a neural network to predict input sequences from embeddings
            b. Calculate R^2 score on the test set
        3. RCT score is the average R^2 across folds

    Notes:
        - Uses a simple multi-layer perceptron for the regression task
        - Paper uses a 200 hidden layer with ReLU activations
    """

    kf = KFold(n_splits=folds, shuffle=True)
    scores = []

    for i, (train_index, test_index) in enumerate(kf.split(embeds)):
        X_train = subseq(embeds, train_index)
        X_test = subseq(embeds, test_index)
        y_train = subseq(seqs, train_index)
        y_test = subseq(seqs, test_index)

        # Initialize and train the model
        model = MLPRegressor(hidden_layer_sizes=(
            200,), max_iter=1000, activation='relu')

        print(f"Training Fold \t{i + 1} / {folds}")
        model.fit(X_train, y_train)

        # Predict and calculate R^2 score
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        scores.append(r2)

    rct = np.mean(scores)
    stddev = np.std(scores)

    print(f"RCT Score: {rct:.4f}")
    print(f"Std Dev: {stddev:.4f}")

    return {
        'Rct': rct,
        'StdDev': np.std(scores)
    }


def subseq(data, indices):
    return np.array([data[i] for i in indices])


def dna2float(seqs):
    map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    array = [[map[c] for c in seq] for seq in seqs]
    return array
