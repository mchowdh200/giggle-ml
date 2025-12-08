import csv
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import kendalltau

import giggleml.utils.roadmap_epigenomics as rme


def parse_scores_tsv(
    path: Path, name_col: int, score_col: int, ext_strip_count: int = 0
) -> list[float]:
    """
    Expects names in form a/b/name.ext where `name` corresponds to known rme names.
    Will strip an amount of extensions given.

    Validation logic:
    - Expects a header row.
    - Ignores names unknown to `rme` (skipped silently).
    - Raises if a KNOWN Bed ID is duplicated in the file.
    - Raises if the file does not contain a score for every known Bed ID.

    Returns scores ordered by bed ID.
    """
    total_beds = len(rme.bed_names)

    # Initialize with NaNs to easily detect missing values later
    scores = np.full(total_beds, np.nan)
    seen_ids = set()

    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t")

        # 1. Skip Header (Strict expectation)
        try:
            _ = next(reader)
        except StopIteration:
            raise ValueError(f"File {path} is empty (missing header).")

        for row_idx, row in enumerate(reader, start=2):
            if not row:
                continue

            # 2. Extract Name
            # Extract "name" from "a/b/name.ext"
            try:
                raw_path = row[name_col]
                name = Path(raw_path).name
                end = None if ext_strip_count <= 0 else -ext_strip_count
                name_key = ".".join(name.split(".")[:end])
            except IndexError:
                raise ValueError(f"Row {row_idx} missing name column {name_col}")

            # 3. Resolve ID
            try:
                bed_id = rme.bed_id(name_key)
            except Exception:
                # RELAXATION: If the name is unknown to rme, we skip it
                # instead of raising an error.
                continue

            # 4. Check Duplicates (Strict for known IDs)
            if bed_id in seen_ids:
                raise ValueError(
                    f"Duplicate Bed ID found for '{name_key}' at row {row_idx}"
                )

            seen_ids.add(bed_id)

            # 5. Extract Score
            try:
                scores[bed_id] = float(row[score_col])
            except (ValueError, IndexError):
                raise ValueError(f"Invalid score at row {row_idx}")

    # 6. Check Completeness
    if len(seen_ids) != total_beds:
        # Calculate exactly which are missing for the error message
        all_ids = set(range(total_beds))
        missing = all_ids - seen_ids
        raise ValueError(
            # f"Incomplete data. Expected {total_beds} scores, found {len(seen_ids)}. "
            # f"Missing Bed IDs: {missing}"
        )

    return scores.tolist()


def invert_scores(series: list[float]):
    peak = max(series)
    return [peak - x for x in series]


def scores_to_ranks(scores: list[float]) -> list[int]:
    """
    Sorts bed ID indices by score descending.
    Returns a list of Bed IDs, where index 0 is the ID with the highest score.
    """
    # argsort sorts ascending, [::-1] flips to descending
    return np.argsort(scores)[::-1].tolist()


def kendall_tau_matrix(sources: Iterable[tuple[str, list[int]]]):
    """
    Pairwise tau values for all (name, ranks) sources.
    Returns a matplotlib figure.
    """
    source_list = list(sources)
    names = [name for name, _ in source_list]
    rankings = [ranks for _, ranks in source_list]
    n = len(source_list)

    matrix = np.zeros((n, n))

    # Helper: Convert Permutation (List of Item IDs ordered by rank)
    # to Rank Vector (List where index=Item ID, value=Rank).
    # Kendall's Tau requires Rank Vectors.
    def invert_permutation(p):
        p_arr = np.array(p)
        inv = np.empty_like(p_arr)
        # If Item X is at rank i (p[i] = X), then rank of X is i (inv[X] = i)
        inv[p_arr] = np.arange(len(p_arr))
        return inv

    # Pre-calculate rank vectors for O(N^2) loop
    rank_vectors = [invert_permutation(r) for r in rankings]

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            elif i < j:
                # Calculate Tau (symmetric)
                tau: float
                tau, _ = kendalltau(rank_vectors[i], rank_vectors[j])  # pyright: ignore[reportAssignmentType]
                matrix[i, j] = tau
                matrix[j, i] = tau

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        xticklabels=names,
        yticklabels=names,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax,
    )

    ax.set_title("Pairwise Kendall's Tau Correlation")
    plt.tight_layout()

    return fig


def rme_heatmap(scores: list[float]): ...


if __name__ == "__main__":
    root = Path("data/Rheumatoid_arthritis")

    giggle_scores = parse_scores_tsv(root / "giggle_scores.tsv", 0, 7, 2)
    seqpare_scores = invert_scores(
        parse_scores_tsv(root / "seqpare_scores.tsv", 5, 4, 1)
    )
    cmodel_scores = invert_scores(parse_scores_tsv(root / "cmodel_scores.tsv", 0, 1))
    fm_scores = invert_scores(parse_scores_tsv(root / "cmodel_scores.tsv", 0, 2))

    giggle_ranks = scores_to_ranks(giggle_scores)
    seqpare_ranks = scores_to_ranks(seqpare_scores)
    cmodel_ranks = scores_to_ranks(cmodel_scores)
    fm_ranks = scores_to_ranks(fm_scores)

    names = ["CModel", "HyenaDNA", "Seqpare", "Giggle"]
    ranks = [cmodel_ranks, fm_ranks, seqpare_ranks, giggle_ranks]

    fig = kendall_tau_matrix(zip(names, ranks))
    fig.savefig(Path("experiments/rheumatoid_arthritis/rme_kt_matrix.png"))
