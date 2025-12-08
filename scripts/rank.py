import os
import pickle
from collections import defaultdict
from collections.abc import Sequence
from math import log
from pathlib import Path

import numpy as np
import scipy.stats

import giggleml.utils.roadmap_epigenomics as rme
from analysis.adv_heatmap import plot_heatmap_with_averages
from giggleml.embed_gen import mean_embed_dict
from giggleml.utils.print_to import print_to


def modern_rank(master_path: str, out_path: str) -> dict[str, list[tuple[str, float]]]:
    if os.path.isfile(out_path):
        return pickle.load(open(out_path, "rb"))

    embeds = mean_embed_dict.parse(master_path)
    top = dict()

    for i, (base, base_em) in enumerate(embeds.items()):

        def em_dist(em2: np.ndarray):
            return np.linalg.norm(em2 - base_em).item()

        neighbors = [(name, em_dist(em)) for (name, em) in embeds.items()]
        top[base] = sorted(neighbors, key=lambda x: x[1])
        print(f"{i + 1} / {len(embeds)}")

    pickle.dump(top, open(out_path, "wb"))
    return top


def _clean_name(name: str):
    name = os.path.basename(name)

    if name.endswith(".tsv"):
        name = name[:-4]
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".bed"):
        name = name[:-4]

    return name


def legacy_rank(paths: Sequence[str]) -> dict[str, list[tuple[str, float]]]:
    top = dict()

    for path in paths:
        scores = list[tuple[str, float]]()

        with open(path, "r") as file:
            lines = iter(file.readlines())

            line0 = next(lines).strip()
            key_col = line0.split("\t").index("combo_score")

            for line in lines:
                if line.startswith("#"):
                    continue

                split = line.split("\t")
                value = float(split[key_col])
                name = _clean_name(split[0])
                scores.append((name, value))

        scores = sorted(scores, key=lambda x: -x[1])
        base_name = _clean_name(path)
        top[base_name] = scores

    return top


def keyword_score(base: str, others: Sequence[str]):
    """
    A normalized score based on the indices of elements in the list with matching keywords
    to the base. Uses nDCG.
    """

    k = 1 * len(others)  # DCG is often calculated with only the upper subset
    target_ct, target_cs = rme.cell_type_chrm_state_split(base)
    dcg = 0

    for i, other in enumerate(others[:k]):
        ct, cs = rme.cell_type_chrm_state_split(other)
        gain = 0

        if ct == target_ct:
            gain += 1
        if cs == target_cs:
            gain += 1

        dcg += gain / log(i + 2, 2)

    # ideal DCG

    idcg = 2  # ideally, the first item contains both CT & CS

    for i in range(len(rme.chromatin_states) + len(rme.cell_types) - 1):
        idcg += 1 / log(i + 3, 2)

    return dcg / idcg


def kendall_sig(
    reference: Sequence[str], observed: Sequence[str], resamples: int = 9999
) -> float:
    # WARN: this will almost always be zero if the observed tau is slightly greater
    # than zero. This is due to the large amount (2k) ranks we're comparing.
    # At sufficiently high N and resample count, the null-hypothesis tau values will
    # cluster very tightly around zero. Anything higher is interpreted as extremely significant.

    """
    Performs a permutation test for Kendall's Tau between two string lists.
    `reference` is considered the ideal order.
    """

    # 1. Preprocessing: Find common items and maintain their relative orders
    common = set(reference) & set(observed)

    if not common:
        print("Warning: No common items between the lists.")
        return np.nan

    ref = [item for item in reference if item in common]
    obs = [item for item in observed if item in common]

    common_size = len(ref)
    if common_size < 2:
        message = (
            f"Warning: Only {common_size} common item(s). "
            "Kendall's Tau requires at least 2 for a meaningful comparison."
        )
        print(message)
        # Calculate Tau for completeness if possible, though p-value might not be robust
        if common_size == 1:  # Only one common item
            # Tau is undefined or trivially 0, p-value is meaningless.
            # Or, based on some conventions, tau might be 1 if we only have one element.
            # SciPy's kendalltau will likely result in nan or error for < 2 elements.
            # Let's return NaN for tau here for clarity.
            return np.nan
        # if num_common_items is 0, it's caught by the `if not common_items` above.

    # 2. Map to Ranks
    ref_rank = {item: i for i, item in enumerate(ref)}
    # Observed ranks for the second list, based on the ordering defined by the first list
    obs_rank = np.array([ref_rank[item] for item in obs])

    # 3. Define Statistic Function for permutation_test
    def statistic_fn(permuted_subject_ranks, fixed_ideal_ranks):
        # permuted_subject_ranks is a permutation of observed_subject_ranks
        # fixed_ideal_ranks is ideal_ranks
        return scipy.stats.kendalltau(
            permuted_subject_ranks, fixed_ideal_ranks
        ).statistic

    # 4. Perform Permutation Test
    # scipy.stats.permutation_test will permute the first element of the `data`
    # tuple (observed_subject_ranks) and keep the second (ideal_ranks) fixed
    # when permutation_type='pairings'.
    try:
        res = scipy.stats.permutation_test(
            (obs_rank, np.array(list(range(common_size)))),  # Data tuple
            statistic_fn,
            permutation_type="pairings",
            n_resamples=resamples,
            alternative="greater",
            rng=42,
        )
        return res.pvalue
    except ValueError as e:
        # This can happen if, for example, all values in one array are identical after filtering,
        # which can lead to issues with Kendall's Tau calculation (e.g. tau is nan).
        print(
            f"Warning: Error during permutation test (possibly due to identical ranks or too few distinct values): {e}"
        )
        return np.nan


def main():
    data = "data/roadmap_epigenomics"
    exp = "experiments/roadmapEpigenomics"

    modern_ranks = modern_rank(f"{data}/embeds/master.pickle", f"{data}/ranks.pickle")
    legacy_ranks = legacy_rank(
        [f"{data}/giggleRanks/{name}.tsv" for name in modern_ranks.keys()]
    )

    # INFO: rank comparison .tsv files
    print("Top keyword & file ranks...")

    Path(f"{exp}/keywordRanks").mkdir(exist_ok=True)
    Path(f"{exp}/fileRanks").mkdir(exist_ok=True)

    for i, name in enumerate(modern_ranks.keys()):
        # raw rank data

        with print_to(f"{exp}/fileRanks/{name}.tsv"):
            print("Embedding-based", "Giggle-legacy", sep="\t")

            for modern_value, legacy_value in zip(
                modern_ranks[name], legacy_ranks[name]
            ):
                print(modern_value[0], legacy_value[0], sep="\t")

        # top keyword ranks

        keyword_sums1 = defaultdict(lambda: 0.0)
        keyword_sums2 = defaultdict(lambda: 0.0)

        for j, (modern_value, legacy_value) in list(
            enumerate(zip(modern_ranks[name], legacy_ranks[name]))
        ):
            modern = rme.cell_type_chrm_state_split(modern_value[0])
            legacy = rme.cell_type_chrm_state_split(legacy_value[0])
            k = 0  # this is like a temperature
            # (mean) reciprocal rank
            keyword_sums1[modern[0]] += 1 / (j + 1 + k)
            keyword_sums1[modern[1]] += 1 / (j + 1 + k)
            keyword_sums2[legacy[0]] += 1 / (j + 1 + k)
            keyword_sums2[legacy[1]] += 1 / (j + 1 + k)

        with print_to(f"{exp}/keywordRanks/{name}-cellType.tsv"):
            cell_type_ranks1 = sorted(rme.cell_types, key=lambda x: -keyword_sums1[x])
            cell_type_ranks2 = sorted(rme.cell_types, key=lambda x: -keyword_sums2[x])

            print(
                "(New) Cell Type",
                "(Old) Cell Type",
                sep="\t",
            )

            for x in zip(cell_type_ranks1, cell_type_ranks2):
                print(*x, sep="\t")

        with print_to(f"{exp}/keywordRanks/{name}-chrmState.tsv"):
            chrm_state_ranks1 = sorted(
                rme.chromatin_states, key=lambda x: -keyword_sums1[x]
            )
            chrm_state_ranks2 = sorted(
                rme.chromatin_states, key=lambda x: -keyword_sums2[x]
            )

            print(
                "(New) Chromatin State",
                "(Old) Chromatin State",
                sep="\t",
            )

            for x in zip(chrm_state_ranks1, chrm_state_ranks2):
                print(*x, sep="\t")

        print(f"{i + 1} / {len(modern_ranks)}")

    # INFO: heatmap
    print("Heatmap...")

    target = "Brain_Hippocampus_Middle_Enhancers"
    modern_hits = {hit: value for (hit, value) in modern_ranks[target]}
    legacy_hits = {hit: value for (hit, value) in legacy_ranks[target]}

    def prepare_heatmap_matrix(sig_map):
        matrix = list()
        y_labels = list()

        for category in rme.cell_category:
            cell_types = rme.cell_category[category]
            y_labels.append((category, len(cell_types)))

            for cell_type in cell_types:
                row = list()

                for chrm_state in rme.chromatin_states:
                    name = f"{cell_type}_{chrm_state}"
                    value = modern_hits.get(name, None)
                    row.append(value)

                matrix.append(row)

        # Apply max - x to valid elements, preserving None
        matrix = np.array(matrix)
        mask = np.ma.masked_invalid(matrix.astype(float))
        matrix = np.where(matrix != None, mask.max() - matrix.astype(float), None)
        return matrix, y_labels

    matrix, y_labels = prepare_heatmap_matrix(modern_hits)
    path = f"{exp}/heatmaps/{target}.png"
    fig, _ = plot_heatmap_with_averages(
        rme.chromatin_states, y_labels, matrix, title=target, desired_tile_in=(0.3, 0.1)
    )
    fig.savefig(path, dpi=300, bbox_inches="tight")

    # INFO: general scores
    print("General scores...")

    modern_key_scores = dict[str, float]()
    legacy_key_scores = dict[str, float]()

    with open(f"{exp}/rankAnalysis.tsv", "w") as f:
        print("name", "(new) nDCG", "(old) nDCG", sep="\t", file=f)

        for i, name in enumerate(modern_ranks.keys()):
            # keyword score

            modern_neighbors = [other for (other, _) in modern_ranks[name]]
            modern_key_score = keyword_score(name, modern_neighbors)
            modern_key_scores[name] = modern_key_score

            legacy_neighbors = [other for (other, _) in legacy_ranks[name]]
            legacy_key_score = keyword_score(name, legacy_neighbors)
            legacy_key_scores[name] = legacy_key_score

            print(name, modern_key_score, legacy_key_score, sep="\t", file=f)
            print(f"{i + 1} / {len(modern_ranks)}")

        modern_key_score_avg = np.mean(list(modern_key_scores.values()))
        legacy_key_score_avg = np.mean(list(legacy_key_scores.values()))
        print("avg", modern_key_score_avg, legacy_key_score_avg, sep="\t", file=f)

    # INFO: kendall's tau, pvalue
    print("Kendall's Tau, P-Value (using legacy-giggle as the reference rankings)")
    legacy_neighbors = [name for (name, _) in legacy_ranks[target]]
    modern_neighbors = [name for (name, _) in modern_ranks[target]]
    # TODO: the kendall tau sig is too good; investigate
    kt_pval = kendall_sig(legacy_neighbors, modern_neighbors, 99999)
    print(f" - for {target}: {kt_pval}")


if __name__ == "__main__":
    main()
