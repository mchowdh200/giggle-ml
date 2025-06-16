import os
import pickle
from collections import defaultdict
from collections.abc import Sequence
from math import log
from pathlib import Path

import numpy as np
import scipy.stats

import giggleml.utils.roadmapEpigenomics as rme
from analysis.advHeatmap import plot_heatmap_with_averages
from giggleml.embedGen import meanEmbedDict
from giggleml.utils.printTo import printTo


def modernRank(masterPath: str, outPath: str) -> dict[str, list[tuple[str, float]]]:
    if os.path.isfile(outPath):
        return pickle.load(open(outPath, "rb"))

    embeds = meanEmbedDict.parse(masterPath)
    top = dict()

    for i, (base, baseEm) in enumerate(embeds.items()):

        def emDist(em2: np.ndarray):
            return np.linalg.norm(em2 - baseEm).item()

        neighbors = [(name, emDist(em)) for (name, em) in embeds.items()]
        top[base] = sorted(neighbors, key=lambda x: x[1])
        print(f"{i + 1} / {len(embeds)}")

    pickle.dump(top, open(outPath, "wb"))
    return top


def _cleanName(name: str):
    name = os.path.basename(name)

    if name.endswith(".tsv"):
        name = name[:-4]
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".bed"):
        name = name[:-4]

    return name


def legacyRank(paths: Sequence[str]) -> dict[str, list[tuple[str, float]]]:
    top = dict()

    for path in paths:
        scores = list[tuple[str, float]]()

        with open(path, "r") as file:
            lines = iter(file.readlines())

            line0 = next(lines).strip()
            keyCol = line0.split("\t").index("combo_score")

            for line in lines:
                if line.startswith("#"):
                    continue

                split = line.split("\t")
                value = float(split[keyCol])
                name = _cleanName(split[0])
                scores.append((name, value))

        scores = sorted(scores, key=lambda x: -x[1])
        baseName = _cleanName(path)
        top[baseName] = scores

    return top


def keywordScore(base: str, others: Sequence[str]):
    """
    A normalized score based on the indices of elements in the list with matching keywords
    to the base. Uses nDCG.
    """

    k = 1 * len(others)  # DCG is often calculated with only the upper subset
    targetCT, targetCS = rme.cellTypeChrmStateSplit(base)
    dcg = 0

    for i, other in enumerate(others[:k]):
        ct, cs = rme.cellTypeChrmStateSplit(other)
        gain = 0

        if ct == targetCT:
            gain += 1
        if cs == targetCS:
            gain += 1

        dcg += gain / log(i + 2, 2)

    # ideal DCG

    idcg = 2  # ideally, the first item contains both CT & CS

    for i in range(len(rme.chromatinStates) + len(rme.cellTypes) - 1):
        idcg += 1 / log(i + 3, 2)

    return dcg / idcg


def kendallSig(
    reference: Sequence[str], observed: Sequence[str], resamples: int = 9999
) -> float:
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

    commonSize = len(ref)
    if commonSize < 2:
        message = (
            f"Warning: Only {commonSize} common item(s). "
            "Kendall's Tau requires at least 2 for a meaningful comparison."
        )
        print(message)
        # Calculate Tau for completeness if possible, though p-value might not be robust
        if commonSize == 1:  # Only one common item
            # Tau is undefined or trivially 0, p-value is meaningless.
            # Or, based on some conventions, tau might be 1 if we only have one element.
            # SciPy's kendalltau will likely result in nan or error for < 2 elements.
            # Let's return NaN for tau here for clarity.
            return np.nan
        # if num_common_items is 0, it's caught by the `if not common_items` above.

    # 2. Map to Ranks
    refRank = {item: i for i, item in enumerate(ref)}
    # Observed ranks for the second list, based on the ordering defined by the first list
    obsRank = np.array([refRank[item] for item in obs])

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
            (obsRank, np.array(list(range(commonSize)))),  # Data tuple
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

    modernRanks = modernRank(f"{data}/embeds/master.pickle", f"{data}/ranks.pickle")
    legacyRanks = legacyRank(
        [f"{data}/giggleRanks/{name}.tsv" for name in modernRanks.keys()]
    )

    # INFO: rank comparison .tsv files
    print("Top keyword & file ranks...")

    Path(f"{exp}/keywordRanks").mkdir(exist_ok=True)
    Path(f"{exp}/fileRanks").mkdir(exist_ok=True)

    for i, name in enumerate(modernRanks.keys()):
        # raw rank data

        with printTo(f"{exp}/fileRanks/{name}.tsv"):
            print("Embedding-based", "Giggle-legacy", sep="\t")

            for modernValue, legacyValue in zip(modernRanks[name], legacyRanks[name]):
                print(modernValue[0], legacyValue[0], sep="\t")

        # top keyword ranks

        keywordSums1 = defaultdict(lambda: 0.0)
        keywordSums2 = defaultdict(lambda: 0.0)

        for j, (modernValue, legacyValue) in list(
            enumerate(zip(modernRanks[name], legacyRanks[name]))
        ):
            modern = rme.cellTypeChrmStateSplit(modernValue[0])
            legacy = rme.cellTypeChrmStateSplit(legacyValue[0])
            k = 0  # this is like a temperature
            # (mean) reciprocal rank
            keywordSums1[modern[0]] += 1 / (j + 1 + k)
            keywordSums1[modern[1]] += 1 / (j + 1 + k)
            keywordSums2[legacy[0]] += 1 / (j + 1 + k)
            keywordSums2[legacy[1]] += 1 / (j + 1 + k)

        with printTo(f"{exp}/keywordRanks/{name}-cellType.tsv"):
            cellTypeRanks1 = sorted(rme.cellTypes, key=lambda x: -keywordSums1[x])
            cellTypeRanks2 = sorted(rme.cellTypes, key=lambda x: -keywordSums2[x])

            print(
                "(New) Cell Type",
                "(Old) Cell Type",
                sep="\t",
            )

            for x in zip(cellTypeRanks1, cellTypeRanks2):
                print(*x, sep="\t")

        with printTo(f"{exp}/keywordRanks/{name}-chrmState.tsv"):
            chrmStateRanks1 = sorted(
                rme.chromatinStates, key=lambda x: -keywordSums1[x]
            )
            chrmStateRanks2 = sorted(
                rme.chromatinStates, key=lambda x: -keywordSums2[x]
            )

            print(
                "(New) Chromatin State",
                "(Old) Chromatin State",
                sep="\t",
            )

            for x in zip(chrmStateRanks1, chrmStateRanks2):
                print(*x, sep="\t")

        print(f"{i + 1} / {len(modernRanks)}")

    # INFO: heatmap
    print("Heatmap...")

    target = "Brain_Hippocampus_Middle_Enhancers"
    modernHits = {hit: value for (hit, value) in modernRanks[target]}
    legacyHits = {hit: value for (hit, value) in legacyRanks[target]}

    def prepareHeatmapMatrix(sigMap):
        matrix = list()
        yLabels = list()

        for category in rme.cellCategory:
            cellTypes = rme.cellCategory[category]
            yLabels.append((category, len(cellTypes)))

            for cellType in cellTypes:
                row = list()

                for chrmState in rme.chromatinStates:
                    name = f"{cellType}_{chrmState}"
                    value = modernHits.get(name, None)
                    row.append(value)

                matrix.append(row)

        # Apply max - x to valid elements, preserving None
        matrix = np.array(matrix)
        mask = np.ma.masked_invalid(matrix.astype(float))
        matrix = np.where(matrix != None, mask.max() - matrix.astype(float), None)
        return matrix, yLabels

    matrix, yLabels = prepareHeatmapMatrix(modernHits)
    path = f"{exp}/heatmaps/{target}.png"
    fig, _ = plot_heatmap_with_averages(
        rme.chromatinStates, yLabels, matrix, title=target, desired_tile_in=(0.3, 0.1)
    )
    fig.savefig(path, dpi=300, bbox_inches="tight")

    # INFO: general scores
    print("General scores...")

    modernKeyScores = dict[str, float]()
    legacyKeyScores = dict[str, float]()

    with open(f"{exp}/rankAnalysis.tsv", "w") as f:
        print("name", "(new) nDCG", "(old) nDCG", sep="\t", file=f)

        for i, name in enumerate(modernRanks.keys()):
            # keyword score

            modernNeighbors = [other for (other, _) in modernRanks[name]]
            modernKeyScore = keywordScore(name, modernNeighbors)
            modernKeyScores[name] = modernKeyScore

            legacyNeighbors = [other for (other, _) in legacyRanks[name]]
            legacyKeyScore = keywordScore(name, legacyNeighbors)
            legacyKeyScores[name] = legacyKeyScore

            print(name, modernKeyScore, legacyKeyScore, sep="\t", file=f)
            print(f"{i + 1} / {len(modernRanks)}")

        modernKeyScoreAvg = np.mean(list(modernKeyScores.values()))
        legacyKeyScoreAvg = np.mean(list(legacyKeyScores.values()))
        print("avg", modernKeyScoreAvg, legacyKeyScoreAvg, sep="\t", file=f)

    # INFO: kendall's tau, pvalue
    print("Kendall's Tau, P-Value (using legacy-giggle as the reference rankings)")
    legacyNeighbors = [name for (name, _) in legacyRanks[target]]
    modernNeighbors = [name for (name, _) in modernRanks[target]]
    # TODO: the kendall tau sig is too good; investigate
    ktPval = kendallSig(legacyNeighbors, modernNeighbors, 99999)
    print(f" - for {target}: {ktPval}")


if __name__ == "__main__":
    main()
