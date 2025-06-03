import contextlib
import os
import pickle
from collections import defaultdict
from collections.abc import Sequence
from math import log
from pathlib import Path

import numpy as np
import scipy.stats

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
        print(f"{i+1} / {len(embeds)}")

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


chromatinStates = [
    "Active_TSS",
    "Flanking_Active_TSS",
    "Strong_transcription",
    "Weak_transcription",
    "Enhancers",
    "Genic_enhancers",
    "ZNF_genes_and_repeats",
    "Heterochromatin",
    "Bivalent_Poised_TSS",
    "Flanking_Bivalent_TSS_Enh",
    "Bivalent_Enhancer",
    "Repressed_PolyComb",
    "Weak_Repressed_PolyComb",
    "Transcr_at_gene_5_and_3",
    "Quiescent_Low",
]

cellTypes = [
    "Adult_Liver",
    "H1_BMP4_Derived_Trophoblast_Cultured_Cells",
    "NHEK_Epidermal_Keratinocytes",
    "HSMMtube_Skeletal_Muscle_Myotubes_Derived_from_HSMM",
    "Monocytes_CD14pp_RO01746",
    "NHDF_Ad_Adult_Dermal_Fibroblasts",
    "Adipose_Nuclei",
    "Fetal_Kidney",
    "Fetal_Intestine_Small",
    "HMEC_Mammary_Epithelial",
    "Stomach_Mucosa",
    "hESC_Derived_CD184pp_Endoderm_Cultured_Cells",
    "Fetal_Brain_Male",
    "H1_Derived_Neuronal_Progenitor_Cultured_Cells",
    "Penis_Foreskin_Fibroblast_Primary_Cells_skin02",
    "Colonic_Mucosa",
    "Breast_vHMEC",
    "Skeletal_Muscle_Female",
    "HUES6_Cell_Line",
    "Chondrocytes_from_Bone_Marrow_Derived_Mesenchymal_Stem_Cell_Cultured_Cells",
    "Muscle_Satellite_Cultured_Cells",
    "Brain_Mid_Frontal_Lobe",
    "ES_I3_Cell_Line",
    "iPS_18_Cell_Line",
    "Penis_Foreskin_Keratinocyte_Primary_Cells_skin03",
    "NH_A_Astrocytes",
    "Pancreas",
    "HUES64_Cell_Line",
    "H1_Derived_Mesenchymal_Stem_Cells",
    "Breast_Myoepithelial_Cells",
    "CD4pp_CD25__IL17__PMA_Ionomycin_stimulated_MACS_purified_Th_Primary_Cells",
    "hESC_Derived_CD56pp_Ectoderm_Cultured_Cells",
    "Esophagus",
    "iPS_20b_Cell_Line",
    "Duodenum_Mucosa",
    "Rectal_Mucosa.Donor_31",
    "H9_Derived_Neuronal_Progenitor_Cultured_Cells",
    "Ovary",
    "IMR90_Cell_Line",
    "Dnd41_TCell_Leukemia",
    "Aorta",
    "Osteoblasts",
    "iPS_15b_Cell_Line",
    "CD34_Cultured_Cells",
    "iPS_DF_19.11_Cell_Line",
    "CD4pp_CD25pp_CD127__Treg_Primary_Cells",
    "Fetal_Adrenal_Gland",
    "HUVEC_Umbilical_Vein_Endothelial_Cells",
    "Fetal_Muscle_Trunk",
    "Colon_Smooth_Muscle",
    "CD4_Naive_Primary_Cells",
    "HUES48_Cell_Line",
    "Spleen",
    "HepG2_Hepatocellular_Carcinoma",
    "CD56_Primary_Cells",
    "CD3_Primary_Cells_Peripheral_UW",
    "CD4pp_CD25__CD45RApp_Naive_Primary_Cells",
    "CD34_Primary_Cells",
    "Sigmoid_Colon",
    "HeLa_S3_Cervical_Carcinoma",
    "Neurosphere_Cultured_Cells_Ganglionic_Eminence_Derived",
    "Mesenchymal_Stem_Cell_Derived_Adipocyte_Cultured_Cells",
    "Fetal_Intestine_Large",
    "Brain_Cingulate_Gyrus",
    "Penis_Foreskin_Fibroblast_Primary_Cells_skin01",
    "CD3_Primary_Cells_Cord_BI",
    "Penis_Foreskin_Melanocyte_Primary_Cells_skin01",
    "Brain_Anterior_Caudate",
    "Skeletal_Muscle_Male",
    "Fetal_Stomach",
    "CD4pp_CD25int_CD127pp_Tmem_Primary_Cells",
    "NHLF_Lung_Fibroblasts",
    "CD14_Primary_Cells",
    "Thymus",
    "Neurosphere_Cultured_Cells_Cortex_Derived",
    "Right_Ventricle",
    "Pancreatic_Islets",
    "Gastric",
    "GM12878_Lymphoblastoid",
    "Mobilized_CD34_Primary_Cells_Female",
    "CD4pp_CD25__IL17pp_PMA_Ionomcyin_stimulated_Th17_Primary_Cells",
    "Penis_Foreskin_Melanocyte_Primary_Cells_skin03",
    "Fetal_Placenta",
    "Fetal_Thymus",
    "H1_Cell_Line",
    "CD15_Primary_Cells",
    "Right_Atrium",
    "CD8_Memory_Primary_Cells",
    "hESC_Derived_CD56pp_Mesoderm_Cultured_Cells",
    "Fetal_Lung",
    "Bone_Marrow_Derived_Mesenchymal_Stem_Cell_Cultured_Cells",
    "Brain_Germinal_Matrix",
    "Small_Intestine",
    "H9_Cell_Line",
    "4star",
    "Brain_Substantia_Nigra",
    "iPS_DF_6.9_Cell_Line",
    "Brain_Inferior_Temporal_Lobe",
    "CD4pp_CD25__CD45ROpp_Memory_Primary_Cells",
    "CD4_Memory_Primary_Cells",
    "CD19_Primary_Cells_Peripheral_UW",
    "Peripheral_Blood_Mononuclear_Primary_Cells",
    "Rectal_Smooth_Muscle",
    "Left_Ventricle",
    "A549_EtOH_0.02pct_Lung_Carcinoma",
    "ES_WA7_Cell_Line",
    "Placenta_Amnion",
    "Adipose_Derived_Mesenchymal_Stem_Cell_Cultured_Cells",
    "Rectal_Mucosa.Donor_29",
    "Brain_Hippocampus_Middle",
    "Stomach_Smooth_Muscle",
    "CD4pp_CD25__Th_Primary_Cells",
    "Mobilized_CD34_Primary_Cells_Male",
    "CD8_Naive_Primary_Cells",
    "K562_Leukemia",
    "Fetal_Heart",
    "H9_Derived_Neuron_Cultured_Cells",
    "CD19_Primary_Cells_Cord_BI",
    "Fetal_Muscle_Leg",
    "Brain_Angular_Gyrus",
    "Fetal_Brain_Female",
    "Lung",
    "Duodenum_Smooth_Muscle",
    "HSMM_Skeletal_Muscle_Myoblasts",
    "Penis_Foreskin_Keratinocyte_Primary_Cells_skin02",
    "Psoas_Muscle",
    "H1_BMP4_Derived_Mesendoderm_Cultured_Cells",
]

cellCategory = {
    "IPSC": [
        "iPS_18_Cell_Line",
        "iPS_20b_Cell_Line",
        "iPS_15b_Cell_Line",
        "iPS_DF_19.11_Cell_Line",
        "iPS_DF_6.9_Cell_Line",
    ],
    "ESC": [
        "HUES6_Cell_Line",
        "ES_I3_Cell_Line",
        "HUES64_Cell_Line",
        "HUES48_Cell_Line",
        "H1_Cell_Line",
        "H9_Cell_Line",
        "4star",
        "ES_WA7_Cell_Line",
    ],
    "ES_derived": [
        "hESC_Derived_CD184pp_Endoderm_Cultured_Cells",
        "H1_Derived_Neuronal_Progenitor_Cultured_Cells",
        "H1_Derived_Mesenchymal_Stem_Cells",
        "hESC_Derived_CD56pp_Ectoderm_Cultured_Cells",
        "H9_Derived_Neuronal_Progenitor_Cultured_Cells",
        "hESC_Derived_CD56pp_Mesoderm_Cultured_Cells",
        "H9_Derived_Neuron_Cultured_Cells",
        "H1_BMP4_Derived_Mesendoderm_Cultured_Cells",
    ],
    "HSC_B_cell": [
        "CD34_Cultured_Cells",
        "CD34_Primary_Cells",
        "GM12878_Lymphoblastoid",
        "Mobilized_CD34_Primary_Cells_Female",
        "CD19_Primary_Cells_Peripheral_UW",
        "Mobilized_CD34_Primary_Cells_Male",
        "CD19_Primary_Cells_Cord_BI",
    ],
    "Blood_T_cell_Myeloid": [
        "Monocytes_CD14pp_RO01746",
        "CD4pp_CD25__IL17__PMA_Ionomycin_stimulated_MACS_purified_Th_Primary_Cells",
        "CD4pp_CD25pp_CD127__Treg_Primary_Cells",
        "CD4_Naive_Primary_Cells",
        "Spleen",
        "CD56_Primary_Cells",
        "CD3_Primary_Cells_Peripheral_UW",
        "CD4pp_CD25__CD45RApp_Naive_Primary_Cells",
        "CD3_Primary_Cells_Cord_BI",
        "CD4pp_CD25int_CD127pp_Tmem_Primary_Cells",
        "CD14_Primary_Cells",
        "CD4pp_CD25__IL17pp_PMA_Ionomcyin_stimulated_Th17_Primary_Cells",
        "CD15_Primary_Cells",
        "CD8_Memory_Primary_Cells",
        "CD4pp_CD25__CD45ROpp_Memory_Primary_Cells",
        "CD4_Memory_Primary_Cells",
        "Peripheral_Blood_Mononuclear_Primary_Cells",
        "CD4pp_CD25__Th_Primary_Cells",
        "CD8_Naive_Primary_Cells",
    ],
    "Cancer_Cell_Line": [
        "Dnd41_TCell_Leukemia",
        "HepG2_Hepatocellular_Carcinoma",
        "HeLa_S3_Cervical_Carcinoma",
        "A549_EtOH_0.02pct_Lung_Carcinoma",
        "K562_Leukemia",
    ],
    "Brain": [
        "Fetal_Brain_Male",
        "Brain_Mid_Frontal_Lobe",
        "NH_A_Astrocytes",
        "Brain_Cingulate_Gyrus",
        "Brain_Anterior_Caudate",
        "Brain_Germinal_Matrix",
        "Brain_Substantia_Nigra",
        "Brain_Inferior_Temporal_Lobe",
        "Brain_Hippocampus_Middle",
        "Brain_Angular_Gyrus",
        "Fetal_Brain_Female",
    ],
    "Neurosphere": [
        "Neurosphere_Cultured_Cells_Ganglionic_Eminence_Derived",
        "Neurosphere_Cultured_Cells_Cortex_Derived",
    ],
    "Digestive": [
        "Adult_Liver",
        "Fetal_Intestine_Small",
        "Stomach_Mucosa",
        "Colonic_Mucosa",
        "Pancreas",
        "Esophagus",
        "Duodenum_Mucosa",
        "Rectal_Mucosa.Donor_31",
        "Sigmoid_Colon",
        "Fetal_Intestine_Large",
        "Fetal_Stomach",
        "Gastric",
        "Small_Intestine",
        "Rectal_Mucosa.Donor_29",
    ],
    "Epithelial": ["HMEC_Mammary_Epithelial", "Breast_vHMEC", "Breast_Myoepithelial_Cells"],
    "Muscle_Skeletal": [
        "HSMMtube_Skeletal_Muscle_Myotubes_Derived_from_HSMM",
        "Skeletal_Muscle_Female",
        "Muscle_Satellite_Cultured_Cells",
        "Fetal_Muscle_Trunk",
        "Skeletal_Muscle_Male",
        "Fetal_Muscle_Leg",
        "HSMM_Skeletal_Muscle_Myoblasts",
        "Psoas_Muscle",
    ],
    "Muscle_Smooth": [
        "Colon_Smooth_Muscle",
        "Rectal_Smooth_Muscle",
        "Stomach_Smooth_Muscle",
        "Duodenum_Smooth_Muscle",
    ],
    "Mesenchymal": [
        "Adipose_Nuclei",
        "Chondrocytes_from_Bone_Marrow_Derived_Mesenchymal_Stem_Cell_Cultured_Cells",
        "Osteoblasts",
        "Mesenchymal_Stem_Cell_Derived_Adipocyte_Cultured_Cells",
        "Bone_Marrow_Derived_Mesenchymal_Stem_Cell_Cultured_Cells",
        "Adipose_Derived_Mesenchymal_Stem_Cell_Cultured_Cells",
    ],
    "Lung": ["IMR90_Cell_Line", "NHLF_Lung_Fibroblasts", "Fetal_Lung", "Lung"],
    "Heart_Vasculature": [
        "Aorta",
        "HUVEC_Umbilical_Vein_Endothelial_Cells",
        "Right_Ventricle",
        "Right_Atrium",
        "Left_Ventricle",
        "Fetal_Heart",
    ],
    "Thymus": ["Thymus", "Fetal_Thymus"],
    "Skin_Cells": [
        "NHEK_Epidermal_Keratinocytes",
        "NHDF_Ad_Adult_Dermal_Fibroblasts",
        "Penis_Foreskin_Fibroblast_Primary_Cells_skin02",
        "Penis_Foreskin_Keratinocyte_Primary_Cells_skin03",
        "Penis_Foreskin_Fibroblast_Primary_Cells_skin01",
        "Penis_Foreskin_Melanocyte_Primary_Cells_skin01",
        "Penis_Foreskin_Melanocyte_Primary_Cells_skin03",
        "Penis_Foreskin_Keratinocyte_Primary_Cells_skin02",
    ],
    "Kidney": ["Fetal_Kidney"],
    "Reproductive": ["Ovary"],
    "Endocrine": ["Fetal_Adrenal_Gland", "Pancreatic_Islets"],
    "Placenta": [
        "H1_BMP4_Derived_Trophoblast_Cultured_Cells",
        "Fetal_Placenta",
        "Placenta_Amnion",
    ],
}

cellCategoryInverse = {
    value: category for (category, values) in cellCategory.items() for value in values
}


def cellTypeChrmStateSplit(name: str) -> tuple[str, str]:
    name = name.strip()
    states = list()

    for chrmState in chromatinStates:
        if name.endswith(chrmState):
            states.append(chrmState)

    if len(states) == 0:
        raise ValueError(f"{name} didn't match a known chromatin state")

    chrmState = sorted(states, key=len)[-1]
    cellType = name[: -len(chrmState) - 1]
    return cellType, chrmState


def keywordScore(base: str, others: Sequence[str]):
    """
    A normalized score based on the indices of elements in the list with matching keywords
    to the base. Uses nDCG.
    """

    k = 1 * len(others)  # DCG is often calculated with only the upper subset
    targetCT, targetCS = cellTypeChrmStateSplit(base)
    dcg = 0

    for i, other in enumerate(others[:k]):
        ct, cs = cellTypeChrmStateSplit(other)
        gain = 0

        if ct == targetCT:
            gain += 1
        if cs == targetCS:
            gain += 1

        dcg += gain / log(i + 2, 2)

    # ideal DCG

    idcg = 2  # ideally, the first item contains both CT & CS

    for i in range(len(chromatinStates) + len(cellTypes) - 1):
        idcg += 1 / log(i + 3, 2)

    return dcg / idcg


def kendallSig(reference: Sequence[str], observed: Sequence[str], resamples: int = 9999) -> float:
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
        return scipy.stats.kendalltau(permuted_subject_ranks, fixed_ideal_ranks).statistic

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
    legacyRanks = legacyRank([f"{data}/giggleRanks/{name}.tsv" for name in modernRanks.keys()])

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
            modern = cellTypeChrmStateSplit(modernValue[0])
            legacy = cellTypeChrmStateSplit(legacyValue[0])
            k = 0  # this is like a temperature
            # (mean) reciprocal rank
            keywordSums1[modern[0]] += 1 / (j + 1 + k)
            keywordSums1[modern[1]] += 1 / (j + 1 + k)
            keywordSums2[legacy[0]] += 1 / (j + 1 + k)
            keywordSums2[legacy[1]] += 1 / (j + 1 + k)

        with printTo(f"{exp}/keywordRanks/{name}-cellType.tsv"):
            cellTypeRanks1 = sorted(cellTypes, key=lambda x: -keywordSums1[x])
            cellTypeRanks2 = sorted(cellTypes, key=lambda x: -keywordSums2[x])

            print(
                "(New) Cell Type",
                "(Old) Cell Type",
                sep="\t",
            )

            for x in zip(cellTypeRanks1, cellTypeRanks2):
                print(*x, sep="\t")

        with printTo(f"{exp}/keywordRanks/{name}-chrmState.tsv"):
            chrmStateRanks1 = sorted(chromatinStates, key=lambda x: -keywordSums1[x])
            chrmStateRanks2 = sorted(chromatinStates, key=lambda x: -keywordSums2[x])

            print(
                "(New) Chromatin State",
                "(Old) Chromatin State",
                sep="\t",
            )

            for x in zip(chrmStateRanks1, chrmStateRanks2):
                print(*x, sep="\t")

        print(f"{i+1} / {len(modernRanks)}")

    # INFO: heatmap
    print("Heatmap...")

    target = "Brain_Hippocampus_Middle_Enhancers"
    modernHits = {hit: value for (hit, value) in modernRanks[target]}
    legacyHits = {hit: value for (hit, value) in legacyRanks[target]}

    def prepareHeatmapMatrix(sigMap):
        matrix = list()
        yLabels = list()

        for category in cellCategory:
            cellTypes = cellCategory[category]
            yLabels.append((category, len(cellTypes)))

            for cellType in cellTypes:
                row = list()

                for chrmState in chromatinStates:
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
        chromatinStates, yLabels, matrix, title=target, desired_tile_in=(0.3, 0.1)
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
            print(f"{i+1} / {len(modernRanks)}")

        modernKeyScoreAvg = np.mean(list(modernKeyScores.values()))
        legacyKeyScoreAvg = np.mean(list(legacyKeyScores.values()))
        print("avg", modernKeyScoreAvg, legacyKeyScoreAvg, sep="\t", file=f)

    # INFO: kendall's tau, pvalue
    print(f"Kendall's Tau, P-Value (using legacy-giggle as the reference rankings)")
    legacyNeighbors = [name for (name, _) in legacyRanks[target]]
    modernNeighbors = [name for (name, _) in modernRanks[target]]
    # TODO: the kendall tau sig is too good; investigate
    ktPval = kendallSig(legacyNeighbors, modernNeighbors, 99999)
    print(f" - for {target}: {ktPval}")


if __name__ == "__main__":
    main()
