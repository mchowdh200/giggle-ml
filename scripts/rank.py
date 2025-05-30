import os
import pickle
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

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


def keywordScore(base: str, others: Sequence[str], keywords: set[str]):
    """
    A normalized score based on the indices of elements in the list with matching keywords
    to the base. That's, lists with lots of elements with matching keywords to base have a
    higher score. The amount of keywords is not considered; binary.
    """

    selectKeywords = list()

    for kw in keywords:
        if kw in base:
            selectKeywords.append(kw)

    weight = 0
    total = 0

    for i, other in enumerate(others):
        for kw in selectKeywords:
            if kw in other:
                weight += i + 1
                total += 1
                break

    if total == 0:
        return 0
    if total == len(others):
        return 1

    gauss = lambda x: x * (x + 1) / 2
    small = gauss(total)
    big = gauss(len(others)) - gauss(len(others) - total)
    return 1 - (weight - small) / (big - small)


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


def chrmStateCellTypeSplit(name: str) -> tuple[str, str]:
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


def main():
    data = "data/roadmap_epigenomics"
    exp = "experiments/roadmapEpigenomics"

    modern = modernRank(f"{data}/embeds/master.pickle", f"{data}/ranks.pickle")
    legacy = legacyRank([f"{data}/giggleRanks/{name}.tsv" for name in modern.keys()])
    keywords = set(cellTypes).union(chromatinStates)

    # INFO: ranks comparison .tsv file

    for i, name in enumerate(modern.keys()):
        with printTo(f"{data}/compare/{name}.tsv"):
            print("Embedding-based", "\t", "Giggle-legacy")

            for modernValue, legacyValue in zip(modern[name], legacy[name]):
                print(modernValue[0], "\t", legacyValue[0])

        print(f"{i+1} / {len(modern)}")

    # INFO: heatmap

    target = "Penis_Foreskin_Melanocyte_Primary_Cells_skin01_Genic_enhancers"
    modernHits = {hit: value for (hit, value) in modern[target]}
    legacyHits = {hit: value for (hit, value) in legacy[target]}

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

    # INFO: keyword rankings

    modernKeyScores = dict[str, float]()
    legacyKeyScores = dict[str, float]()

    with printTo(f"{exp}/rankAnalysis.txt"):
        print("name\tEmbedding-based\tGiggle-legacy")

        for i, name in enumerate(modern.keys()):
            modernNeighbors = [other for (other, _) in modern[name]]
            modernKeyScore = keywordScore(name, modernNeighbors, keywords)
            modernKeyScores[name] = modernKeyScore

            legacyNeighbors = [other for (other, _) in legacy[name]]
            legacyKeyScore = keywordScore(name, legacyNeighbors, keywords)
            legacyKeyScores[name] = legacyKeyScore

            print(name, "\t", modernKeyScore, "\t", legacyKeyScore)

        modernKeyScoreAvg = np.mean(list(modernKeyScores.values()))
        legacyKeyScoreAvg = np.mean(list(legacyKeyScores.values()))
        print("avg", "\t", modernKeyScoreAvg, "\t", legacyKeyScoreAvg)


if __name__ == "__main__":
    main()
