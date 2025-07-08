import os
import pickle
import subprocess

import giggleml.utils.roadmapEpigenomics as rme
from giggleml.dataWrangling.intervalDataset import BedDataset
from giggleml.utils.misc import fix_bed_ext
from giggleml.utils.types import GenomicInterval


def seqpare_raw(roadep_path: str, content: list[GenomicInterval]):
    with open("tmp.bed", "w") as f:
        for chr, start, end in content:
            f.write(f"{chr}\t{start}\t{end}\n")

    seqpare = subprocess.run(
        [
            "seqpare",
            roadep_path + "/*",
            "tmp.bed",
            "-m",
            "1",
            "-o",
            "out.tmp.tsv",
        ],  # Command and its arguments as a list
        capture_output=True,
        text=True,  # Decode stdout/stderr as text (UTF-8 by default)
        check=True,  # Raise an exception if the command returns a non-zero exit code
    )

    if "execution time" not in seqpare.stdout:
        raise RuntimeError("Encountered issue executing seqpare: " + str(seqpare))

    print("seqpare: ", seqpare.stdout)
    scores = dict[str, float]()

    with open("out.tmp.tsv", "r") as f:
        next(f)  # skip header

        for line in f:
            split = line.split("\t")

            if len(split) != 6:
                raise RuntimeError(
                    "Seqpare output parsing issue; expected 5 columns in out.tmp.tsv"
                )

            score = float(split[4])
            # a/b/c.d -> c
            other = split[5].split("/")[-1].split(".")[0]
            scores[other] = score

    os.remove("out.tmp.tsv")
    os.remove("tmp.bed")
    return scores


def ensure_dataset(combo_path, roadep_path):
    cache = dict[int, list[GenomicInterval]]()

    for i, name in enumerate(rme.bedNames):
        path = fix_bed_ext(f"{roadep_path}/{name}")
        cache[i] = list(iter(BedDataset(path)))

    progress = 0

    for i in range(len(rme.bedNames)):
        for j in range(i + 1, len(rme.bedNames)):
            print(progress)
            progress += 1

            terms = [i, j]
            k = j

            while len(cache[k]) < 100:
                k = (k + 1) % len(rme.bedNames)
                terms.append(k)

            content: list[GenomicInterval] = [
                interval for i in terms for interval in cache[i]
            ]

            scores = seqpare_raw(roadep_path, content)
            combo_name = "-".join([rme.bedNames[i] for i in terms])
            pickle.dump(scores, open(f"{combo_path}/{combo_name}.pickle", "wb"))


def main():
    combo_path = "data/roadep_combo"
    roadep_path = "data/roadmap_epigenomics/beds"
    fasta_path = "data/hg/hg19.fa"
    model_path = "models/hdnaSeqpareFT"
    ensure_dataset(combo_path, roadep_path)
