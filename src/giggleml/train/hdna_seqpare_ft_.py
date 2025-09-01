import os
import pickle
import subprocess
from collections.abc import Iterator, Sequence
from functools import cache

from torch.utils.data import IterableDataset

import giggleml.utils.roadmap_epigenomics as rme
from giggleml.data_wrangling.interval_dataset import BedDataset
from giggleml.utils.misc import fix_bed_ext
from giggleml.utils.types import GenomicInterval

Bed = Sequence[GenomicInterval]
Example = tuple[Sequence[Bed], float]


class SeqpareDataset(IterableDataset):
    def __init__(self, roadep_path: str, combo_path: str):
        self.roadep_path: str = roadep_path
        self.combo_path: str = combo_path

    @cache
    def _roadep_cache(self, i: int):
        name = rme.bed_names[i]
        path = fix_bed_ext(f"{self.roadep_path}/{name}")
        return list(iter(BedDataset(path)))

    def seqpare_raw(self, content: list[GenomicInterval]):
        with open("tmp.bed", "w") as f:
            for chr, start, end in content:
                f.write(f"{chr}\t{start}\t{end}\n")

        seqpare = subprocess.run(
            [
                "seqpare",
                self.roadep_path + "/*",
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

    def _seqpare_cache(self, terms: Sequence[int]) -> dict[str, float]:
        combo_name = "-".join([rme.bed_names[i] for i in terms])
        file_path = f"{self.combo_path}/{combo_name}.pickle"

        if os.path.isfile(file_path):
            scores = pickle.load(open(file_path, "rb"))

            if not isinstance(scores, dict):
                raise RuntimeError("Invalid format: " + file_path)

            return scores

        content: list[GenomicInterval] = [
            interval for i in terms for interval in self._roadep_cache(i)
        ]
        scores = self.seqpare_raw(content)
        pickle.dump(scores, open(file_path, "wb"))
        return scores

    def __iter__(self) -> Iterator[Example]:
        for idx in range(len(rme.bed_names)):
            for idx2 in range(idx + 1, len(rme.bed_names)):
                terms = [idx, idx2]
                idx3 = idx2

                while len(self._roadep_cache(idx3)) < 100:
                    idx3 = (idx3 + 1) % len(rme.bed_names)
                    terms.append(idx3)

                scores = self._seqpare_cache(terms)


def main():
    combo_path = "data/roadep_combo"
    roadep_path = "data/roadmap_epigenomics/beds"
    fasta_path = "data/hg/hg19.fa"
    model_path = "models/hdnaSeqpareFT"
    ensure_dataset(combo_path, roadep_path)
