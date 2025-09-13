"""
SeqpareDB for reading and processing seqpare similarity files.
"""

from functools import cache
from pathlib import Path

import numpy as np
from numpy._typing import NDArray

from giggleml.utils.path_utils import as_path
from giggleml.utils.types import lazy


@lazy
class SeqpareDB:
    """reads all seqpare form .tsv files in the directory, taking the name, without last suffix, as the label"""

    def __init__(self, dir: str | Path):
        self.dir: Path = as_path(dir)
        self._labels: dict[str, int] = dict()

        for file in self.dir.iterdir():
            if file.suffix == ".tsv":
                self._labels[file.stem] = len(self._labels)

    @cache
    def _fetch_dense(self, label: str, positive_threshold: float) -> NDArray[np.bool]:
        path = self.dir / (label + ".bed.tsv")

        if not path.exists():
            raise FileNotFoundError(path)

        with open(path, "r") as f:
            next(f)
            bits: NDArray[np.bool] = np.zeros(len(self._labels), dtype=np.bool)

            for line in f:
                # parse the seqpare tsv file
                terms = line.split()
                if len(terms) < 6:
                    continue  # skip malformed lines
                # the column that corresponds to file names
                item = terms[5]
                # these are in the form ./label.bed
                if not item.startswith("./") or len(item) < 3:
                    continue  # skip unexpected format
                label_name = item[2:]
                if label_name not in self._labels:
                    continue  # skip unknown labels
                item_id = self._labels[label_name]
                positive = float(terms[4]) > positive_threshold
                bits[item_id] = positive

            return bits

    def fetch(
        self, label: str, positive_threshold: float = 0.7
    ) -> tuple[list[str], list[str]]:
        """
        positives are items less than the positive_threshold to the anchor
        @returns (positives, negatives) for a label
        """
        bits = self._fetch_dense(label, positive_threshold)
        positives, negatives = list(), list()
        labels = list(self._labels.keys())

        for i, bit in enumerate(bits):
            label = labels[i]

            if bit:
                positives.append(label)
            else:
                negatives.append(label)

        return positives, negatives
