"""
SeqpareDB for reading and processing seqpare similarity files.
"""

import re
from collections.abc import Iterable, Iterator
from functools import cache
from os import PathLike
from pathlib import Path

import numpy as np
from numpy._typing import NDArray

from giggleml.utils.types import lazy
from giggleml.utils.utils.collection_utils import as_list


@lazy
class SeqpareDB:
    """reads all seqpare form .tsv files in the directory, taking the name, without last suffix, as the label"""

    def __init__(self, dir: PathLike):
        self.dir: Path = Path(dir)
        self._labels: dict[str, int] = dict()

        for file in self.dir.iterdir():
            suffix = "".join(file.suffixes)

            if suffix.endswith(".bed.gz.tsv"):
                label = file.name[: -len(".bed.gz.tsv")]
                self._labels[label] = len(self._labels)
            elif suffix.endswith(".bed.tsv"):
                label = file.name[: -len(".bed.tsv")]
                self._labels[label] = len(self._labels)

    @staticmethod
    @as_list
    def parse_seqpare_tsv(
        dir_root: PathLike, label: str
    ) -> Iterator[tuple[str, float]]:
        dir_root = Path(dir_root)
        path = dir_root / (label + ".bed.tsv")

        if not path.exists():
            path = dir_root / (label + ".bed.gz.tsv")

        if not path.exists():
            raise FileNotFoundError(path)

        with open(path, "r") as f:
            next(f)

            for line in f:
                if not line:
                    # skip empty lines
                    continue

                # parse the seqpare tsv file
                terms = line.split()

                if len(terms) < 6:
                    print(
                        f'Encountered malformed line without enough columns: "{line}"'
                    )
                    continue  # skip malformed lines

                # the column that corresponds to file names
                item_filename = terms[5]

                # these are in the form ./dir/dir2/label.bed.gz
                if match := re.match(r"(.+/)*(.+)\.bed(\.gz)?", item_filename):
                    item_label = match.group(2)
                else:
                    raise ValueError(f"malformed seqpare item: {item_filename}")

                # mapping into seqpare distance allows reasoning with the triangle inequality
                similarity = float(terms[4])
                yield (item_label, similarity)

    @cache
    def fetch_mask(self, label: str, distance_percentile: float) -> NDArray[np.bool]:
        data = SeqpareDB.parse_seqpare_tsv(self.dir, label)
        # mapping into seqpare distance allows reasoning with the triangle inequality
        data = [(label, 1 - sim) for label, sim in data]  # map into distance
        data.sort(key=lambda x: x[1])  # sort ascending

        distances = [x[1] for x in data]
        threshold = distances[round(len(distances) * distance_percentile)]

        bits: NDArray[np.bool] = np.zeros(len(self._labels), dtype=np.bool)

        for label, value in data:
            # skip unknown labels
            if label in self._labels:
                item_id = self._labels[label]
                bits[item_id] = value <= threshold

        return bits

    def fetch_labels(
        self, label: str, distance_percentile: float = 0.1
    ) -> tuple[list[str], list[str]]:
        """
        fetches the (positives, negatives) labels for a given label
        @param label: the label to fetc
        @param distance_percentile: the percentile for which items below that distance should be considered positive
        @returns (positives, negatives) for a label
        """
        mask = self.fetch_mask(label, distance_percentile)
        return self.mask_to_labels(mask)

    def mask_to_labels(self, mask: NDArray[np.bool]) -> tuple[list[str], list[str]]:
        """
        converts a mask to (positives, negatives) labels
        @returns (positives, negatives) for a label
        """
        positives, negatives = list(), list()
        labels = list(self._labels.keys())

        for i, bit in enumerate(mask):
            label = labels[i]

            if bit:
                positives.append(label)
            else:
                negatives.append(label)

        return positives, negatives

    def labels_to_mask(self, positives: Iterable[str]) -> NDArray[np.bool]:
        """
        converts a list of positive labels to a mask
        @param positives: the positive labels
        @returns the mask
        """
        mask: NDArray[np.bool] = np.zeros(len(self._labels), dtype=np.bool)

        for label in positives:
            if label not in self._labels:
                raise ValueError(f"unknown label: {label}")
            item_id = self._labels[label]
            mask[item_id] = True

        return mask

    def label_to_idx(self, label: str) -> int:
        """
        converts a label to its index
        @param label: the label to convert
        @returns the index of the label
        """
        if label not in self._labels:
            raise ValueError(f"unknown label: {label}")
        return self._labels[label]

    def idx_to_label(self, idx: int) -> str:
        """
        converts an index to its label
        @param idx: the index to convert
        @returns the label of the index
        """
        if idx < 0 or idx >= len(self._labels):
            raise ValueError(f"index out of range: {idx}")
        return list(self._labels.keys())[idx]
