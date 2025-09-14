"""
SeqpareDB for reading and processing seqpare similarity files.
"""

import re
from collections.abc import Iterable
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
            suffix = "".join(file.suffixes)

            if suffix.endswith(".bed.gz.tsv"):
                label = file.name[: -len(".bed.gz.tsv")]
                self._labels[label] = len(self._labels)
            elif suffix.endswith(".bed.tsv"):
                label = file.name[: -len(".bed.tsv")]
                self._labels[label] = len(self._labels)

    @cache
    def fetch_mask(self, label: str, positive_threshold: float) -> NDArray[np.bool]:
        path = self.dir / (label + ".bed.tsv")

        if not path.exists():
            path = self.dir / (label + ".bed.gz.tsv")

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

                # these are in the form ./dir/dir2/label.bed.gz
                if match := re.match(r"(.+/)*(.+)\.bed(\.gz)?", item):
                    other_label = match.group(2)
                else:
                    raise ValueError(f"malformed seqpare item: {item}")

                if other_label not in self._labels:
                    continue  # skip unknown labels

                item_id = self._labels[other_label]
                positive = float(terms[4]) >= positive_threshold
                bits[item_id] = positive

            return bits

    def fetch_labels(
        self, label: str, positive_threshold: float = 0.7
    ) -> tuple[list[str], list[str]]:
        """
        fetches the (positives, negatives) labels for a given label
        @param label: the label to fetch
        @param positive_threshold: the threshold above which a label is considered positive
        @returns (positives, negatives) for a label
        """
        mask = self.fetch_mask(label, positive_threshold)
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
