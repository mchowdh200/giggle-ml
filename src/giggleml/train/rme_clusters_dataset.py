"""
RME (Roadmap Epigenomics) dataset for sampling interval clusters.
"""

import itertools
from collections.abc import Iterable, Iterator
from os import PathLike
from pathlib import Path
from random import Random
from typing import NamedTuple, override

import numpy as np
import torch  # <-- Added import
from numpy._typing import NDArray
from torch.utils.data import IterableDataset

from giggleml.data_wrangling.interval_dataset import BedDataset
from giggleml.train.seqpare_db import SeqpareDB
from giggleml.utils.path_utils import fix_bed_ext
from giggleml.utils.types import GenomicInterval, lazy
from giggleml.utils.utils.collection_utils import as_list
from giggleml.utils.utils.lru_cache import lru_cache

# A cluster is a list of bed files
Cluster = list[list[GenomicInterval]]


# a batch, to be mined for triplets
class MiningBatch(NamedTuple):
    interval_groups: list[list[GenomicInterval]]
    adjacency_matrix: torch.Tensor


@lazy
class RmeSeqpareClusters(IterableDataset):
    """
    Dataset yields a series of interval clusters where only intervals within the same cluster should be considered positive.

    Where a cluster is a collection of interval groups, all interval groups within the cluster are guaranteed to be above a
    similarity threshold. Interval groups in other clusters are guaranteed to be below that similarity threshold.

    This dataset is typically used in combination with a subsequent triplet mining algorithm that operates on the
    generated clusters to create training triplets across all ranks simultaneously.

    Supports resumption via save_dict() and load_dict() methods to preserve random number generator states and
    training progress across sessions.
    """

    def __init__(
        self,
        road_epig_path: PathLike,
        seqpare: SeqpareDB,
        world_size: int,
        rank: int,
        positive_threshold: float = 0.96,
        clusters_amnt: int = 10,
        groups_per_cluster: int = 10,
        intervals_per_group: int = 30,
        allowed_rme_names: Iterable[str] | None = None,
        seed: int = 42,
    ) -> None:
        """
        Initializes the dataset.

        Args:
            road_epig_path: Path to the road epigenomics data.
            seqpare: An instance of the SeqpareDB.
            positive_threshold: the minimum similarity for all items in a cluster.
            clusters_amnt: Total clusters
            groups_per_cluster: Number of interval groups per cluster.
            intervals_per_group: Number of intervals per group.
            allowed_rme_names: Iterable of RME names for cross-validation. Defaults to all names.
            seed: Random seed for reproducibility.
        """
        # Assign attributes first
        self.rme_dir: PathLike = road_epig_path
        self.sdb: SeqpareDB = seqpare
        self.positive_threshold: float = positive_threshold
        self.clusters_amnt: int = clusters_amnt
        self.groups_per_cluster: int = groups_per_cluster
        self.intervals_per_group: int = intervals_per_group
        self.seed: int = seed

        # Validate parameters
        if clusters_amnt <= 0:
            raise ValueError(f"clusters_amnt must be positive, got {clusters_amnt}")
        if not (0.0 < positive_threshold <= 1.0):
            raise ValueError(
                f"positive_threshold must be in (0, 1], got {positive_threshold}"
            )
        if groups_per_cluster <= 0:
            raise ValueError(
                f"groups_per_cluster must be positive, got {groups_per_cluster}"
            )
        if intervals_per_group <= 0:
            raise ValueError(
                f"intervals_per_group must be positive, got {intervals_per_group}"
            )

        def total_possible_rme_names() -> Iterator[str]:
            for entry in Path(self.rme_dir).iterdir():
                if entry.is_file():
                    name = entry.name
                    if name.endswith(".bed.gz"):
                        yield name[:-7]  # Remove .bed.gz
                    elif name.endswith(".bed"):
                        yield name[:-4]  # Remove .bed

        self.allowed_rme_names: set[str] = set(
            total_possible_rme_names()
            if allowed_rme_names is None
            else allowed_rme_names
        )

        self._allowed_mask: NDArray[np.bool_] = self.sdb.labels_to_mask(
            self.allowed_rme_names
        )
        self._rng: Random = Random(self.seed)

    @override
    def __iter__(self) -> Iterator[MiningBatch]:
        while True:
            anchors = self._sample_anchors()  # same across ranks
            # since _sample_neighbors can self-sample anchors, this is all sampled labels
            nodes = list(
                itertools.chain.from_iterable(
                    [self._sample_neighbors(anchor) for anchor in anchors]
                )
            )
            node_labels = [self.sdb.idx_to_label(idx) for idx in nodes]
            # only with indices contained in the subgraph
            adjacency_matrix_list = [
                self.sdb.fetch_mask(label, self.positive_threshold)[nodes]
                for label in node_labels
            ]
            interval_samples = [self._sample_group(label) for label in node_labels]

            # --- Modified Lines ---
            # 1. Stack the list of numpy arrays into one contiguous numpy array
            adj_matrix_np = np.array(adjacency_matrix_list)
            # 2. Convert the numpy array to a torch tensor
            adj_matrix_tensor = torch.from_numpy(adj_matrix_np)
            yield MiningBatch(interval_samples, adj_matrix_tensor)
            # --- End Modified Lines ---

    @as_list
    def _sample_neighbors(self, anchor: str) -> Iterator[int]:
        positive_mask = (
            self.sdb.fetch_mask(anchor, self.positive_threshold) & self._allowed_mask
        )
        positive_indices = np.flatnonzero(positive_mask)

        if len(positive_indices) == 0:
            # this shouldn't happen because the anchor is self-similar
            raise RuntimeError(
                f"No positive samples found for anchor '{anchor}' with threshold {self.positive_threshold}. "
                f"Try lowering positive_threshold or expanding allowed_rme_names."
            )

        for _ in range(self.groups_per_cluster):
            yield self._rng.choice(positive_indices)

    def _sample_group(self, anchor: str) -> list[GenomicInterval]:
        bed = self._fetch_bed(fix_bed_ext(Path(self.rme_dir, anchor)))

        if len(bed) == 0:
            raise RuntimeError(
                f"Bed file for label '{anchor}' is empty. Check data integrity."
            )

        k = min(len(bed), self.intervals_per_group)
        return self._rng.choices(bed, k=k)

    @lru_cache(max_size=128)
    def _fetch_bed(self, path: Path) -> list[GenomicInterval]:
        return list(iter(BedDataset(path)))

    def _sample_anchors(self) -> list[str]:
        """same across ranks"""
        return self._rng.choices(list(self.allowed_rme_names), k=self.clusters_amnt)

    def save_state(self) -> dict:
        """
        Save the current state of the dataset for resumption.

        Returns:
            Dictionary containing all necessary state information for resumption.
        """
        return {
            "rng_state": self._rng.getstate(),
        }

    def load_state(self, state_dict: dict) -> "RmeSeqpareClusters":
        """
        Load dataset from saved state for resumption.

        Args:
            state_dict: Dictionary containing saved state information.

        Returns:
            self
        """
        # Restore random number generator state
        self._rng.setstate(state_dict["rng_state"])
        return self
