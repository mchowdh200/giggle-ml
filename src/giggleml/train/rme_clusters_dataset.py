"""
RME (Roadmap Epigenomics) dataset for sampling interval clusters.
"""

from collections.abc import Iterable, Iterator
from os import PathLike
from pathlib import Path
from random import Random
from typing import override

import numpy as np
from numpy._typing import NDArray
from torch.utils.data import IterableDataset

from giggleml.data_wrangling.interval_dataset import BedDataset
from giggleml.train.seqpare_db import SeqpareDB
from giggleml.utils.misc import partition_list
from giggleml.utils.path_utils import fix_bed_ext
from giggleml.utils.types import GenomicInterval, lazy
from giggleml.utils.utils.collection_utils import as_list
from giggleml.utils.utils.lru_cache import lru_cache

# A cluster is a list of bed files
Cluster = list[list[GenomicInterval]]


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
        positive_threshold: float = 0.7,
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
            world_size: The total number of processes in the distributed setup.
            rank: The rank of the current process.
            positive_threshold: the minimum similarity for all items in a cluster.
            clusters_amnt: Total clusters across all processes. Must be divisible by world_size.
            groups_per_cluster: Number of interval groups per cluster.
            intervals_per_group: Number of intervals per group.
            allowed_rme_names: Iterable of RME names for cross-validation. Defaults to all names.
            seed: Random seed for reproducibility.
        """
        # Assign attributes first
        self.rme_dir: PathLike = road_epig_path
        self.sdb: SeqpareDB = seqpare
        self.world_size: int = world_size
        self.rank: int = rank
        self.positive_threshold: float = positive_threshold
        self.clusters_amnt: int = clusters_amnt
        self.groups_per_cluster: int = groups_per_cluster
        self.intervals_per_group: int = intervals_per_group
        self.seed: int = seed

        # Validate parameters
        if clusters_amnt % world_size != 0:
            raise ValueError(
                f"clusters_amnt ({clusters_amnt}) must be divisible by world_size ({world_size})"
            )
        if clusters_amnt <= 0:
            raise ValueError(f"clusters_amnt must be positive, got {clusters_amnt}")
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank ({rank}) must be in range [0, {world_size})")
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

        @as_list
        def total_possible_rme_names() -> Iterable[str]:
            for entry in Path(self.rme_dir).iterdir():
                if entry.is_file() and entry.suffix in {".bed", ".bed.gz"}:
                    yield entry.stem

        self.allowed_rme_names: set[str] = set(
            allowed_rme_names if allowed_rme_names else total_possible_rme_names()
        )

        self._allowed_mask: NDArray[np.bool_] = self.sdb.labels_to_mask(
            self.allowed_rme_names
        )
        self._world_rng: Random = Random(self.seed)
        self._rank_rng: Random = Random(self.seed + self.rank)

    @override
    def __iter__(self) -> Iterator[list[Cluster]]:
        anchors = self._sample_anchors()  # same across ranks
        rank_anchors = partition_list(anchors, self.world_size, self.rank)
        yield [self._sample_positives(anchor) for anchor in rank_anchors]

    @as_list
    def _sample_positives(self, anchor: str) -> Iterable[list[GenomicInterval]]:
        positive_mask = (
            self.sdb.fetch_mask(anchor, self.positive_threshold / 2)
            & self._allowed_mask
        )

        positive_indices = np.flatnonzero(positive_mask)
        if len(positive_indices) == 0:
            raise RuntimeError(
                f"No positive samples found for anchor '{anchor}' with threshold {self.positive_threshold}. "
                f"Try lowering positive_threshold or expanding allowed_rme_names."
            )

        for _ in range(self.groups_per_cluster):
            # 1. choose label (of the positives)
            # 2. collect intervals from that bed file
            label_idx = self._rank_rng.choice(positive_indices)
            label = self.sdb.idx_to_label(label_idx)
            bed = self._fetch_bed(fix_bed_ext(Path(self.rme_dir, label)))

            if len(bed) == 0:
                raise RuntimeError(
                    f"Bed file for label '{label}' is empty. Check data integrity."
                )

            yield self._rank_rng.choices(bed, k=self.intervals_per_group)

    @lru_cache(max_size=128)
    def _fetch_bed(self, path: Path) -> list[GenomicInterval]:
        return list(iter(BedDataset(path)))

    @as_list
    def _sample_anchors(self) -> Iterator[str]:
        """same across ranks"""

        seed_anchor = self._world_rng.choice(list(self.allowed_rme_names))
        available = (
            ~self.sdb.fetch_mask(seed_anchor, 2 * self.positive_threshold)
            & self._allowed_mask
        )

        yield seed_anchor

        for _ in range(self.clusters_amnt // self.world_size - 1):
            available_indices = np.flatnonzero(available)
            if len(available_indices) == 0:
                raise RuntimeError(
                    f"Ran out of sufficiently distant anchors after {_ + 1} iterations. "
                    f"Current settings: clusters_amnt={self.clusters_amnt}, world_size={self.world_size}, "
                    f"positive_threshold={self.positive_threshold}, allowed_rme_names={len(self.allowed_rme_names)}. "
                    f"Try reducing clusters_amnt, increasing allowed_rme_names, or decreasing positive_threshold."
                )

            next_anchor_idx = self._world_rng.choice(available_indices)
            yield (next_anchor := self.sdb.idx_to_label(next_anchor_idx))
            available &= ~self.sdb.fetch_mask(next_anchor, self.positive_threshold)

    def save_dict(self) -> dict:
        """
        Save the current state of the dataset for resumption.

        Returns:
            Dictionary containing all necessary state information for resumption.
        """
        return {
            "rme_dir": str(self.rme_dir),
            "world_size": self.world_size,
            "rank": self.rank,
            "positive_threshold": self.positive_threshold,
            "clusters_amnt": self.clusters_amnt,
            "groups_per_cluster": self.groups_per_cluster,
            "intervals_per_group": self.intervals_per_group,
            "allowed_rme_names": list(self.allowed_rme_names),
            "seed": self.seed,
            # Random number generator states
            "world_rng_state": self._world_rng.getstate(),
            "rank_rng_state": self._rank_rng.getstate(),
        }

    @classmethod
    def load_dict(cls, state_dict: dict, seqpare: SeqpareDB) -> "RmeSeqpareClusters":
        """
        Load dataset from saved state for resumption.

        Args:
            state_dict: Dictionary containing saved state information.
            seqpare: An instance of the SeqpareDB (not serialized due to complexity).

        Returns:
            RmeSeqpareClusters instance restored from saved state.
        """
        # Create instance with saved parameters
        instance = cls(
            road_epig_path=state_dict["rme_dir"],
            seqpare=seqpare,
            world_size=state_dict["world_size"],
            rank=state_dict["rank"],
            positive_threshold=state_dict["positive_threshold"],
            clusters_amnt=state_dict["clusters_amnt"],
            groups_per_cluster=state_dict["groups_per_cluster"],
            intervals_per_group=state_dict["intervals_per_group"],
            allowed_rme_names=state_dict["allowed_rme_names"],
            seed=state_dict["seed"],
        )

        # Restore random number generator states
        instance._world_rng.setstate(state_dict["world_rng_state"])
        instance._rank_rng.setstate(state_dict["rank_rng_state"])

        return instance
