"""
RME (Roadmap Epigenomics) dataset for sampling interval clusters.
"""

from collections.abc import Iterator
from pathlib import Path
from random import Random
from typing import Any, override

from torch.utils.data import IterableDataset

import giggleml.utils.roadmap_epigenomics as rme
from giggleml.data_wrangling.interval_dataset import BedDataset
from giggleml.train.seqpare_db import SeqpareDB
from giggleml.utils.misc import partition_list
from giggleml.utils.path_utils import as_path, fix_bed_ext
from giggleml.utils.types import GenomicInterval

# A cluster is a list of bed files
Cluster = list[list[GenomicInterval]]


class RmeSeqpareClusters(IterableDataset):
    """The dataset yields a series of interval clusters where only intervals within the same cluster should be considered positive."""

    def __init__(
        self,
        road_epig_path: str | Path,
        seqpare: SeqpareDB,
        world_size: int,
        rank: int,
        clusters_amnt: int = 10,
        cluster_size: int = 10,
        density: int = 30,
        allowed_rme_names: list[str] | None = None,
        seed: int = 42,
        start_iteration: int = 0,
    ) -> None:
        """
        @param density: the amount of intervals per candidate
        @param allowed_rme_names: List of RME names to restrict sampling to. If None, uses all available.
        @param seed: Random seed for reproducibility
        @param start_iteration: For resumption, skip this many iterations
        """
        super().__init__()
        self.road_epig_path: Path = as_path(road_epig_path)
        self.seqpare_db: "SeqpareDB" = seqpare
        self.world_size: int = world_size
        self.rank: int = rank
        self.anchors: int = clusters_amnt
        self.cluster_size: int = cluster_size
        self.density: int = density
        self.allowed_rme_names: set[str] = (
            set(allowed_rme_names) if allowed_rme_names else set(rme.bed_names)
        )
        self.seed: int = seed
        self.start_iteration: int = start_iteration
        self._rng: Random = Random(seed)
        self._iteration_count: int = 0

    @override
    def __iter__(self) -> Iterator[list[Cluster]]:
        # "my" -- this rank
        my_rng: Random = Random(self.rank + self.seed)

        # Filter available bed names to only allowed ones
        available_names = [
            name for name in rme.bed_names if name in self.allowed_rme_names
        ]
        if not available_names:
            raise ValueError("No available RME names after filtering")

        # Reset iteration counter for each iterator
        self._iteration_count = 0

        while True:
            # Skip iterations for resumption
            if self._iteration_count < self.start_iteration:
                # Advance RNG state to maintain reproducibility
                self._rng.choices(available_names, k=self.anchors)
                my_anchors = partition_list(
                    list(range(self.anchors)), self.world_size, self.rank
                )
                for _ in my_anchors:
                    my_rng.choices(
                        [0, 1], k=self.cluster_size
                    )  # Dummy choices to advance state
                    for _ in range(self.cluster_size):
                        my_rng.choices(
                            [0, 1], k=self.density
                        )  # Dummy choices to advance state
                self._iteration_count += 1
                continue

            # an identical value across the world size
            anchors = self._rng.choices(available_names, k=self.anchors)
            my_anchors = partition_list(anchors, self.world_size, self.rank)
            my_clusters = list[Cluster]()

            for anchor in my_anchors:
                neighbors, _ = self.seqpare_db.fetch(anchor)
                # Filter neighbors to only allowed names
                allowed_neighbors = [
                    n for n in neighbors if n in self.allowed_rme_names
                ]
                # in the event that an anchor has little neighbors, this will repeatedly resample
                # from the same labels
                candidate_labels = [anchor] + allowed_neighbors
                labels = my_rng.choices(candidate_labels, k=self.cluster_size)
                cluster: Cluster = list()

                # map into interval list
                for label in labels:
                    path = fix_bed_ext(self.road_epig_path / label)
                    if not path.exists():
                        raise FileNotFoundError(f"BED file not found: {path}")
                    bed = list(iter(BedDataset(path)))
                    intervals = my_rng.choices(bed, k=self.density)
                    cluster.append(intervals)

                my_clusters.append(cluster)

            self._iteration_count += 1
            yield my_clusters

    def get_state(self) -> dict[str, Any]:
        """Get current dataset state for resumption."""
        return {
            "iteration_count": self._iteration_count,
            "rng_state": self._rng.getstate(),
            "seed": self.seed,
            "start_iteration": self.start_iteration,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Set dataset state for resumption."""
        self._iteration_count = state["iteration_count"]
        self._rng.setstate(state["rng_state"])
        self.seed = state["seed"]
        self.start_iteration = state["start_iteration"]

