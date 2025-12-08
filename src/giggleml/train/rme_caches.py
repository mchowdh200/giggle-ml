from collections.abc import Iterable, Iterator
from functools import cache
from os import PathLike
from pathlib import Path
from typing import final

import torch
import zarr
from torch import Tensor

import giggleml.utils.roadmap_epigenomics as rme
from giggleml.data_wrangling.interval_dataset import BedDataset
from giggleml.utils.path_utils import fix_bed_ext
from giggleml.utils.utils.collection_utils import as_list


@final
class RmeFMCache:
    def __init__(self, embeds_dir: PathLike) -> None:
        self.embeds_dir = Path(embeds_dir)

    @cache
    def get_tensor(self, bed_name: str) -> Tensor:
        """
        Reads the entire Zarr array into memory and converts to a CPU Tensor.
        Cached, so this massive read only happens once per bed_name.
        """
        zarr_path = self.embeds_dir / f"{bed_name}.zarr"
        z_array = zarr.open_array(zarr_path, mode="r")
        # full in-memory cache
        return torch.from_numpy(z_array[:])

    @as_list
    def map(self, items: Iterable[tuple[int, Tensor]]) -> Iterator[Tensor]:
        """
        Map (set, row) indices to embeddings using in-memory Tensors.
        """
        for set_idx, row_indices in items:
            bed_name = rme.bed_names[set_idx]
            full_embeds = self.get_tensor(bed_name)

            # Direct Tensor Indexing
            indices = row_indices.long()
            yield full_embeds[indices]


@final
class RmeBedCache:
    def __init__(self, rme_dir: PathLike) -> None:
        self.rme_dir = Path(rme_dir)

    @staticmethod
    def clean_intervals(intervals: list[tuple[str, int, int]]) -> Tensor:
        def _chrm_id(chrm):
            if not chrm.startswith("chr"):
                raise ValueError(f"Unknown chromosome {chrm}")

            # FIXME: this is basically tokenization and should be moved into the model's tokenizer

            id = chrm[3:].lower()

            mapping = {
                "x": 23,
                "y": 24,
                "xy": 25,  # PAR
                "par": 25,
                "m": 26,
                "mt": 26,
            }

            if id in mapping:
                return mapping[id] - 1
            return int(id) - 1

        clean_intervals = [
            (_chrm_id(chrm), start, end) for (chrm, start, end) in intervals
        ]
        return torch.tensor(clean_intervals, dtype=torch.long, pin_memory=True)

    @cache
    def get_tensor(self, bed_name: str) -> Tensor:
        """
        Reads the entire Zarr array into memory and converts to a CPU Tensor.
        Cached, so this massive read only happens once per bed_name.
        """
        path = fix_bed_ext(self.rme_dir / bed_name)
        bed = list(iter(BedDataset(path)))
        return RmeBedCache.clean_intervals(bed)

    @as_list
    def map(self, items: Iterable[tuple[int, Tensor]]) -> Iterator[Tensor]:
        """
        Map (set, row) indices to embeddings using in-memory Tensors.
        """
        for set_idx, row_indices in items:
            bed_name = rme.bed_names[set_idx]
            full_embeds = self.get_tensor(bed_name)

            # Direct Tensor Indexing
            indices = row_indices.long()
            yield full_embeds[indices]
