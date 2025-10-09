import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import cast, final, overload, override

import numpy as np
import torch
import torch.distributed as dist
import zarr

from giggleml.utils.torch_utils import get_rank

# INFO: ----------------
#         Targets
# ----------------------


class _Target(ABC):
    @property
    @abstractmethod
    def rank_ordinal(self) -> int: ...


class _AllRanks(_Target):
    @property
    @override
    def rank_ordinal(self) -> int:
        return -1


class _ThisRank(_Target):
    @property
    @override
    def rank_ordinal(self) -> int:
        return get_rank()


class _Global(_Target):
    @property
    @override
    def rank_ordinal(self) -> int:
        return -2


@dataclass(frozen=True)
class _SpecificRank(_Target):
    rank: int

    @property
    @override
    def rank_ordinal(self) -> int:
        return self.rank


@final
class RankConsumerTarget:
    """A namespace for referencing target types."""

    # This makes the class act as a pure namespace and prevents instantiation.
    def __init__(self):
        raise TypeError("PossibleTargets is a namespace and cannot be instantiated.")

    # Assign the "private" types to public class attributes.
    AllRanks = _AllRanks()
    ThisRank = _ThisRank()
    Global = _Global()
    SpecificRank = _SpecificRank


# INFO: ----------------
#        RankZarr
# ----------------------


class _RankZarrWriter:
    """A callable object that appends tensors to a specific Zarr array."""

    def __init__(self, path: PathLike, chunk_rows: int | None = None):
        self.path = str(path)
        self.chunk_rows = chunk_rows
        self._zarr_array: zarr.Array | None = None

    def __call__(self, tensors: Iterable[torch.Tensor]):
        """Appends an iterable of tensors to the Zarr array."""
        numpy_tensors = [t.cpu().numpy() for t in tensors]
        if not numpy_tensors:
            return

        if self._zarr_array is None:
            # First write op: create the Zarr array based on the first tensor.
            first_tensor = numpy_tensors[0]
            shape = (0, *first_tensor.shape[1:])
            chunks = (
                (self.chunk_rows, *(-1 for _ in first_tensor.shape[1:]))
                if self.chunk_rows
                else True
            )
            self._zarr_array = cast(
                zarr.Array,
                zarr.open(
                    self.path,
                    mode="w",
                    shape=shape,
                    chunks=chunks,
                    dtype=first_tensor.dtype,
                ),
            )
        self._zarr_array.append(np.array(numpy_tensors))


class RankZarr:
    """Manages a collection of Zarr arrays distributed by rank."""

    def __init__(self, zarr_path: PathLike):
        """
        Intermediate zarr files are stored as zarr_path.rank{rank}
        """
        self.base_path = Path(zarr_path)
        self.base_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_path(self, rank: _Target) -> Path:
        """Gets the file path for a specific, non-aggregate rank target."""
        if isinstance(rank, _Global):
            return self.base_path
        return self.base_path.with_name(
            f"{self.base_path.name}.rank{rank.rank_ordinal}"
        )

    def _get_all_rank_paths(self) -> Iterator[Path]:
        """Gets all rank-specific file paths, excluding the global one."""
        parent = self.base_path.parent
        pattern = f"{self.base_path.name}.rank*"
        return parent.glob(pattern)

    def writer(
        self, rank: _Target = RankConsumerTarget.ThisRank, chunk_rows: int | None = None
    ) -> _RankZarrWriter:
        """
        Returns a writer object for a specific target rank.

        Args:
            rank: The target rank to write to. Cannot be AllRanks.
            chunk_rows: The number of rows to store per chunk in the Zarr array.
        """
        if isinstance(rank, _AllRanks):
            raise ValueError("Cannot create a writer for all ranks simultaneously.")

        path = self._get_path(rank)
        return _RankZarrWriter(path, chunk_rows)

    def read(
        self, rank: _Target = RankConsumerTarget.ThisRank
    ) -> Iterator[torch.Tensor]:
        """Returns an iterator over tensors from the specified target rank(s)."""
        paths_to_read: list[Path] = []
        if isinstance(rank, _AllRanks):
            paths_to_read.extend(self._get_all_rank_paths())
        else:
            paths_to_read.append(self._get_path(rank))

        for path in paths_to_read:
            if path.exists() and path.is_dir():
                try:
                    zarr_array = zarr.open(str(path), mode="r")
                    for item in zarr_array:
                        # .copy() decouples from Zarr's internal buffers
                        assert isinstance(item, np.ndarray)
                        yield torch.from_numpy(item.copy())
                except Exception:
                    # Skips empty or corrupted Zarr directories.
                    continue

    def delete(self, rank: _Target = RankConsumerTarget.ThisRank):
        """
        Deletes the Zarr store(s) for the given rank target.
        RankConsumerTarget.AllRanks deletes all rank-specific files but
        leaves the global zarr file untouched.
        """
        paths_to_delete: list[Path] = []
        if isinstance(rank, _AllRanks):
            paths_to_delete.extend(self._get_all_rank_paths())
        else:
            paths_to_delete.append(self._get_path(rank))

        for path in paths_to_delete:
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
