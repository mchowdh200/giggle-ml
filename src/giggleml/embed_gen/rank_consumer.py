import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Protocol, cast, final, overload, override

import numpy as np
import torch
import torch.distributed as dist
import zarr

from giggleml.iter_utils.rank_iter import RankIter
from giggleml.utils.torch_utils import get_rank

# INFO: -----------------
#        Targets
# -----------------------


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


# INFO: --------------------
#         Interfaces
# --------------------------


class RankConsumerWriter[T](Protocol):
    def __call__(self, items: Iterable[T]) -> None: ...


class RankConsumer[T](Protocol):
    def writer(self, rank: _Target = RankConsumerTarget.ThisRank) -> RankConsumerWriter[T]: ...
    
    def read(self, rank: _Target = RankConsumerTarget.ThisRank) -> list[T] | list[list[T]] | Iterator[T]: ...


# INFO: --------------------
#         RankCache
# --------------------------


class _RankCacheWriter[T]:
    def __init__(self, cache: list[T]):
        self.cache: list[T] = cache

    def __call__(self, items: Iterable[T]):
        self.cache.extend(items)


class RankCache[T]:
    def __init__(self):
        """Initializes a cache that stores data locally for the current rank."""
        self._local_cache: list[T] = []

    def writer(
        self, rank: _Target = RankConsumerTarget.ThisRank
    ) -> _RankCacheWriter[T]:
        """
        Returns a writer object for the current rank.

        Note: Writing is only permitted for `ThisRank`.
        """
        if rank is not RankConsumerTarget.ThisRank:
            raise ValueError(
                "Writers for RankCache can only be created for the current process (ThisRank)."
            )
        return _RankCacheWriter(self._local_cache)

    @overload
    def read(self, rank: _AllRanks) -> list[list[T]]: ...

    @overload
    def read(self, rank: _Target) -> list[T]: ...

    def read(
        self, rank: _Target = RankConsumerTarget.ThisRank
    ) -> list[T] | list[list[T]]:
        """
        Reads data from the specified rank(s).

        If the target is `ThisRank`, it returns the local data directly.
        If the target is `AllRanks` or a different `SpecificRank`, it performs
        a gather operation across all processes.
        If the target is `Global`, it gathers data from all ranks and deshards it
        using RankIter.inverse to reconstruct the original order.
        """
        if isinstance(rank, _Global):
            # Recursively call with AllRanks and then apply inverse to deshard the data
            all_ranks_data = self.read(RankConsumerTarget.AllRanks)
            return list(RankIter.inverse(all_ranks_data))

        # Local read is now a direct return of the instance's list.
        if rank.rank_ordinal == get_rank():
            return self._local_cache

        # Cross-rank read requires a distributed environment.
        if not dist.is_initialized():
            if isinstance(rank, _AllRanks):
                return [self._local_cache]
            raise ValueError(
                f"Cannot read from rank {rank.rank_ordinal} in a non-distributed environment."
            )

        # Perform the distributed gather operation using the local cache.
        world_size = dist.get_world_size()
        gathered_caches: list[list[T]] = [[] for _ in range(world_size)]
        dist.all_gather_object(gathered_caches, self._local_cache)

        if isinstance(rank, _AllRanks):
            return gathered_caches

        if isinstance(rank, _SpecificRank):
            if 0 <= rank.rank_ordinal < world_size:
                return gathered_caches[rank.rank_ordinal]
            raise ValueError(
                f"Target rank {rank.rank_ordinal} is out of bounds for world size {world_size}."
            )

        raise TypeError(f"Unhandled target type for cross-rank read: {type(rank)}")


# INFO: -------------------
#         RankZarr
# -------------------------


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

    def consolidate(self, chunk_rows: int | None = None):
        """
        Consolidates all rank-specific zarr files into the global zarr file.

        Reads from all rank-specific files, uses RankIter.inverse to deshard
        the data in streaming fashion, and writes to the global zarr file.

        Args:
            chunk_rows: The number of rows to store per chunk in the global Zarr array.
        """
        # Get iterators for all ranks using existing read method
        all_ranks_iterators = self.read(RankConsumerTarget.AllRanks)

        # Use RankIter.inverse to deshard the data in streaming fashion
        desharded_tensors = RankIter.inverse(all_ranks_iterators)

        # Write consolidated data to global zarr file
        global_writer = self.writer(RankConsumerTarget.Global, chunk_rows)
        global_writer(desharded_tensors)

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

    @overload
    def get_zarr_array(
        self, rank: _AllRanks = RankConsumerTarget.AllRanks
    ) -> list[zarr.Array]: ...

    @overload
    def get_zarr_array(self, rank: _Target) -> zarr.Array | None: ...

    def get_zarr_array(
        self, rank: _Target = RankConsumerTarget.ThisRank
    ) -> zarr.Array | None | list[zarr.Array]:
        """
        Returns the underlying Zarr array(s) for direct access.

        Args:
            rank: The target rank to get the Zarr array for.

        Returns:
            - For ThisRank, Global, or SpecificRank: The Zarr array if it exists, None otherwise.
            - For AllRanks: A list of Zarr arrays from all available rank files.

        Raises:
            ValueError: If SpecificRank is out of bounds in distributed environment.
            RuntimeError: If Zarr file is corrupted or cannot be opened.
        """
        if isinstance(rank, _AllRanks):
            # Return arrays from all available rank files
            arrays = []
            for path in self._get_all_rank_paths():
                if path.exists() and path.is_dir():
                    try:
                        arrays.append(zarr.open(str(path), mode="r"))
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to open Zarr array at {path}: {e}"
                        ) from e
            return arrays

        # Validate SpecificRank bounds in distributed environment
        if isinstance(rank, _SpecificRank) and dist.is_initialized():
            world_size = dist.get_world_size()
            if not (0 <= rank.rank_ordinal < world_size):
                raise ValueError(
                    f"Target rank {rank.rank_ordinal} is out of bounds for world size {world_size}."
                )

        path = self._get_path(rank)
        if not path.exists():
            return None

        if not path.is_dir():
            return None

        try:
            return cast(zarr.Array, zarr.open(str(path), mode="r"))
        except Exception as e:
            raise RuntimeError(f"Failed to open Zarr array at {path}: {e}") from e
