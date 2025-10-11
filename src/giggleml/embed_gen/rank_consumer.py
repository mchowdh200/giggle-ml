"""Distributed data consumption and persistence for multi-rank environments.

This module provides abstractions for writing and reading data across multiple
ranks in distributed training environments. It supports both in-memory caching
and persistent Zarr-based storage with automatic sharding and consolidation.

The utilities consist of a series of channels that support reading and writing.
Each rank is given a channel and have access to a global channel.

Key Components:
    - RankConsumer: Protocol for rank-aware data consumers
    - RankCache: In-memory storage with distributed gather operations
    - RankZarr: Persistent Zarr-based storage with rank sharding
    - RankConsumerTarget: Target specification for rank operations

Example:
    >>> cache = RankCache[torch.Tensor]()
    >>> writer = cache.writer()
    >>> writer([tensor1, tensor2])
    >>> data = cache.read(RankConsumerTarget.AllRanks)
"""

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
    """Abstract base class for rank targeting in distributed operations.

    Defines the interface for specifying which rank(s) to target for
    read/write operations in a distributed environment.
    """

    @property
    @abstractmethod
    def rank_ordinal(self) -> int:
        """Returns the ordinal number identifying this target.

        Returns:
            int: Rank ordinal. Special values:
                -1: All ranks
                -2: Global (consolidated across all ranks)
                >=0: Specific rank number
        """
        ...


class _AllRanks(_Target):
    """Target representing all ranks in the distributed environment.

    Used for operations that should gather data from or operate on
    all participating ranks simultaneously.
    """

    @property
    @override
    def rank_ordinal(self) -> int:
        """Returns -1 to indicate all ranks target."""
        return -1


class _ThisRank(_Target):
    """Target representing the current process's rank.

    Automatically resolves to the rank of the current process,
    making it suitable for local operations.
    """

    @property
    @override
    def rank_ordinal(self) -> int:
        """Returns the current process's rank."""
        return get_rank()


class _Global(_Target):
    """Target representing globally consolidated data.

    Used for operations that require data to be desharded and
    reconstructed in its original order across all ranks.
    """

    @property
    @override
    def rank_ordinal(self) -> int:
        """Returns -2 to indicate global consolidation target."""
        return -2


@dataclass(frozen=True)
class _SpecificRank(_Target):
    """Target representing a specific rank by its ordinal number.

    Attributes:
        rank: The specific rank number to target (0-based indexing).
    """

    rank: int
    """The rank number to target (must be >= 0)."""

    @property
    @override
    def rank_ordinal(self) -> int:
        """Returns the specific rank number."""
        return self.rank


@final
class RankConsumerTarget:
    """Namespace providing predefined rank targeting options.

    This class acts as a factory and namespace for creating rank targets
    used in distributed operations. It cannot be instantiated and serves
    purely as a collection of target types.

    Attributes:
        AllRanks: Target all ranks simultaneously
        ThisRank: Target the current process's rank
        Global: Target globally consolidated data
        SpecificRank: Factory for targeting a specific rank by number

    Example:
        >>> # Target current rank
        >>> writer = consumer.writer(RankConsumerTarget.ThisRank)
        >>>
        >>> # Target all ranks
        >>> data = consumer.read(RankConsumerTarget.AllRanks)
        >>>
        >>> # Target specific rank
        >>> rank_2_data = consumer.read(RankConsumerTarget.SpecificRank(2))
    """

    def __init__(self):
        """Prevents instantiation of this namespace class."""
        raise TypeError("RankConsumerTarget is a namespace and cannot be instantiated.")

    # Predefined target instances
    AllRanks = _AllRanks()
    """Target for operations affecting all ranks."""

    ThisRank = _ThisRank()
    """Target for operations on the current process's rank."""

    Global = _Global()
    """Target for globally consolidated operations."""

    SpecificRank = _SpecificRank
    """Factory for creating targets for specific rank numbers."""


# INFO: --------------------
#         Interfaces
# --------------------------


class RankConsumerWriter[T](Protocol):
    """Protocol for writing data items to a rank-specific destination.

    This protocol defines the interface for writer objects that can
    consume iterables of items and persist them to rank-specific storage.

    Type Parameters:
        T: The type of items being written

    Example:
        >>> writer = consumer.writer(RankConsumerTarget.ThisRank)
        >>> writer([item1, item2, item3])
    """

    def __call__(self, items: Iterable[T]) -> None:
        """Writes an iterable of items to the target destination.

        Args:
            items: An iterable of items to write. The items will be
                  consumed and stored according to the writer's configuration.
        """
        ...


class RankConsumer[T](Protocol):
    """Protocol for distributed data consumption with rank-aware operations.

    This protocol defines the interface for objects that can write data
    to and read data from rank-specific storage in distributed environments.
    Implementations handle the complexity of distributed coordination,
    data sharding, and consolidation.

    Type Parameters:
        T: The type of items being stored and retrieved

    Example:
        >>> consumer = RankCache[torch.Tensor]()
        >>> writer = consumer.writer(RankConsumerTarget.ThisRank)
        >>> writer([tensor1, tensor2])
        >>> local_data = consumer.read(RankConsumerTarget.ThisRank)
        >>> all_data = consumer.read(RankConsumerTarget.AllRanks)
    """

    def writer(
        self, rank: _Target = RankConsumerTarget.ThisRank
    ) -> RankConsumerWriter[T]:
        """Creates a writer for the specified rank target.

        Args:
            rank: The target rank for write operations. Defaults to
                 the current process's rank.

        Returns:
            A writer object that can consume iterables of items.

        Raises:
            ValueError: If the rank target is not supported for writing.
        """
        ...

    def read(
        self, rank: _Target = RankConsumerTarget.ThisRank
    ) -> list[T] | list[list[T]] | Iterator[T]:
        """Reads data from the specified rank target.

        Args:
            rank: The target rank for read operations. Defaults to
                 the current process's rank.

        Returns:
            The return type depends on the target:
            - ThisRank/SpecificRank: list[T] - Data from single rank
            - AllRanks: list[list[T]] - Data from all ranks as nested lists
            - Global: list[T] or Iterator[T] - Consolidated data

        Raises:
            ValueError: If the rank target is invalid or inaccessible.
        """
        ...


# INFO: --------------------
#         RankCache
# --------------------------


class _RankCacheWriter[T]:
    """Writer implementation for RankCache that appends to an in-memory list.

    This writer provides direct access to a local cache list, allowing
    efficient in-memory accumulation of items for the current rank.

    Type Parameters:
        T: The type of items being cached
    """

    def __init__(self, cache: list[T]):
        """Initializes the writer with a reference to the cache list.

        Args:
            cache: The list to append items to. This is typically a reference
                  to the RankCache's internal storage.
        """
        self.cache: list[T] = cache

    def __call__(self, items: Iterable[T]) -> None:
        """Appends all items from the iterable to the cache.

        Args:
            items: An iterable of items to add to the cache.
        """
        self.cache.extend(items)


class RankCache[T]:
    """In-memory cache for rank-specific data with distributed operations.

    RankCache provides a simple in-memory storage solution that can
    coordinate with other ranks in a distributed environment. It supports
    local writes and distributed reads with automatic data gathering.

    Type Parameters:
        T: The type of items being cached

    Features:
        - Local in-memory storage for fast access
        - Distributed gather operations for cross-rank data access
        - Automatic data desharding for global reconstruction
        - Thread-safe within single rank operations

    Example:
        >>> cache = RankCache[str]()
        >>> writer = cache.writer()
        >>> writer(["item1", "item2"])
        >>>
        >>> # Read local data
        >>> local_items = cache.read(RankConsumerTarget.ThisRank)
        >>>
        >>> # Read from all ranks (requires distributed environment)
        >>> all_items = cache.read(RankConsumerTarget.AllRanks)
    """

    def __init__(self):
        """Initializes an empty cache for the current rank."""
        self._local_cache: list[T] = []

    def writer(
        self, rank: _Target = RankConsumerTarget.ThisRank
    ) -> _RankCacheWriter[T]:
        """Creates a writer for adding items to the cache.

        RankCache only supports writing to the current rank to maintain
        data consistency and avoid cross-process coordination complexity.

        Args:
            rank: Must be RankConsumerTarget.ThisRank. Other targets
                 will raise an error.

        Returns:
            A writer object that appends items to the local cache.

        Raises:
            ValueError: If rank is not ThisRank.

        Example:
            >>> cache = RankCache[int]()
            >>> writer = cache.writer()
            >>> writer([1, 2, 3])
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
        """Reads cached data from the specified rank target.

        This method provides flexible access to cached data across different
        scopes in a distributed environment. It automatically handles the
        coordination required for cross-rank data access.

        Args:
            rank: The target to read from. Options:
                - ThisRank: Returns local cache directly (no coordination)
                - SpecificRank(n): Returns cache from rank n (requires distributed env)
                - AllRanks: Returns list of caches from all ranks
                - Global: Returns desharded data in original order

        Returns:
            - For ThisRank/SpecificRank: list[T] containing items from that rank
            - For AllRanks: list[list[T]] where each inner list is one rank's data
            - For Global: list[T] with items desharded to original order

        Raises:
            ValueError: If trying to access specific rank in non-distributed environment
                       or if rank number is out of bounds.

        Note:
            Cross-rank operations require torch.distributed to be initialized.
            In non-distributed mode, only ThisRank and AllRanks (returning local
            cache only) are supported.

        Example:
            >>> cache = RankCache[str]()
            >>> writer = cache.writer()
            >>> writer(["local_item"])
            >>>
            >>> # Local read
            >>> local_data = cache.read()  # Returns ["local_item"]
            >>>
            >>> # Distributed read (if multiple ranks)
            >>> all_data = cache.read(RankConsumerTarget.AllRanks)
            >>> # Returns [["rank0_items"], ["rank1_items"], ...]
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
    """Writer for appending PyTorch tensors to Zarr arrays.

    This writer handles the conversion from PyTorch tensors to NumPy arrays
    and manages the creation and growth of Zarr arrays on disk. The Zarr array
    is created lazily on the first write operation based on the shape and dtype
    of the first tensor.

    Attributes:
        path: The filesystem path where the Zarr array is stored
        chunk_rows: Number of rows per chunk for the Zarr array
        zarr_array: The underlying Zarr array (None until first write)

    Note:
        The zarr_array attribute is None until the first call with data.
        This allows the writer to infer the correct shape and dtype from
        the actual data rather than requiring upfront specification.
    """

    def __init__(self, path: PathLike, chunk_rows: int | None = None):
        """Initializes a Zarr writer for the specified path.

        Args:
            path: Filesystem path where the Zarr array will be stored.
                 The path will be converted to a string for Zarr compatibility.
            chunk_rows: Number of rows to store per chunk in the Zarr array.
                       If None, Zarr will use automatic chunking. Chunking
                       affects compression and I/O performance.
        """
        self.path = str(path)
        self.chunk_rows = chunk_rows
        self.zarr_array: zarr.Array | None = None

    def __call__(self, tensors: Iterable[torch.Tensor]) -> None:
        """Appends tensors to the Zarr array, creating it if necessary.

        This method converts PyTorch tensors to NumPy arrays and appends them
        to the Zarr array. On the first call, it creates the Zarr array with
        the appropriate shape and dtype inferred from the data.

        Args:
            tensors: An iterable of PyTorch tensors to append. All tensors
                    must have the same shape (except for the batch dimension)
                    and dtype. Empty iterables are handled gracefully.

        Note:
            - Tensors are automatically moved to CPU before conversion
            - The Zarr array shape is (batch_size, *tensor_shape[1:])
            - Batch dimension grows dynamically with each append operation
            - All tensors in a single call are batched together

        Example:
            >>> writer = _RankZarrWriter("/path/to/data.zarr")
            >>> tensors = [torch.randn(10, 256), torch.randn(5, 256)]
            >>> writer(tensors)  # Creates array with shape (15, 256)
        """
        numpy_tensors = [t.cpu().numpy() for t in tensors]
        if not numpy_tensors:
            return

        if self.zarr_array is None:
            # First write: create the Zarr array based on the first tensor
            first_tensor = numpy_tensors[0]
            shape = (0, *first_tensor.shape[1:])
            chunks = (
                (self.chunk_rows, *(-1 for _ in first_tensor.shape[1:]))
                if self.chunk_rows
                else True
            )
            self.zarr_array = cast(
                zarr.Array,
                zarr.open(
                    self.path,
                    mode="w",
                    shape=shape,
                    chunks=chunks,
                    dtype=first_tensor.dtype,
                ),
            )
        self.zarr_array.append(np.array(numpy_tensors))


class RankZarr:
    """Persistent storage manager for rank-distributed Zarr arrays.

    RankZarr provides a distributed storage solution using Zarr arrays,
    where each rank writes to its own file and data can be consolidated
    globally. It supports both rank-specific operations and global
    consolidation with automatic desharding.

    Features:
        - Rank-specific Zarr file management
        - Lazy array creation based on data characteristics
        - Global consolidation with data desharding
        - Streaming operations for large datasets
        - Automatic directory management

    File Organization:
        - Individual rank files: base_path.rank{N} (e.g., data.zarr.rank0)
        - Global consolidated file: base_path (e.g., data.zarr)

    Example:
        >>> zarr_storage = RankZarr("/path/to/embeddings.zarr")
        >>> writer = zarr_storage.writer(chunk_rows=1000)
        >>> writer([tensor1, tensor2])
        >>>
        >>> # Read from current rank
        >>> local_tensors = list(zarr_storage.read())
        >>>
        >>> # Consolidate all ranks into global file
        >>> zarr_storage.consolidate()
    """

    def __init__(self, zarr_path: PathLike):
        """Initializes the Zarr storage manager.

        Args:
            zarr_path: Base path for Zarr storage. Individual rank files
                      will be created with .rank{N} suffixes, and the
                      global consolidated file will use this path directly.

        Note:
            The parent directory is created automatically if it doesn't exist.
        """
        self.base_path = Path(zarr_path)
        self.base_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_path(self, rank: _Target) -> Path:
        """Resolves the filesystem path for a given rank target.

        Args:
            rank: The rank target to resolve a path for.

        Returns:
            Path object pointing to the appropriate Zarr storage location.
            Global targets return the base path, others get rank-specific names.
        """
        if isinstance(rank, _Global):
            return self.base_path
        return self.base_path.with_name(
            f"{self.base_path.name}.rank{rank.rank_ordinal}"
        )

    def _get_all_rank_paths(self) -> Iterator[Path]:
        """Discovers all existing rank-specific Zarr files.

        Returns:
            Iterator over Path objects for all rank-specific files
            found in the base directory. Excludes the global file.

        Note:
            This method performs filesystem globbing to find files matching
            the rank naming pattern. It only returns existing files.
        """
        parent = self.base_path.parent
        pattern = f"{self.base_path.name}.rank*"
        return parent.glob(pattern)

    def writer(
        self, rank: _Target = RankConsumerTarget.ThisRank, chunk_rows: int | None = None
    ) -> _RankZarrWriter:
        """Creates a writer for appending tensors to a rank-specific Zarr array.

        Args:
            rank: The target rank to write to. Must be a specific rank
                 (ThisRank, SpecificRank, or Global). AllRanks is not supported
                 since it would require coordinating writes across processes.
            chunk_rows: Number of rows per chunk in the Zarr array. This affects
                       compression efficiency and I/O performance. If None,
                       Zarr uses automatic chunking.

        Returns:
            A writer object that can append iterables of tensors to the
            specified rank's Zarr array.

        Raises:
            ValueError: If rank is AllRanks (ambiguous write target).

        Example:
            >>> zarr_storage = RankZarr("/data/embeddings.zarr")
            >>> writer = zarr_storage.writer(chunk_rows=1000)
            >>> tensors = [torch.randn(100, 768) for _ in range(10)]
            >>> writer(tensors)
        """
        if isinstance(rank, _AllRanks):
            raise ValueError("Cannot create a writer for all ranks simultaneously.")

        path = self._get_path(rank)
        return _RankZarrWriter(path, chunk_rows)

    def read(
        self, rank: _Target = RankConsumerTarget.ThisRank
    ) -> Iterator[torch.Tensor]:
        """Reads tensors from the specified rank target(s).

        Args:
            rank: The target to read from:
                - ThisRank/SpecificRank/Global: Reads from single rank file
                - AllRanks: Reads from all existing rank files in sequence

        Returns:
            Iterator yielding PyTorch tensors. For AllRanks, tensors from
            different ranks are yielded in file discovery order (not
            necessarily rank order).

        Note:
            - Only existing Zarr directories are processed
            - Corrupted or empty Zarr files are silently skipped
            - Tensors are copied from Zarr buffers to avoid memory issues
            - For desharded/global reading, use consolidate() first

        Example:
            >>> zarr_storage = RankZarr("/data/embeddings.zarr")
            >>> # Read from current rank
            >>> for tensor in zarr_storage.read():
            ...     process(tensor)
            >>>
            >>> # Read from all ranks
            >>> all_tensors = list(zarr_storage.read(RankConsumerTarget.AllRanks))
        """
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
                    # Skips empty or corrupted Zarr directories
                    continue

    def consolidate(self, chunk_rows: int | None = None) -> None:
        """Consolidates all rank-specific Zarr files into a single global file.

        This method reads data from all rank-specific files and uses RankIter.inverse
        to deshard the data back to its original order before distributed processing.
        The consolidated data is written to the global Zarr file in streaming fashion
        to handle large datasets efficiently.

        Args:
            chunk_rows: Number of rows per chunk in the consolidated Zarr array.
                       This can be different from the original rank-specific chunking
                       to optimize for the consolidated access patterns.

        Note:
            - This operation requires all rank files to exist
            - The global file will be overwritten if it exists
            - Data is processed in streaming fashion to minimize memory usage
            - Original rank files are preserved (use delete() to remove them)

        Example:
            >>> zarr_storage = RankZarr("/data/embeddings.zarr")
            >>> # After all ranks have written their data
            >>> zarr_storage.consolidate(chunk_rows=5000)
            >>> # Now global file contains desharded data
            >>> consolidated_data = list(zarr_storage.read(RankConsumerTarget.Global))
        """
        # Get iterators for all ranks using existing read method
        all_ranks_iterators = self.read(RankConsumerTarget.AllRanks)

        # Use RankIter.inverse to deshard the data in streaming fashion
        desharded_tensors = RankIter.inverse(all_ranks_iterators)

        # Write consolidated data to global zarr file
        global_writer = self.writer(RankConsumerTarget.Global, chunk_rows)
        global_writer(desharded_tensors)

    def delete(self, rank: _Target = RankConsumerTarget.ThisRank) -> None:
        """Deletes Zarr storage for the specified rank target(s).

        This method removes Zarr directories from the filesystem. It can target
        individual ranks or all rank-specific files while preserving the global file.

        Args:
            rank: The target to delete:
                - ThisRank/SpecificRank: Deletes single rank file
                - AllRanks: Deletes all rank-specific files (preserves global)
                - Global: Deletes the global consolidated file

        Note:
            - Only existing directories are deleted
            - AllRanks specifically excludes the global file for safety
            - Deletion is permanent and cannot be undone
            - No error is raised if files don't exist

        Example:
            >>> zarr_storage = RankZarr("/data/embeddings.zarr")
            >>> # Clean up after consolidation
            >>> zarr_storage.delete(RankConsumerTarget.AllRanks)
            >>> # Or delete specific rank
            >>> zarr_storage.delete(RankConsumerTarget.SpecificRank(0))
        """
        paths_to_delete: list[Path] = []
        if isinstance(rank, _AllRanks):
            paths_to_delete.extend(self._get_all_rank_paths())
        else:
            paths_to_delete.append(self._get_path(rank))

        for path in paths_to_delete:
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
