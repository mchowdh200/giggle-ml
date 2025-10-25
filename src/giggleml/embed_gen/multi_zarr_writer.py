"""Direct zarr writer for efficient distributed embedding storage."""

import os
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any, final

import numpy as np
import zarr
from filelock import FileLock


@final
class MultiZarrWriter:
    """
    Handles direct writing to zarr files with on-demand creation and safe resizing.
    Manages a collection of backing Zarr arrays assumed to have the same homogeneous
    shapes after the first dimension. Write index locality matters for performance.
    The first dimension of shape is ignored in favor of the pass initial length or zero.
    """

    def __init__(
        self,
        output_paths: Sequence[str],
        shape: tuple[int, ...],
        initial_lengths: Sequence[int] | None = None,
        chunks: tuple[int, ...] | bool | None = None,
        dtype: Any = None,
        grow_size: int = 1,
        **kwargs: Any,
    ):
        """Initialize the zarr writer with zarr.open-like configuration.

        Args:
            output_paths: Paths to output zarr files
            shape: Shape of the zarr arrays (e.g., (None, embed_dim))
            chunks: Chunk shape for zarr arrays
            dtype: Data type for zarr arrays
            grow_size: the amount to increase the array by when maxed out
            **kwargs: Additional arguments passed to zarr.open
        """
        self.output_paths = output_paths
        self.shape = shape
        self.initial_lengths = initial_lengths or tuple(0 for _ in output_paths)
        assert len(self.initial_lengths) == len(output_paths)
        self.chunks = chunks
        self.dtype = dtype
        self.kwargs = kwargs
        self.grow_size = grow_size

        # Track current state
        self.current_zarr_array: zarr.Array | None = None
        self.current_set_idx: int | None = None

    def write_batch(
        self,
        data: Sequence[np.ndarray[Any, Any]],
        batch_indices: Sequence[tuple[int, int]],
    ) -> None:
        """Write a batch of data using optimized sliced writes for contiguous runs."""

        if len(data) != len(batch_indices):
            raise ValueError(
                f"Data and indices must have the same length. "
                f"Got {len(data)} data items and {len(batch_indices)} indices."
            )

        start_of_run_idx = 0
        while start_of_run_idx < len(batch_indices):
            # A run is a contiguous block of indices for the same zarr array.
            # First, find the end of the current run.
            run_set_idx, run_start_pos = batch_indices[start_of_run_idx]
            end_of_run_idx = start_of_run_idx

            for i in range(start_of_run_idx + 1, len(batch_indices)):
                next_set_idx, next_pos = batch_indices[i]
                prev_pos = batch_indices[i - 1][1]

                # A run breaks if the set_idx changes or the position is not consecutive.
                if next_set_idx != run_set_idx or next_pos != prev_pos + 1:
                    break
                end_of_run_idx = i
            else:
                # This 'else' belongs to the 'for' loop and executes if the loop
                # completed without a 'break', meaning the run goes to the end of the batch.
                end_of_run_idx = len(batch_indices) - 1

            # Now, process the entire run from start_of_run_idx to end_of_run_idx.
            # 1. Switch the active zarr array if necessary.
            if run_set_idx != self.current_set_idx:
                self.current_zarr_array = self._open_or_create_zarr_file(run_set_idx)
                self.current_set_idx = run_set_idx

            if self.current_zarr_array is None:
                raise RuntimeError("No zarr array available for writing.")

            # 2. Ensure the array is large enough for the entire run.
            run_end_pos = batch_indices[end_of_run_idx][1]
            self.ensure_zarr_size(run_set_idx, required_size=run_end_pos)

            # 3. Stack the data for the run and perform a single sliced write.
            data_chunk = np.vstack(data[start_of_run_idx : end_of_run_idx + 1])
            self.current_zarr_array[run_start_pos : run_end_pos + 1] = data_chunk

            # 4. Advance to the start of the next run.
            start_of_run_idx = end_of_run_idx + 1

    @contextmanager
    def _modify_zarr_array_safely(
        self, set_idx: int
    ) -> Generator[zarr.Array, None, None]:
        """
        Provides a locked, refreshed zarr.Array object for safe modification.

        This method acquires a file lock, re-opens the zarr array from disk
        to get the ground truth, yields it for modification, and then updates
        the instance's current_zarr_array *if* it was the one being modified.
        """
        lock_path = f"{self.output_paths[set_idx]}.resize.lock"
        with FileLock(lock_path, timeout=30):
            # After acquiring the lock, we MUST get the ground truth from disk.
            z_refreshed = zarr.open(self.output_paths[set_idx], mode="a")
            assert isinstance(z_refreshed, zarr.Array)

            try:
                yield z_refreshed
            finally:
                # Crucially, update the instance's array to the refreshed one
                # *if* it's the one we currently have active.
                if set_idx == self.current_set_idx:
                    self.current_zarr_array = z_refreshed

    def ensure_zarr_size(self, set_idx: int, required_size: int) -> None:
        """Use a file lock to safely grow the zarr array if needed."""
        if self.current_zarr_array is None:
            raise RuntimeError("Cannot ensure size of a non-existent zarr array.")

        # First check (optimistic, no lock)
        # We only check the cached array reference if it's the active one.
        if (
            set_idx == self.current_set_idx
            and required_size <= self.current_zarr_array.shape[0]
        ):
            return

        # If check fails (or it's not the active array),
        # acquire lock and perform locked check + resize
        with self._modify_zarr_array_safely(set_idx) as z_refreshed:
            # This code runs *inside* the lock
            current_size = z_refreshed.shape[0]
            if required_size > current_size:
                # Resize to accommodate new position along the first dimension
                new_size = max(required_size, current_size + self.grow_size)
                new_shape = (new_size,) + self.shape[1:]
                z_refreshed.resize(new_shape)

    def set_zarr_size(self, set_idx: int, new_size: int) -> None:
        """
        Use a file lock to safely set the zarr array to an exact size.
        This function can truncate or grow the array.
        """
        with self._modify_zarr_array_safely(set_idx) as z_refreshed:
            # This code runs *inside* the lock
            current_size = z_refreshed.shape[0]
            if new_size != current_size:
                new_shape = (new_size,) + self.shape[1:]
                z_refreshed.resize(new_shape)

    def _open_or_create_zarr_file(self, set_idx: int) -> zarr.Array:
        """Open existing zarr file or create new one safely."""
        output_path = self.output_paths[set_idx]
        lock_path = f"{output_path}.init.lock"
        with FileLock(lock_path, timeout=30):
            if os.path.exists(output_path):
                zarr_array = zarr.open(output_path, mode="a")
            else:
                zarr_array = zarr.open(
                    output_path,
                    mode="w",
                    shape=(self.initial_lengths[set_idx], *self.shape[1:]),
                    chunks=self.chunks,
                    dtype=self.dtype,
                    **self.kwargs,
                )

            assert isinstance(zarr_array, zarr.Array)
            return zarr_array
