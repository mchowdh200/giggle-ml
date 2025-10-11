"""Direct zarr writer for efficient distributed embedding storage."""

import os
from collections.abc import Iterable, Sequence
from typing import Any, final

import numpy as np
import zarr
from filelock import FileLock


@final
class MultiZarrWriter:
    """Handles direct writing to zarr files with on-demand creation and safe resizing."""

    def __init__(
        self,
        output_paths: Sequence[str],
        shape: tuple[int, ...],
        chunks: tuple[int, ...] | bool | None = None,
        dtype: Any = None,
        **kwargs: Any,
    ):
        """Initialize the zarr writer with zarr.open-like configuration.

        Args:
            output_paths: Paths to output zarr files
            shape: Shape of the zarr arrays (e.g., (None, embed_dim))
            chunks: Chunk shape for zarr arrays
            dtype: Data type for zarr arrays
            **kwargs: Additional arguments passed to zarr.open
        """
        self.output_paths = output_paths
        self.shape = shape
        self.chunks = chunks
        self.dtype = dtype
        self.kwargs = kwargs

        # Track current state
        self.current_zarr_array: zarr.Array | None = None
        self.current_set_idx: int | None = None

    def write_batch(
        self, data: Iterable[np.ndarray[Any, Any]], batch_indices: list[tuple[int, int]]
    ) -> None:
        """Write a batch of data using optimized sliced writes for contiguous runs."""
        if not batch_indices:
            return

        # Convert iterable to a list to allow slicing for chunking.
        # This is a trade-off for performance, holding the batch in memory.
        data_list = list(data)
        if len(data_list) != len(batch_indices):
            raise ValueError("Data and indices must have the same length.")

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
                self.current_zarr_array = self._open_or_create_zarr_file(
                    self.output_paths[run_set_idx]
                )
                self.current_set_idx = run_set_idx

            if self.current_zarr_array is None:
                raise RuntimeError("No zarr array available for writing.")

            # 2. Ensure the array is large enough for the entire run.
            run_end_pos = batch_indices[end_of_run_idx][1]
            self._ensure_zarr_size(run_set_idx, required_size=run_end_pos + 1)

            # 3. Stack the data for the run and perform a single sliced write.
            data_chunk = np.vstack(data_list[start_of_run_idx : end_of_run_idx + 1])
            self.current_zarr_array[run_start_pos : run_end_pos + 1] = data_chunk

            # 4. Advance to the start of the next run.
            start_of_run_idx = end_of_run_idx + 1

    def _ensure_zarr_size(self, set_idx: int, required_size: int) -> None:
        """Use a file lock to safely resize the zarr array if needed."""
        if self.current_zarr_array is None:
            raise RuntimeError("Cannot ensure size of a non-existent zarr array.")

        current_size = self.current_zarr_array.shape[0]
        if required_size > current_size:
            lock_path = f"{self.output_paths[set_idx]}.resize.lock"
            with FileLock(lock_path):
                # Re-check size after acquiring lock (another process may have resized)
                current_size = self.current_zarr_array.shape[0]
                if required_size > current_size:
                    # Resize to accommodate new position along the first dimension
                    new_shape = (required_size,) + self.shape[1:]
                    self.current_zarr_array.resize(new_shape)

    def _open_or_create_zarr_file(self, output_path: str) -> zarr.Array:
        """Open existing zarr file or create new one safely."""
        # This method remains unchanged from your version.

        # concurrency-safe because in the event of double-initialization, the processes
        # write the same information

        if os.path.exists(output_path):
            zarr_array = zarr.open(output_path, mode="a")
        else:
            zarr_array = zarr.open(
                output_path,
                mode="w",
                shape=(0,) + self.shape[1:],
                chunks=self.chunks,
                dtype=self.dtype,
                **self.kwargs,
            )

        assert isinstance(zarr_array, zarr.Array)
        return zarr_array
