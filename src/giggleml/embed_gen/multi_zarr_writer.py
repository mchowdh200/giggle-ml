"""Direct zarr writer for efficient distributed embedding storage."""

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
            shape: Shape of the zarr arrays (initial shape)
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
        """Write a batch of data to their corresponding zarr positions."""
        for item, (set_idx, position_in_set) in zip(data, batch_indices):
            # Switch zarr file if set index changed
            if set_idx != self.current_set_idx:
                self.current_zarr_array = self._open_or_create_zarr_file(
                    self.output_paths[set_idx]
                )
                self.current_set_idx = set_idx

            # Ensure zarr array is large enough for this position
            if self.current_zarr_array is None:
                raise RuntimeError("No zarr array available for writing")

            required_size = position_in_set + 1
            current_size = self.current_zarr_array.shape[0]

            if required_size > current_size:
                # Use filelock to synchronize resize operations across processes
                lock_path = f"{self.output_paths[set_idx]}.resize.lock"
                with FileLock(lock_path):
                    # Re-check size after acquiring lock (another process may have resized)
                    current_size = self.current_zarr_array.shape[0]
                    if required_size > current_size:
                        # Resize to accommodate new position along first dimension
                        new_shape = (required_size,) + self.shape[1:]
                        self.current_zarr_array.resize(new_shape)

            # Write data directly to the calculated position
            self.current_zarr_array[position_in_set] = item

    def _open_or_create_zarr_file(self, output_path: str) -> zarr.Array:
        """Open existing zarr file or create new one safely."""
        try:
            # Try to open existing zarr file in append mode
            zarr_array = zarr.open(output_path, mode="a")
        except FileNotFoundError:
            # File doesn't exist, create it (zarr.open handles concurrent creation safely)
            # Start with shape where first dimension is 0 (empty, but ready to grow)
            initial_shape = (0,) + self.shape[1:]
            zarr_array = zarr.open(
                output_path,
                mode="w",
                shape=initial_shape,
                chunks=self.chunks,
                dtype=self.dtype,
                **self.kwargs,
            )

        assert isinstance(zarr_array, zarr.Array)
        return zarr_array
