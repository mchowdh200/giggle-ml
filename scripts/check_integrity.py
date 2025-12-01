import argparse
from collections import defaultdict
from pathlib import Path

import dask.array as da
import zarr

from giggleml.data_wrangling.interval_dataset import BedDataset
from giggleml.utils.print_utils import progress_logger


def main(paths: list[Path], bed_paths: list[Path] | None = None):
    issues: dict[Path, list[str]] = defaultdict(list)
    issue = lambda path, msg: issues[path].append(msg)
    total_elements = 0

    if bed_paths is not None:
        assert len(bed_paths) == len(paths), (
            "bed_paths and paths must be the same length"
        )

    with progress_logger(len(paths), "reading") as ckpt:
        for i, path in enumerate(paths):
            if not path.exists():
                issue(path, "missing file")
                continue

            # Open in read-only mode
            try:
                array = zarr.open_array(path, mode="r")
            except Exception as e:
                issue(path, f"Failed to open zarr array: {e}")
                ckpt()
                continue

            if any(x == 0 for x in array.shape):
                issue(path, "zero-length dimension(s)")

            # --- Check 1: Missing Chunks (Metadata check) ---
            # In Zarr, a "missing" chunk often simply implies a chunk full of the
            # fill_value (0), but for dense ML datasets, initialized < total
            # usually implies data loss or incomplete writes.
            if array.nchunks_initialized < array.nchunks:
                issue(
                    path,
                    f"Missing chunks: {array.nchunks_initialized}/{array.nchunks} initialized",
                )

            # --- Check 2: All Zeros (Content check) ---
            try:
                darr = da.from_zarr(array)

                # We want to check if an item (dim 0) is entirely zeros.
                # We reduce over all dimensions except the first one.
                reduce_axes = tuple(range(1, darr.ndim))

                # If 1D, reduce_axes is empty, we just check if element == 0
                # If nD, we check if all elements in that slice are 0
                if reduce_axes:
                    is_zero_item = (darr == 0).all(axis=reduce_axes)
                else:
                    is_zero_item = darr == 0

                # Compute the count of zero-items
                # .compute() triggers the actual lazy evaluation
                zero_count = is_zero_item.sum().compute()

                if zero_count > 0:
                    issue(
                        path,
                        f"Found {zero_count} items along primary dim with all zeros",
                    )

            except Exception as e:
                issue(path, f"Dask computation failed: {e}")

            # --- Check 3: (Optional) Compare bed file size ---
            size = array.shape[0]
            total_elements += size

            if bed_paths:
                bed = BedDataset(bed_paths[i])

                if len(bed) != size:
                    issue(
                        path,
                        f"Found {size} items, but expected {len(bed)} from BED file",
                    )

            ckpt()

        for path, messages in issues.items():
            print(path)
            for msg in messages:
                print("  -", msg)
            print("   ", len(messages))

        total = sum(len(msgs) for msgs in issues.values())
        print(f"Found {total_elements} items in all ({len(paths)}) files.")
        print(f"Found {total} issues in {len(issues)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check integrity of zarr array files")
    parser.add_argument("files", nargs="+", type=Path, help="Zarr array files to check")
    parser.add_argument(
        "--bed-files",
        nargs="*",
        type=Path,
        help="Optional BED files to compare against zarr array sizes (must match order of files)",
    )

    args = parser.parse_args()
    main(args.files, bed_paths=args.bed_files)
