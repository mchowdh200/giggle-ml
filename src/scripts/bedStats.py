#!/usr/bin/env python3

import argparse
import gzip
import os
import statistics
import sys


def calculate_bed_stats(bed_files):
    """
    Calculates statistics from a list of gzipped BED files.

    Args:
        bed_files (list): A list of paths to gzipped BED files.

    Returns:
        dict: A dictionary containing calculated statistics.
              Returns None if no valid data could be processed.
    """
    all_interval_lengths = []
    intervals_per_file = []
    total_intervals_processed = 0
    files_processed_count = 0
    files_with_errors = 0  # Counter for files skipped or having line errors

    print(f"Processing {len(bed_files)} file(s)...", file=sys.stderr)

    for bed_path in bed_files:
        current_file_interval_count = 0
        file_had_line_error = False  # Track errors within this specific file
        intervals_in_this_file = []  # Store lengths for this file temporarily

        if not os.path.exists(bed_path):
            print(f"Warning: File not found, skipping: {bed_path}", file=sys.stderr)
            # Increment files_with_errors here as the file itself is problematic
            # Don't increment files_processed_count
            continue  # Skip to the next file path

        try:
            with gzip.open(bed_path, "rt") as f:  # 'rt' mode reads as text
                print(f"Reading file: {bed_path}", file=sys.stderr)
                line_num = 0
                for line in f:
                    line_num += 1
                    line = line.strip()
                    # Skip empty lines or common header/comment lines
                    if not line or line.startswith(("#", "track", "browser")):
                        continue

                    fields = line.split("\t")
                    if len(fields) < 3:
                        print(
                            f"Warning: Skipping malformed line {line_num} in {bed_path} (less than 3 columns): {line}",
                            file=sys.stderr,
                        )
                        file_had_line_error = True
                        continue

                    try:
                        # BED format: chrom (col 0), start (col 1, 0-based), end (col 2, exclusive)
                        start = int(fields[1])
                        end = int(fields[2])

                        if start < 0 or end < 0:
                            print(
                                f"Warning: Skipping line {line_num} in {bed_path} (negative coordinate): {line}",
                                file=sys.stderr,
                            )
                            file_had_line_error = True
                            continue
                        if end < start:
                            print(
                                f"Warning: Skipping line {line_num} in {bed_path} (end < start): {line}",
                                file=sys.stderr,
                            )
                            file_had_line_error = True
                            continue

                        length = end - start
                        intervals_in_this_file.append(length)
                        current_file_interval_count += 1

                    except ValueError:
                        print(
                            f"Warning: Skipping line {line_num} in {bed_path} (non-integer start/end): {line}",
                            file=sys.stderr,
                        )
                        file_had_line_error = True
                        continue
                    except IndexError:
                        print(
                            f"Warning: Skipping malformed line {line_num} in {bed_path} (index error): {line}",
                            file=sys.stderr,
                        )
                        file_had_line_error = True
                        continue

            # File was opened and read successfully, increment processed count
            files_processed_count += 1
            intervals_per_file.append(current_file_interval_count)
            all_interval_lengths.extend(
                intervals_in_this_file
            )  # Add lengths from this file to the global list
            total_intervals_processed += current_file_interval_count
            if file_had_line_error:
                # If there were line errors, also count this file towards the error count
                files_with_errors += 1

        except gzip.BadGzipFile:
            print(
                f"Error: File is not a valid gzip file or is corrupted, skipping: {bed_path}",
                file=sys.stderr,
            )
            files_with_errors += 1  # Count as an errored file
        except Exception as e:
            print(
                f"Error: An unexpected error occurred while processing {bed_path}, skipping: {e}",
                file=sys.stderr,
            )
            files_with_errors += 1  # Count as an errored file

    print(
        f"Finished processing. Files successfully read: {files_processed_count}. Files skipped or with read/line errors: {files_with_errors}.",
        file=sys.stderr,
    )

    # --- Calculate Statistics ---
    results = {
        "files_processed": files_processed_count,
        "total_intervals": total_intervals_processed,
        "interval_length_mean": None,
        "interval_length_stdev": None,
        "interval_length_min": None,
        "interval_length_max": None,
        "intervals_per_file_mean": None,
        "intervals_per_file_stdev": None,
        "intervals_per_file_min": None,
        "intervals_per_file_max": None,
    }

    # Interval Length Stats (across all intervals from all files)
    if len(all_interval_lengths) > 0:
        results["interval_length_min"] = min(all_interval_lengths)
        results["interval_length_max"] = max(all_interval_lengths)
        results["interval_length_mean"] = statistics.mean(all_interval_lengths)
        if len(all_interval_lengths) > 1:
            results["interval_length_stdev"] = statistics.stdev(all_interval_lengths)
        else:
            results["interval_length_stdev"] = 0.0  # Stdev is 0 for a single point
    else:
        print(
            "Warning: No valid intervals found across all files to calculate length statistics.",
            file=sys.stderr,
        )

    # Intervals Per File Stats (distribution of counts)
    if len(intervals_per_file) > 0:
        results["intervals_per_file_min"] = min(intervals_per_file)
        results["intervals_per_file_max"] = max(intervals_per_file)
        results["intervals_per_file_mean"] = statistics.mean(intervals_per_file)
        if len(intervals_per_file) > 1:
            results["intervals_per_file_stdev"] = statistics.stdev(intervals_per_file)
        else:
            results["intervals_per_file_stdev"] = 0.0  # Stdev is 0 for a single file
    else:
        print(
            "Warning: No files were successfully processed to calculate per-file statistics.",
            file=sys.stderr,
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate basic statistics from gzipped BED files."
    )
    parser.add_argument(
        "bed_files",
        metavar="BED_FILE",
        nargs="+",  # Accepts one or more file names
        help="Path(s) to gzipped BED file(s) (.bed.gz).",
    )

    args = parser.parse_args()

    stats = calculate_bed_stats(args.bed_files)

    # Check if *any* files were processed before attempting to print stats
    if stats and stats["files_processed"] > 0:
        print("\n--- Statistics Summary ---")
        print(f"Files successfully processed: {stats['files_processed']}")
        print(f"Total valid intervals found: {stats['total_intervals']}")

        print("\n# Interval Length Stats (across all intervals):")
        # Check if interval stats could be calculated (i.e., total_intervals > 0)
        if stats["total_intervals"] > 0:
            print(f"  Min Length:    {stats['interval_length_min']}")  # Typically integer
            print(f"  Max Length:    {stats['interval_length_max']}")  # Typically integer
            print(f"  Mean Length:   {stats['interval_length_mean']:.2f}")
            # Stdev can be None if only 1 interval total, or calculated as 0.0
            stdev_val = stats["interval_length_stdev"]
            print(f"  Length Stdev: {stdev_val:.2f}")
        else:
            print("  (No intervals found)")

        print("\n# Intervals Per File Stats:")
        # Check if per-file stats could be calculated (i.e., files_processed > 0)
        if stats["files_processed"] > 0:
            print(f"  Min Intervals/File:    {stats['intervals_per_file_min']}")  # Integer
            print(f"  Max Intervals/File:    {stats['intervals_per_file_max']}")  # Integer
            print(f"  Mean Intervals/File:   {stats['intervals_per_file_mean']:.2f}")
            # Stdev can be None if only 1 file, or calculated as 0.0
            stdev_val_per_file = stats["intervals_per_file_stdev"]
            print(f"  Stdev Intervals/File: {stdev_val_per_file:.2f}")
        # This else shouldn't be strictly needed due to the outer check, but safe
        # else:
        #    print("  (No files processed)")
        print("------------------------")

    elif stats and stats["files_processed"] == 0:
        print(
            "\nNo files could be successfully processed. Check file paths and formats.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        # Should not happen if calculate_bed_stats returns None, but as a fallback
        print("\nNo statistics could be calculated.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
