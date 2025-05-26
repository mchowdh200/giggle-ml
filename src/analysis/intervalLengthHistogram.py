#!/usr/bin/env python3

import argparse
import gzip
import sys

import matplotlib.pyplot as plt
import numpy as np  # For percentile calculation


def get_interval_lengths(filepath, limit=None):
    """
    Reads a BED file (or gzipped BED file) and yields interval lengths.

    Args:
        filepath (str): Path to the BED file.
        limit (int, optional): Maximum number of intervals to read.
                               Defaults to None (read all).

    Yields:
        int: The length of each interval (end - start).
    """
    open_func = gzip.open if filepath.endswith(".gz") else open
    count = 0
    try:
        with open_func(filepath, "rt") as f:  # "rt" for text mode
            for line in f:
                if limit is not None and count >= limit:
                    break
                line = line.strip()
                if (
                    not line
                    or line.startswith("#")
                    or line.startswith("track")
                    or line.startswith("browser")
                ):
                    continue
                try:
                    fields = line.split("\t")
                    if len(fields) >= 3:
                        start = int(fields[1])
                        end = int(fields[2])
                        length = end - start
                        if length < 0:
                            print(
                                f"Warning: Skipped interval with negative length in {filepath}: {line}",
                                file=sys.stderr,
                            )
                            continue
                        yield length
                        count += 1
                    else:
                        print(
                            f"Warning: Skipped malformed line (fewer than 3 columns) in {filepath}: {line}",
                            file=sys.stderr,
                        )
                except ValueError:
                    print(
                        f"Warning: Skipped line with non-integer coordinates in {filepath}: {line}",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(
                        f"Warning: An unexpected error occurred processing line in {filepath}: {line} - {e}",
                        file=sys.stderr,
                    )
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return []  # Return empty iterator if file not found
    except gzip.BadGzipFile:
        print(f"Error: File is not a valid gzip file: {filepath}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error: Could not process file {filepath}: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Create a histogram of interval lengths from BED files with nuanced binning.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "bed_files",
        metavar="FILE",
        nargs="+",
        help="One or more BED file paths (.bed or .bed.gz).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: Maximum number of intervals to read from each file.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Optional: Number of bins for the histogram. Default is 50.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Histogram of Interval Lengths",
        help="Optional: Title for the histogram.",
    )
    parser.add_argument(
        "--xlabel",
        type=str,
        default="Interval Length (bp)",
        help="Optional: Label for the x-axis.",
    )
    parser.add_argument(
        "--ylabel", type=str, default="Frequency", help="Optional: Label for the y-axis."
    )
    parser.add_argument(
        "--log_y",  # Renamed for clarity
        action="store_true",
        help="Optional: Use a logarithmic scale for the y-axis.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Path to save the histogram image (e.g., plot.png). If not provided, displays the plot.",
    )
    # New arguments for nuanced binning/display
    parser.add_argument(
        "--min_display_length",
        type=int,
        default=None,
        help="Optional: Exclude interval lengths below this value from histogram display.",
    )
    parser.add_argument(
        "--max_display_percentile",
        type=float,
        default=None,
        help="Optional: Display interval lengths up to this percentile (0-100) of the data "
        "(after --min_display_length filter). E.g., 99.5",
    )
    parser.add_argument(
        "--max_display_absolute",
        type=int,
        default=None,
        help="Optional: Absolute maximum interval length for histogram display. "
        "Overrides --max_display_percentile if this value is smaller.",
    )

    args = parser.parse_args()

    all_lengths = []
    print(f"Processing {len(args.bed_files)} file(s)...")
    for i, bed_file in enumerate(args.bed_files):
        print(f"Reading intervals from: {bed_file} ({i+1}/{len(args.bed_files)})")
        current_file_lengths_count = 0
        for length in get_interval_lengths(bed_file, args.limit):
            all_lengths.append(length)
            current_file_lengths_count += 1
        print(f"  Read {current_file_lengths_count} intervals from this file.")

    if not all_lengths:
        print("No interval lengths collected from any file. Exiting.")
        return

    print(f"\nTotal intervals collected from all files: {len(all_lengths)}")

    # Convert to numpy array for easier manipulation
    all_lengths_np = np.array(all_lengths)

    if len(all_lengths_np) > 0:
        original_min_len = np.min(all_lengths_np)
        original_max_len = np.max(all_lengths_np)
        original_avg_len = np.mean(all_lengths_np)
        print(f"Overall statistics for all {len(all_lengths_np)} intervals:")
        print(
            f"  Min length: {original_min_len}, Max length: {original_max_len}, Avg length: {original_avg_len:.2f}"
        )
    else:  # Should have been caught by the `if not all_lengths` earlier
        print("No data to process further.")
        return

    # --- Data filtering for histogram display ---
    lengths_for_display = all_lengths_np.copy()
    num_before_min_filter = len(lengths_for_display)

    if args.min_display_length is not None:
        lengths_for_display = lengths_for_display[lengths_for_display >= args.min_display_length]
        filtered_count = num_before_min_filter - len(lengths_for_display)
        if filtered_count > 0:
            print(
                f"Filtered out {filtered_count} intervals shorter than {args.min_display_length} bp for display."
            )

    if not lengths_for_display.size:  # Check if array is empty
        print(
            f"No data remains after applying --min_display_length filter (if any). Cannot generate histogram."
        )
        return

    # Determine the upper bound for display
    current_max_for_display = np.max(lengths_for_display)  # Max of currently filtered data
    effective_upper_bound = current_max_for_display

    percentile_cap = None
    if args.max_display_percentile is not None:
        if 0 < args.max_display_percentile <= 100:
            percentile_cap = np.percentile(lengths_for_display, args.max_display_percentile)
            effective_upper_bound = percentile_cap
            print(
                f"Using {args.max_display_percentile}th percentile ({percentile_cap:.2f} bp) as a potential upper display limit."
            )
        else:
            print(
                f"Warning: --max_display_percentile ({args.max_display_percentile}) is out of range (0-100). Ignoring.",
                file=sys.stderr,
            )

    if args.max_display_absolute is not None:
        if percentile_cap is not None:
            effective_upper_bound = min(percentile_cap, args.max_display_absolute)
        else:
            effective_upper_bound = args.max_display_absolute
        print(f"Applying absolute max display length ({args.max_display_absolute} bp).")

    print(f"Effective upper bound for histogram display: {effective_upper_bound:.2f} bp.")

    num_before_upper_filter = len(lengths_for_display)
    lengths_for_hist = lengths_for_display[lengths_for_display <= effective_upper_bound]
    outliers_excluded_count = num_before_upper_filter - len(lengths_for_hist)

    if outliers_excluded_count > 0:
        print(
            f"Excluded {outliers_excluded_count} intervals longer than {effective_upper_bound:.2f} bp from histogram display."
        )
        # You could add more stats about these outliers if desired
        # outlier_values = lengths_for_display[lengths_for_display > effective_upper_bound]
        # print(f"  (Outlier lengths range from {np.min(outlier_values):.2f} to {np.max(outlier_values):.2f} bp)")

    if not lengths_for_hist.size:
        print(
            "No data remains for histogram after all display filters. Original data range was [{original_min_len}, {original_max_len}]."
        )
        print(
            "Consider adjusting --min_display_length, --max_display_percentile, or --max_display_absolute."
        )
        return

    # --- Plotting ---
    print(f"\nGenerating histogram for {len(lengths_for_hist)} intervals...")
    if len(lengths_for_hist) > 0:
        hist_min_val = np.min(lengths_for_hist)
        hist_max_val = np.max(lengths_for_hist)
        print(
            f"  Displayed interval lengths range from: {hist_min_val:.2f} to {hist_max_val:.2f} bp."
        )

        plt.figure(figsize=(12, 7))  # Slightly larger figure

        # Define bins to span the range of data being plotted
        # If hist_min_val and hist_max_val are the same (e.g. all filtered data points are identical)
        # linspace would fail or produce unhelpful bins. plt.hist handles this better if bins is an int.
        if hist_min_val == hist_max_val:
            # Special handling if all values are the same after filtering
            # Create a small range around the value for binning, or let plt.hist decide
            # For simplicity, we'll let plt.hist with an integer number of bins handle it,
            # or we can define a single bin centered at the value.
            # For now, args.bins (integer) should be okay.
            bin_edges = args.bins
            if (
                args.bins == 1 and hist_min_val > 0
            ):  # if only one bin requested for a single value.
                bin_edges = [hist_min_val - 0.5, hist_min_val + 0.5]
            elif hist_min_val == 0 and hist_max_val == 0 and args.bins > 0:  # all zeros
                bin_edges = np.linspace(-0.5, 0.5, args.bins + 1)
            # else let plt.hist decide with args.bins count
        else:
            bin_edges = np.linspace(hist_min_val, hist_max_val, args.bins + 1)

        plt.hist(lengths_for_hist, bins=bin_edges, color="steelblue", edgecolor="black")

        plot_title = args.title
        plt.title(plot_title)

        current_xlabel = args.xlabel
        is_filtered_display = (
            args.min_display_length is not None
            or (args.max_display_percentile is not None and args.max_display_percentile < 100)
            or args.max_display_absolute is not None
        ) and (
            effective_upper_bound < original_max_len
            or (args.min_display_length or 0) > original_min_len
        )

        if is_filtered_display:
            actual_display_min = hist_min_val
            actual_display_max = hist_max_val
            current_xlabel += (
                f"\n(Displaying range: {actual_display_min:.0f} - {actual_display_max:.0f} bp)"
            )
            if outliers_excluded_count > 0 or (
                num_before_min_filter - len(lengths_for_display) > 0
            ):
                current_xlabel += f"; some intervals excluded)"

        plt.xlabel(current_xlabel)
        plt.ylabel(args.ylabel)

        if args.log_y:
            plt.yscale("log", nonpositive="clip")

        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping

        if args.output:
            try:
                plt.savefig(args.output, dpi=300)
                print(f"Histogram saved to {args.output}")
            except Exception as e:
                print(f"Error saving plot: {e}", file=sys.stderr)
        else:
            print("Displaying histogram...")
            plt.show()
    else:  # Should be caught earlier
        print("No data to plot in histogram after filtering.")


if __name__ == "__main__":
    main()
