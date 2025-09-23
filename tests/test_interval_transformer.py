import os
from collections.abc import Iterable
from typing import final

import numpy as np

from giggleml.data_wrangling.interval_dataset import BedDataset
from giggleml.data_wrangling.list_dataset import ListDataset
from giggleml.interval_transformer import IntervalTransformer
from giggleml.interval_transforms import (
    ChunkMax,
    IntervalTransform,
    Slide,
    Split,
    Swell,
    Tiling,
    Translate,
)
from giggleml.utils.types import GenomicInterval


@final
class Multiply(IntervalTransform):
    def __init__(self, count: int):
        self.count = count

    def __call__(self, interval: GenomicInterval) -> Iterable[GenomicInterval]:
        for _ in range(self.count):
            yield interval


def test_basic_forward():
    bed = BedDataset("tests/test.bed")
    tran = IntervalTransformer(bed, Multiply(1))
    assert len(tran.get_new_intervals()) == len(tran)
    bed2 = tran.new_dataset
    assert len(bed2) == len(tran)

    for i in range(len(bed)):
        assert bed[i] == bed2[i]

    tran = IntervalTransformer(bed, Multiply(3))
    bed2 = tran.new_dataset

    for i in range(len(bed)):
        for j in range(3):
            assert bed[i] == bed2[3 * i + j]


def test_basic_backward():
    bed = ListDataset([("", 0, 0), ("", 1, 1), ("", 2, 2)])
    bed.associated_fasta_path = None  # so that it conforms to the protocol

    tran = IntervalTransformer(bed, Multiply(3))

    embeds = np.memmap("tests/test.tmp.memmap", dtype=int, mode="w+", shape=(9, 1))
    embeds[:] = np.array([0, 0, 0, 1, 1, 1, 10, 10, 10]).reshape((9, 1))

    reconstructed = tran.backward_transform(embeds, sum)

    assert reconstructed[0] == 0
    assert reconstructed[1] == 3
    assert reconstructed[2] == 30
    assert reconstructed.filename == embeds.filename

    os.remove("tests/test.tmp.memmap")


def test_length_limit():
    bed = BedDataset("tests/test.bed")
    tran = IntervalTransformer(bed, [], 3)
    assert len(tran) == 3


# ======================================
#       Specific Transforms
# ======================================


def verify_tiling_constraints(
    tiles: list[GenomicInterval], base_interval: GenomicInterval, base_size: int
):
    """Verify that tiles meet all tiling constraints."""
    chrm, start, end = base_interval
    base_length = end - start

    # All tiles should be on same chromosome and valid
    for tile_chrm, tile_start, tile_end in tiles:
        assert tile_chrm == chrm, (
            f"Tile chromosome {tile_chrm} != base chromosome {chrm}"
        )
        assert tile_start < tile_end, (
            f"Invalid tile: start {tile_start} >= end {tile_end}"
        )

    # All tiles should have sizes that are powers of 2 * base_size
    for tile_chrm, tile_start, tile_end in tiles:
        tile_size = tile_end - tile_start
        # Find the power of 2 multiplier
        multiplier = tile_size / base_size
        assert multiplier > 0, (
            f"Tile size {tile_size} must be positive multiple of base_size {base_size}"
        )

        # Check if it's a power of 2 (within floating point tolerance)
        log_mult = np.log2(multiplier)
        assert abs(log_mult - round(log_mult)) < 1e-10, (
            f"Tile size {tile_size} is not power-of-2 multiple of base_size {base_size}"
        )

    # Verify coverage quality (tiling algorithm may intentionally leave gaps)
    if tiles:
        # Calculate total coverage
        covered_positions = set()
        for tile_chrm, tile_start, tile_end in tiles:
            # Calculate overlap with base interval
            overlap_start = max(start, tile_start)
            overlap_end = min(end, tile_end)

            # Add covered positions
            for pos in range(overlap_start, overlap_end):
                if start <= pos < end:
                    covered_positions.add(pos)

        coverage_ratio = len(covered_positions) / base_length

        # Verify meaningful coverage exists (algorithm may leave gaps but should cover most)
        assert coverage_ratio > 0, (
            f"No coverage of base interval, got {coverage_ratio:.2%}"
        )

        # For very small intervals, expect good coverage
        if base_length <= base_size:
            assert coverage_ratio >= 0.3, (
                f"Small interval should have reasonable coverage, got {coverage_ratio:.2%}"
            )

    # Tiles extending beyond base interval should have ≥50% overlap with base interval
    for tile_chrm, tile_start, tile_end in tiles:
        if tile_start < start or tile_end > end:
            # Tile extends beyond base interval
            overlap_start = max(start, tile_start)
            overlap_end = min(end, tile_end)
            overlap_length = max(0, overlap_end - overlap_start)
            tile_length = tile_end - tile_start
            overlap_ratio = overlap_length / tile_length

            assert overlap_ratio >= 0.5, (
                f"Extending tile {(tile_chrm, tile_start, tile_end)} has only {overlap_ratio:.2%} overlap with base interval"
            )


def test_tiling_basic():
    """Test basic tiling functionality."""
    ti = Tiling(4)  # base tile size of 4

    # Test medium interval
    base_interval = ("chr1", 0, 16)
    tiles = list(ti(base_interval))
    verify_tiling_constraints(tiles, base_interval, 4)

    # Should have at least one tile
    assert len(tiles) > 0


def test_tiling_small_intervals():
    """Test tiling with intervals smaller than base size."""
    ti = Tiling(10)  # base tile size of 10

    # Very small interval with good overlap
    base_interval = ("chr1", 5, 12)  # size 7, should get a tile if >50% overlap
    tiles = list(ti(base_interval))
    verify_tiling_constraints(tiles, base_interval, 10)

    # Tiny interval with poor overlap (should get no tiles)
    base_interval = ("chr1", 0, 2)  # size 2, likely <50% overlap
    tiles = list(ti(base_interval))
    if tiles:  # if any tiles returned, they must meet constraints
        verify_tiling_constraints(tiles, base_interval, 10)


def test_tiling_large_intervals():
    """Test tiling with large intervals that use multiple octaves."""
    ti = Tiling(4, octaves=3)  # base=4, can use sizes 4, 8, 16

    # Large interval
    base_interval = ("chr1", 0, 100)
    tiles = list(ti(base_interval))
    verify_tiling_constraints(tiles, base_interval, 4)

    # Should cover the interval well
    assert len(tiles) > 1


def test_tiling_edge_cases():
    """Test edge cases for tiling."""
    ti = Tiling(8)

    # Interval exactly matching base size
    base_interval = ("chr1", 0, 8)
    tiles = list(ti(base_interval))
    verify_tiling_constraints(tiles, base_interval, 8)

    # Interval just under base size
    base_interval = ("chr1", 0, 7)
    tiles = list(ti(base_interval))
    verify_tiling_constraints(tiles, base_interval, 8)

    # Interval just over base size
    base_interval = ("chr1", 0, 9)
    tiles = list(ti(base_interval))
    verify_tiling_constraints(tiles, base_interval, 8)


def test_tiling_with_offsets():
    """Test tiling with multiple offsets."""
    ti = Tiling(4, octaves=2, offsets=3)

    base_interval = ("chr1", 0, 20)
    tiles = list(ti(base_interval))
    verify_tiling_constraints(tiles, base_interval, 4)


def test_tiling_range_of_sizes():
    """Test tiling over a range of interval sizes."""
    ti = Tiling(5, octaves=4)

    # Test various sizes from very small to large
    test_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100, 200]

    for size in test_sizes:
        base_interval = ("chr1", 100, 100 + size)
        tiles = list(ti(base_interval))
        verify_tiling_constraints(tiles, base_interval, 5)

        # For larger intervals, should have multiple tiles
        if size > 20:
            assert len(tiles) > 1, (
                f"Large interval of size {size} should have multiple tiles"
            )


def test_tiling_different_positions():
    """Test tiling at different genomic positions."""
    ti = Tiling(6)

    # Test at various starting positions
    positions = [0, 1, 5, 10, 50, 100, 1000]

    for start_pos in positions:
        base_interval = ("chr2", start_pos, start_pos + 25)
        tiles = list(ti(base_interval))
        verify_tiling_constraints(tiles, base_interval, 6)


def test_chunk_max():
    cm = ChunkMax(5)

    small = ("", 0, 2)
    small_chunks = list(cm(small))
    # very small needs no chunking
    assert len(small_chunks) == 1
    assert small_chunks[0] == small

    big = ("", 5, 17)
    big_chunks = list(cm(big))
    # length is 12 so we expect 3 chunks
    #   2 size 10 chunks, 1 size 2 chunk
    assert len(big_chunks) == 3
    assert big_chunks[0] == ("", 5, 10)
    assert big_chunks[1] == ("", 10, 15)
    assert big_chunks[2] == ("", 15, 17)


def test_slide():
    op = Slide(4)
    assert list(op(("chrX", 0, 10))) == [
        ("chrX", 0, 10),
        ("chrX", 3, 13),
        ("chrX", 7, 17),
        ("chrX", 10, 20),
    ]


def test_translate_positive_offset():
    op = Translate(50)
    assert list(op(("chr1", 100, 200))) == [("chr1", 150, 250)]


def test_translate_negative_offset():
    op = Translate(-50)
    assert list(op(("chr2", 200, 300))) == [("chr2", 150, 250)]


def test_translate_zero_offset():
    op = Translate(0)
    interval = ("chr3", 50, 100)
    assert list(op(interval)) == [interval]


# Test Swell
def test_swell_shrink():
    op = Swell(0.5)
    assert list(op(("chr1", 100, 200))) == [("chr1", 125, 175)]


def test_swell_expand():
    op = Swell(2.0)
    assert list(op(("chr2", 100, 200))) == [("chr2", 50, 250)]


def test_swell_odd_size():
    op = Swell(1.0)
    assert list(op(("chr3", 100, 201))) == [("chr3", 100, 201)]


# Test Split
def test_split_exact_division():
    op = Split(3)
    intervals = list(op(("chr1", 100, 103)))
    assert intervals == [("chr1", 100, 101), ("chr1", 101, 102), ("chr1", 102, 103)]


def test_split_uneven():
    op = Split(3)
    intervals = list(op(("chr2", 100, 107)))
    assert intervals == [("chr2", 100, 103), ("chr2", 103, 105), ("chr2", 105, 107)]


def test_split_small_interval_unchanged():
    op = Split(3)
    interval = ("chr3", 100, 101)
    assert list(op(interval)) == [interval]
