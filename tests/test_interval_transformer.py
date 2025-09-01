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


def test_tiling():
    ti = Tiling(3)
    assert list(ti(("", 0, 10))) == [("", 0, 3), ("", 3, 6), ("", 6, 9), ("", 9, 12)]
    assert np.all(list(ti.weights([("", 0, 4)])) == np.array([0.75, 0.25]))
    ti = Tiling(3, 3)
    assert list(ti.tile(("", 5, 18))) == [[1], [1, 2], []]
    assert list(ti.tile(("", 7, 19))) == [[6], [1, 2], []]
    assert list(ti.tile(("", 7, 39))) == [[12], [1], [1, 2]]
    assert list(ti.tile(("", 6, 12))) == [[], [1], []]
    ti = Tiling(340, 5)
    assert list(ti.tile(("", 6800, 8160))) == [[], [], [5], [], []]


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
