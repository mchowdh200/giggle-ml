import os
from collections.abc import Iterable
from typing import final

import numpy as np

from giggleml.dataWrangling.intervalDataset import BedDataset
from giggleml.dataWrangling.listDataset import ListDataset
from giggleml.intervalTransformer import (
    ChunkMax,
    IntervalTransformer,
    Slide,
    Split,
    Swell,
    Translate,
)
from giggleml.utils.types import GenomicInterval


@final
class Multiply:
    def __init__(self, count: int):
        self.count = count

    def __call__(self, interval: GenomicInterval) -> Iterable[GenomicInterval]:
        for _ in range(self.count):
            yield interval


def testBasicForward():
    bed = BedDataset("tests/test.bed")
    tran = IntervalTransformer(bed, Multiply(1))
    assert len(tran.getNewIntervals()) == len(tran)
    bed2 = tran.newDataset
    assert len(bed2) == len(tran)

    for i in range(len(bed)):
        assert bed[i] == bed2[i]

    tran = IntervalTransformer(bed, Multiply(3))
    bed2 = tran.newDataset

    for i in range(len(bed)):
        for j in range(3):
            assert bed[i] == bed2[3 * i + j]


def testBasicBackward():
    bed = ListDataset([("", 0, 0), ("", 1, 1), ("", 2, 2)])
    bed.associatedFastaPath = None  # so that it conforms to the protocol

    tran = IntervalTransformer(bed, Multiply(3))

    embeds = np.memmap("tests/test.tmp.memmap", dtype=int, mode="w+", shape=(9, 1))
    embeds[:] = np.array([0, 0, 0, 1, 1, 1, 10, 10, 10]).reshape((9, 1))

    reconstructed = tran.backwardTransform(embeds, sum)

    assert reconstructed[0] == 0
    assert reconstructed[1] == 3
    assert reconstructed[2] == 30
    assert reconstructed.filename == embeds.filename

    os.remove("tests/test.tmp.memmap")


def testLengthLimit():
    bed = BedDataset("tests/test.bed")
    tran = IntervalTransformer(bed, [], 3)
    assert len(tran) == 3


# ======================================
#       Specific Transforms
# ======================================


def testChunkMax():
    cm = ChunkMax(5)

    small = ("", 0, 2)
    smallChunks = list(cm(small))
    # very small needs no chunking
    assert len(smallChunks) == 1
    assert smallChunks[0] == small

    big = ("", 5, 17)
    bigChunks = list(cm(big))
    # length is 12 so we expect 3 chunks
    #   2 size 10 chunks, 1 size 2 chunk
    assert len(bigChunks) == 3
    assert bigChunks[0] == ("", 5, 10)
    assert bigChunks[1] == ("", 10, 15)
    assert bigChunks[2] == ("", 15, 17)


def testSlide():
    op = Slide(4)
    assert list(op(("chrX", 0, 10))) == [
        ("chrX", 0, 10),
        ("chrX", 3, 13),
        ("chrX", 7, 17),
        ("chrX", 10, 20),
    ]


def testTranslatePositiveOffset():
    op = Translate(50)
    assert list(op(("chr1", 100, 200))) == [("chr1", 150, 250)]


def testTranslateNegativeOffset():
    op = Translate(-50)
    assert list(op(("chr2", 200, 300))) == [("chr2", 150, 250)]


def testTranslateZeroOffset():
    op = Translate(0)
    interval = ("chr3", 50, 100)
    assert list(op(interval)) == [interval]


# Test Swell
def testSwellShrink():
    op = Swell(0.5)
    assert list(op(("chr1", 100, 200))) == [("chr1", 125, 175)]


def testSwellExpand():
    op = Swell(2.0)
    assert list(op(("chr2", 100, 200))) == [("chr2", 50, 250)]


def testSwellOddSize():
    op = Swell(1.0)
    assert list(op(("chr3", 100, 201))) == [("chr3", 100, 201)]


# Test Split
def testSplitExactDivision():
    op = Split(3)
    intervals = list(op(("chr1", 100, 103)))
    assert intervals == [("chr1", 100, 101), ("chr1", 101, 102), ("chr1", 102, 103)]


def testSplitUneven():
    op = Split(3)
    intervals = list(op(("chr2", 100, 107)))
    assert intervals == [("chr2", 100, 103), ("chr2", 103, 105), ("chr2", 105, 107)]


def testSplitSmallIntervalUnchanged():
    op = Split(3)
    interval = ("chr3", 100, 101)
    assert list(op(interval)) == [interval]
