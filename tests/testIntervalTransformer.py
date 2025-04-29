import os
from collections.abc import Iterable
from typing import final

import numpy as np

from dataWrangling.intervalDataset import BedDataset
from dataWrangling.listDataset import ListDataset
from intervalTransformer import ChunkMax, IntervalTransformer
from utils.types import GenomicInterval


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
    bed2 = tran.newDataset

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


########################################
#     Specific Transforms
########################################


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
