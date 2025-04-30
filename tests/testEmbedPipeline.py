import contextlib
import os

import numpy as np

from giggleml.dataWrangling.intervalDataset import BedDataset
from giggleml.embedGen.embedModel import CountACGT
from giggleml.embedGen.embedPipeline import EmbedPipeline


def testPipeline():
    with contextlib.suppress(FileNotFoundError):
        os.remove("tests/test_out.tmp.npy")

    assert CountACGT().maxSeqLen == 10  # to keep tests consistent

    pipeline = EmbedPipeline(embedModel=CountACGT(), batchSize=2, workerCount=2)
    bed = BedDataset("tests/test.bed", "tests/test.fa")
    results = pipeline.embed(bed, "tests/test_out.tmp.npy").data

    assert len(results) == len(bed)
    # the second item 0-40 should have been split into
    #   [0-10), [10-20), [20-30), [30-40)
    #   that's
    #     ACGTAGCTTA GACTACGCAG CATATAGCGC GCTAGCTACC
    #   with ACGT counts
    #     3223       3331       3322       2422
    #   so after mean aggregation (perhaps not ideal for a counter)
    #     2.75  3  2.25  2
    expecting = [
        [1, 0, 0, 0],
        [2.75, 3, 2.25, 2],
        [3.0, 2.0, 2.0, 3.0],
        [3.0, 3.0, 3.0, 1.0],
        [3.0, 3.0, 2.0, 2.0],
        [3.0, 3.0, 2.5, 1.5],
        [3.0, 2.0, 2.5, 2.5],
        [5.0, 2.0, 1.0, 2.0],
        [3.0, 0.0, 0.0, 2.0],
        [5.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 5.0],
    ]

    for result, expect in zip(results, expecting):
        assert np.array_equal(result, np.array(expect))

    with contextlib.suppress(FileNotFoundError):
        os.remove("tests/test_out.tmp.npy")
        os.remove("tests/test_out.tmp.npy.meta")
