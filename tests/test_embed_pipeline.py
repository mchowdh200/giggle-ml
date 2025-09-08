import contextlib
import os

import numpy as np
import torch

from giggleml.data_wrangling.interval_dataset import BedDataset
from giggleml.embed_gen.embed_model import CountACGT
from giggleml.embed_gen.embed_pipeline import DirectPipeline


def test_pipeline():
    with contextlib.suppress(FileNotFoundError):
        os.remove("tests/test_out.tmp.npy")
        os.remove("tests/test_out.tmp.npy.meta")

    assert CountACGT(10).max_seq_len == 10  # to keep tests consistent

    # TODO: needs way more testing examples

    # if not cuda.is_available(), workerCount is CPU count
    worker_count = min(2, torch.cuda.device_count()) if torch.cuda.is_available() else 2

    pipeline = DirectPipeline(
        embed_model=CountACGT(), batch_size=2, worker_count=worker_count
    )
    bed = BedDataset("tests/test.bed", "tests/test.fa")
    embed = pipeline.embed(bed, "tests/test_out.tmp.npy")

    assert len(embed.data) == len(bed)
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

    for result, expect in zip(embed.data, expecting):
        assert np.array_equal(result, np.array(expect))

    embed.delete()


def test_pipeline_in_memory():
    assert CountACGT(10).max_seq_len == 10  # to keep tests consistent

    # if not cuda.is_available(), workerCount is CPU count
    worker_count = min(2, torch.cuda.device_count()) if torch.cuda.is_available() else 2

    pipeline = DirectPipeline(
        embed_model=CountACGT(), batch_size=2, worker_count=worker_count
    )
    bed = BedDataset("tests/test.bed", "tests/test.fa")
    embed = pipeline.embed(bed)

    assert len(embed) == len(bed)
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

    for result, expect in zip(embed, expecting):
        assert np.array_equal(result, np.array(expect))
