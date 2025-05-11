import sys
from collections.abc import Sequence
from os.path import isfile
from typing import final, overload

import numpy as np
import torch

from ..dataWrangling import fasta
from ..dataWrangling.intervalDataset import IntervalDataset
from ..intervalTransformer import ChunkMax, IntervalTransform, IntervalTransformer
from . import embedIO
from .embedIO import Embed, EmbedMeta
from .embedModel import EmbedModel
from .gpuMaster import GpuMaster


@final
class EmbedPipeline:
    def __init__(
        self,
        embedModel: EmbedModel,
        batchSize: int,
        workerCount: int | None = None,
        subWorkers: int = 0,
    ):
        """
        @param workerCount: should be <= gpu count. None implies torch.accelerator.device_count()
        @param subWorkerCount: corresponds to pytorch::DataLoader::num_worker
        argument -- used to prepare subprocesses for batch construction.
        zero
        """
        self.model: EmbedModel = embedModel
        self.batchSize: int = batchSize
        workerCount = workerCount or torch.accelerator.device_count()
        self.gpuMaster = GpuMaster(embedModel, batchSize, workerCount, subWorkers)

    @overload
    def embed(
        self,
        intervals: IntervalDataset,
        out: str,
        transforms: list[IntervalTransform] | None = None,
    ) -> Embed: ...

    @overload
    # this returns None because for large jobs, the OS may not be able to handle
    # the amount of memmaps we would be returning.
    def embed(
        self,
        intervals: Sequence[IntervalDataset],
        out: Sequence[str],
        transforms: list[IntervalTransform] | None = None,
    ) -> None: ...

    def embed(
        self,
        intervals: Sequence[IntervalDataset] | IntervalDataset,
        out: Sequence[str] | str,
        transforms: list[IntervalTransform] | None = None,
    ) -> Embed | None:
        if isinstance(intervals, IntervalDataset) != isinstance(out, str):
            raise ValueError("Expecting either both or neither of data & out to be sequences")
        if isinstance(intervals, IntervalDataset):
            intervals = [intervals]
        if isinstance(out, str):
            out = [out]

        for outPath in out:
            if isfile(outPath):
                # TODO:EmbedPipeline does not yet support continuing interrupted jobs.
                #  Issue:
                #    There are critical zones that when interrupted leave the files in an ambiguous state.
                #      1. GpuMaster when interrupted leaves lots of out.npy.0 files instead of completed items.
                #      2. There's ambiguity between .npy at worker aggregation step .npy.(.0 .1 .2...) -> .npy
                #         and dechunking step .npy (len N) -> .npy (len < N)
                print(f"Output already exists ({outPath})", file=sys.stderr)

        maxSeqLen = self.model.maxSeqLen
        eDim = self.model.embedDim

        if self.model.wants == "sequences":
            for datum in intervals:
                faPath = datum.associatedFastaPath

                if faPath is not None:
                    fasta.ensureFa(faPath)

        if transforms is None:
            transforms = list()

        if maxSeqLen is not None:
            transforms.append(ChunkMax(maxSeqLen))

        transformers = [IntervalTransformer(data, transforms) for data in intervals]
        newData = [transformer.newDataset for transformer in transformers]
        meta = EmbedMeta(self.model.embedDim, np.float32, str(self.model))
        post = [DeChunk(transformer, meta) for transformer in transformers]
        self.gpuMaster.batch(newData, out, post)

        # only the overload for single input, output returns a value
        if len(out) == 1:
            size = len(intervals[0])
            data = np.memmap(out[0], np.float32, "r", shape=(size, eDim))
            return embedIO.Embed(data, meta)


@final
class DeChunk:
    def __init__(self, transformer: IntervalTransformer, meta: EmbedMeta):
        self.transformer = transformer
        self.meta = meta

    def _defaultAggregator(self, slice: np.ndarray) -> np.ndarray:
        return np.mean(slice, axis=0)

    def __call__(self, item: np.memmap):
        self.transformer.backwardTransform(item, self._defaultAggregator)
        embedIO.writeMeta(item, self.meta)
