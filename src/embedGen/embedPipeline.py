import os
from collections.abc import Iterable, Sequence
from os.path import exists, isfile
from re import sub
from typing import final, overload

import numpy as np
import pyfastx
from dataWrangling import fasta
from dataWrangling.intervalDataset import IntervalDataset
from embedGen import embedIO
from embedGen.embedIO import Embed, EmbedMeta
from embedModel import EmbedModel
from intervalTransformer import ChunkMax, IntervalTransformer

from .gpuMaster import GpuMaster


@final
class EmbedPipeline:
    def __init__(
        self, embedModel: EmbedModel, batchSize: int, workerCount: int, subWorkerCount: int = 0
    ):
        """
        @param workerCount: should be <= gpu count
        @param subWorkerCount: corresponds to pytorch::DataLoader::num_worker
        argument -- used to prepare subprocesses for batch construction.
        zero
        """
        self.model: EmbedModel = embedModel
        self.batchSize: int = batchSize
        self.gpuMaster = GpuMaster(embedModel, batchSize, workerCount, subWorkerCount)

    @overload
    def embed(self, intervals: IntervalDataset, out: str) -> Embed: ...

    @overload
    def embed(
        self, intervals: Sequence[IntervalDataset], out: Sequence[str]
    ) -> Iterable[Embed]: ...

    def embed(
        self,
        intervals: Sequence[IntervalDataset] | IntervalDataset,
        out: Sequence[str] | str,
    ) -> Embed | Iterable[Embed]:
        if isinstance(intervals, IntervalDataset) != isinstance(out, str):
            raise ValueError("Expecting either both or neither of data & out to be sequences")
        if isinstance(intervals, IntervalDataset):
            intervals = [intervals]
        expectSingle = isinstance(out, str)  # simple input; simple output
        if isinstance(out, str):
            out = [out]

        # filtering for outputs that do not yet exist

        taskCount = len(out)
        intervals, out = zip(
            *[(item, out) for item, out in zip(intervals, out) if not isfile(out)]
        )

        if len(out) != taskCount:
            print(f"Skipping {taskCount - len(out)} already complete outputs")
        if len(out) == 0:
            return list()

        del taskCount  # we won't be using this again
        maxSeqLen = self.model.maxSeqLen

        if maxSeqLen is None:
            # as intervals cannot exceed model input size, they are not chunked,
            # so processing is simple.
            mmaps = self.gpuMaster.batch(intervals, out)
        else:
            if self.model.wants == "sequences":
                for datum in intervals:
                    faPath = datum.associatedFastaPath

                    if faPath is not None:
                        fasta.ensureFa(faPath)

            transformers = [IntervalTransformer(item, ChunkMax(maxSeqLen)) for item in intervals]
            chunked = [transformer.newDataset for transformer in transformers]
            results = self.gpuMaster.batch(chunked, out)
            # dechunked
            mmaps: Iterable[MmapF32] = [
                transformer.backwardTransform(result, self._defaultAggregator)
                for transformer, result in zip(transformers, results)
            ]

        meta = EmbedMeta(self.model.embedDim, np.float32)
        embeds = [embedIO.writeMeta(mmap, meta) for mmap in mmaps]

        if expectSingle:
            return embeds[0]
        return embeds

    def _defaultAggregator(self, slice: np.ndarray) -> np.ndarray:
        return np.mean(slice, axis=0)
