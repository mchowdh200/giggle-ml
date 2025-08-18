import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from os.path import getsize, isfile
from typing import final, overload, override

import numpy as np
import torch

from giggleml.embedGen.batchInfer import BatchInfer

from ..dataWrangling import fasta
from ..dataWrangling.intervalDataset import IntervalDataset
from ..intervalTransformer import IntervalTransformer
from ..intervalTransforms import ChunkMax, IntervalTransform
from . import embedIO
from .embedIO import Embed, EmbedMeta
from .embedModel import EmbedModel


class EmbedPipeline(ABC):
    @overload
    def embed(
        self,
        intervals: IntervalDataset,
        out: str,
        transforms: list[IntervalTransform] | None = None,
    ) -> Embed: ...

    @overload
    def embed(
        self,
        intervals: Sequence[IntervalDataset],
        out: Sequence[str],
        transforms: list[IntervalTransform] | None = None,
    ) -> Sequence[Embed]: ...

    @abstractmethod
    def embed(
        self,
        intervals: Sequence[IntervalDataset] | IntervalDataset,
        out: Sequence[str] | str,
        transforms: list[IntervalTransform] | None = None,
    ) -> Sequence[Embed] | Embed: ...


class DirectPipeline(EmbedPipeline):
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

        workerCount = workerCount or torch.accelerator.device_count() or 1
        self.infer: BatchInfer = BatchInfer(
            embedModel, batchSize, workerCount, subWorkers
        )

    @overload
    def embed(
        self,
        intervals: IntervalDataset,
        out: str,
        transforms: list[IntervalTransform] | None = None,
    ) -> Embed: ...

    @overload
    def embed(
        self,
        intervals: Sequence[IntervalDataset],
        out: Sequence[str],
        transforms: list[IntervalTransform] | None = None,
    ) -> Sequence[Embed]: ...

    @override
    def embed(
        self,
        intervals: Sequence[IntervalDataset] | IntervalDataset,
        out: Sequence[str] | str,
        transforms: list[IntervalTransform] | None = None,
    ) -> Sequence[Embed] | Embed:
        if isinstance(intervals, Sequence) == isinstance(out, str):
            raise ValueError(
                "Expecting either both or neither of data & out to be sequences"
            )
        if not isinstance(intervals, Sequence):
            intervals = [intervals]
        if isinstance(out, str):
            out = [out]

        for outPath in out:
            if isfile(outPath):
                if getsize(outPath) != 0:
                    # TODO:EmbedPipeline does not yet support continuing interrupted jobs.
                    #  Issue:
                    #    There are critical zones that when interrupted leave the files in an ambiguous state.
                    #      1. BatchInfer, when interrupted leaves lots of out.npy.0 files instead of completed items.
                    #      2. There's ambiguity between .npy at worker aggregation step .npy.(.0 .1 .2...) -> .npy
                    #         and dechunking step .npy (len N) -> .npy (len < N)
                    print(
                        f"Output already exists ({outPath}). Continuing interrupted jobs not yet supported.",
                        file=sys.stderr,
                    )

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
        post = [_DeChunk(transformer, meta) for transformer in transformers]
        self.infer.batch(newData, out, post)

        # all Embeds were already embed.IO.writeMeta(.) due to DeChunk
        # so we need references and can avoid parsing because their meta is known
        outEmbeds = [embedIO.Embed(path, meta) for path in out]

        if len(out) == 1:
            return outEmbeds[0]
        return outEmbeds


@final
class _DeChunk:
    def __init__(self, transformer: IntervalTransformer, meta: EmbedMeta):
        self.transformer = transformer
        self.meta = meta

    def _defaultAggregator(self, slice: np.ndarray) -> np.ndarray:
        return np.mean(slice, axis=0)

    def __call__(self, item: np.memmap):
        self.transformer.backwardTransform(item, self._defaultAggregator)
        embedIO.writeMeta(item, self.meta)
