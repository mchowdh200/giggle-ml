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

    @overload
    def embed(
        self,
        intervals: IntervalDataset,
        out: None = None,
        transforms: list[IntervalTransform] | None = None,
    ) -> Embed: ...

    @overload
    def embed(
        self,
        intervals: Sequence[IntervalDataset],
        out: None = None,
        transforms: list[IntervalTransform] | None = None,
    ) -> Sequence[Embed]: ...

    @abstractmethod
    def embed(
        self,
        intervals: Sequence[IntervalDataset] | IntervalDataset,
        out: Sequence[str] | str | None = None,
        transforms: list[IntervalTransform] | None = None,
    ) -> Sequence[Embed] | Embed | Sequence[np.ndarray] | np.ndarray: ...


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

    @overload
    def embed(
        self,
        intervals: IntervalDataset,
        out: None = None,
        transforms: list[IntervalTransform] | None = None,
    ) -> Embed: ...

    @overload
    def embed(
        self,
        intervals: Sequence[IntervalDataset],
        out: None = None,
        transforms: list[IntervalTransform] | None = None,
    ) -> Sequence[Embed]: ...

    @override
    def embed(
        self,
        intervals: Sequence[IntervalDataset] | IntervalDataset,
        out: Sequence[str] | str | None = None,
        transforms: list[IntervalTransform] | None = None,
    ) -> Sequence[Embed] | Embed:
        # Handle input validation for file-based mode
        if out is not None:
            if isinstance(intervals, Sequence) == isinstance(out, str):
                raise ValueError(
                    "Expecting either both or neither of data & out to be sequences"
                )

        single_input = not isinstance(intervals, Sequence)

        if not isinstance(intervals, Sequence):
            intervals = [intervals]
        if isinstance(out, str):
            out = [out]

        # File existence check only for file-based mode
        if out is not None:
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

        if out is None:
            # In-memory mode
            post = [
                _DeChunk(transformer, meta, in_memory=True)
                for transformer in transformers
            ]
            raw_embeddings = self.infer.batch(newData)
            # Apply backward transform to get final embeddings
            final_embeddings = []

            # manually apply the post call
            for i, raw_embedding in enumerate(raw_embeddings):
                final_embedding = post[i].process(raw_embedding)
                final_embeddings.append(final_embedding)

            outEmbeds = [embedIO.Embed(meta, data=datum) for datum in final_embeddings]
        else:
            # File-based mode (original implementation)
            post = [_DeChunk(transformer, meta) for transformer in transformers]
            self.infer.batch(newData, out, post)

            # all Embeds were already embed.IO.writeMeta(.) due to DeChunk
            # so we need references and can avoid parsing because their meta is known
            outEmbeds = [embedIO.Embed(meta, dataPath=path) for path in out]

        if single_input:
            return outEmbeds[0]
        return outEmbeds


@final
class _DeChunk:
    def __init__(
        self, transformer: IntervalTransformer, meta: EmbedMeta, in_memory: bool = False
    ):
        self.transformer = transformer
        self.meta = meta
        self.in_memory = in_memory

    def _defaultAggregator(self, slice: np.ndarray) -> np.ndarray:
        return np.mean(slice, axis=0)

    def process(self, item: np.memmap | np.ndarray) -> np.ndarray | None:
        if self.in_memory:
            # In-memory mode: return the backward transformed array
            return self.transformer.backwardTransform(item, self._defaultAggregator)
        else:
            # File-based mode: perform backward transform and write metadata
            self.transformer.backwardTransform(item, self._defaultAggregator)
            assert isinstance(item, np.memmap)
            embedIO.writeMeta(item, self.meta)
            return None

    def __call__(self, item: np.memmap) -> None:
        if self.in_memory:
            raise RuntimeError("only callable while not operating in in-memory mode")
        self.process(item)
