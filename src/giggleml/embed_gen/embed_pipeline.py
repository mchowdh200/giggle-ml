import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from os.path import getsize, isfile
from typing import final, overload, override

import numpy as np
import torch
from torch import Tensor

from giggleml.embed_gen.batch_infer import BatchInfer

from ..data_wrangling import fasta
from ..data_wrangling.interval_dataset import IntervalDataset
from ..interval_transformer import IntervalTransformer
from ..interval_transforms import ChunkMax, IntervalTransform
from . import embed_io
from .embed_io import Embed, EmbedMeta
from .embed_model import EmbedModel


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
    ) -> Tensor: ...

    @overload
    def embed(
        self,
        intervals: Sequence[IntervalDataset],
        out: None = None,
        transforms: list[IntervalTransform] | None = None,
    ) -> Tensor: ...

    @abstractmethod
    def embed(
        self,
        intervals: Sequence[IntervalDataset] | IntervalDataset,
        out: Sequence[str] | str | None = None,
        transforms: list[IntervalTransform] | None = None,
    ) -> Tensor | Sequence[Embed] | Embed | Sequence[np.ndarray] | np.ndarray: ...


class DirectPipeline(EmbedPipeline):
    def __init__(
        self,
        embed_model: EmbedModel,
        batch_size: int,
        worker_count: int | None = None,
        sub_workers: int = 0,
    ):
        """
        @param workerCount: should be <= gpu count. None implies torch.accelerator.device_count()
        @param subWorkerCount: corresponds to pytorch::DataLoader::num_worker
        argument -- used to prepare subprocesses for batch construction.
        zero
        """
        self.model: EmbedModel = embed_model
        self.batch_size: int = batch_size

        worker_count = worker_count or torch.accelerator.device_count() or 1
        self.infer: BatchInfer = BatchInfer(
            embed_model, batch_size, worker_count, sub_workers
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
    ) -> Tensor: ...

    @overload
    def embed(
        self,
        intervals: Sequence[IntervalDataset],
        out: None = None,
        transforms: list[IntervalTransform] | None = None,
    ) -> Tensor: ...

    @override
    def embed(
        self,
        intervals: Sequence[IntervalDataset] | IntervalDataset,
        out: Sequence[str] | str | None = None,
        transforms: list[IntervalTransform] | None = None,
    ) -> Tensor | Sequence[Embed] | Embed:
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
            for out_path in out:
                if isfile(out_path):
                    if getsize(out_path) != 0:
                        # TODO:EmbedPipeline does not yet support continuing interrupted jobs.
                        #  Issue:
                        #    There are critical zones that when interrupted leave the files in an ambiguous state.
                        #      1. BatchInfer, when interrupted leaves lots of out.npy.0 files instead of completed items.
                        #      2. There's ambiguity between .npy at worker aggregation step .npy.(.0 .1 .2...) -> .npy
                        #         and dechunking step .npy (len N) -> .npy (len < N)
                        print(
                            f"Output already exists ({out_path}). Continuing interrupted jobs not yet supported.",
                            file=sys.stderr,
                        )

        max_seq_len = self.model.max_seq_len

        if self.model.wants == "sequences":
            for datum in intervals:
                fa_path = datum.associated_fasta_path

                if fa_path is not None:
                    fasta.ensure_fa(fa_path)

        if transforms is None:
            transforms = list()

        if max_seq_len is not None:
            transforms.append(ChunkMax(max_seq_len))

        transformers = [IntervalTransformer(data, transforms) for data in intervals]
        new_data = [transformer.new_dataset for transformer in transformers]
        meta = EmbedMeta(self.model.embed_dim, np.float32, str(self.model))

        if out is None:
            # In-memory mode
            post = [
                _DeChunk(transformer, meta, in_memory=True)
                for transformer in transformers
            ]
            raw_embeddings = self.infer.batch(new_data)
            # Apply backward transform to get final embeddings
            final_embeddings = []

            # manually apply the post call
            for i, raw_embedding in enumerate(raw_embeddings):
                final_embedding = post[i].process(raw_embedding)
                final_embeddings.append(final_embedding)

            if single_input:
                return final_embeddings[0]
            else:
                return torch.stack(final_embeddings)

        # File-based mode (original implementation)
        post = [_DeChunk(transformer, meta) for transformer in transformers]
        self.infer.batch(new_data, out, post)

        # all Embeds were already embed.IO.writeMeta(.) due to DeChunk
        # so we need references and can avoid parsing because their meta is known
        out_embeds = [embed_io.Embed(meta, data_path=path) for path in out]

        if single_input:
            return out_embeds[0]
        return out_embeds


@final
class _DeChunk:
    def __init__(
        self, transformer: IntervalTransformer, meta: EmbedMeta, in_memory: bool = False
    ):
        self.transformer = transformer
        self.meta = meta
        self.in_memory = in_memory

    def _default_aggregator(self, slice: np.ndarray) -> np.ndarray:
        return np.mean(slice, axis=0)

    def _tensor_aggregator(self, slice: Tensor) -> Tensor:
        return torch.mean(slice, dim=0)

    def process(self, item: np.ndarray | Tensor) -> np.ndarray | Tensor | None:
        if self.in_memory:
            # In-memory mode: return the backward transformed array
            if isinstance(item, Tensor):
                return self.transformer.backward_transform(item, self._tensor_aggregator)
            else:
                return self.transformer.backward_transform(item, self._default_aggregator)
        else:
            # File-based mode: perform backward transform and write metadata
            if isinstance(item, Tensor):
                raise ValueError("File-based mode does not support tensor input")
            self.transformer.backward_transform(item, self._default_aggregator)
            assert isinstance(item, np.memmap)
            embed_io.write_meta(item, self.meta)
            return None

    def __call__(self, item: np.memmap) -> None:
        if self.in_memory:
            raise RuntimeError("only callable while not operating in in-memory mode")
        self.process(item)
