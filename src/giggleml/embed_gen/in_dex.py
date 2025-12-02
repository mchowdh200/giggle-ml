from collections.abc import Iterable, Iterator
from typing import final, override

import torch
from torch.utils.data import default_collate

from giggleml.embed_gen.dex import (
    CollateFn,
    ConsumerFn,
    DecollateFn,
    DefaultPostprocessFn,
    Dex,
    PostprocessorFn,
    PreprocessorFn,
)
from giggleml.utils.nothing import yield_from, yield_through


@final
class _Pre:
    def __init__(self, base: PreprocessorFn):
        self.base = base

    def __call__(self, item):
        idx, value = item
        head, *tail = list(self.base(value))
        yield idx, head
        yield from [(None, x) for x in tail]


@final
class _Collate:
    def __init__(self, base: CollateFn):
        self.base = base

    def __call__(self, batch):
        indices, items = zip(*batch)
        return indices, self.base(list(items))


@final
class _Module(torch.nn.Module):
    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.base = base

    @override
    def forward(self, batch):
        indices, default_batch = batch
        return indices, self.base(default_batch)


@final
class _Decollate:
    def __init__(self, base: DecollateFn) -> None:
        self.base = base

    def __call__(self, output):
        indices, base_output = output
        return zip(indices, self.base(base_output))


@final
class _Post:
    def __init__(self, base: PostprocessorFn) -> None:
        self.base = base

    def __call__(self, outputs):
        indices, base_outputs = zip(*outputs)
        return indices[0], self.base(base_outputs)


class _Enumerate[T]:
    def __init__(self, data: Iterable[T]):
        self.data: Iterable[T] = data

    def __iter__(self) -> Iterator[tuple[int, T]]:
        yield from enumerate(self.data)


class InDex[T_in, U_pre, V_post, W_out, Batch_in, Batch_out](Dex):
    """
    Maintains indices parallel to items through the pipeline for post-reordering.
    Necessary in lieu to a separate fast indices pass because torch's DataLoader
    with num_workers permutes order in a non-deterministic way. It is possible
    to use the faster strategy with num_workers=0, but for the general case,
    a custom multithreaded DataLoader would be necessary.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        preprocessor_fn: PreprocessorFn[T_in, U_pre] = yield_through,
        postprocessor_fn: PostprocessorFn[V_post, W_out] = DefaultPostprocessFn,
        collate_fn: CollateFn[U_pre, Batch_in] = default_collate,
        decollate_fn: DecollateFn[Batch_out, V_post] = yield_from,
    ):
        """Initialize Dex with model and pipeline functions."""
        self.module: torch.nn.Module = module
        self.preprocessor_fn = preprocessor_fn
        self.postprocessor_fn = postprocessor_fn
        self.collate_fn = collate_fn
        self.decollate_fn = decollate_fn

        super().__init__(
            module=_Module(self.module),
            preprocessor_fn=_Pre(self.preprocessor_fn),
            postprocessor_fn=_Post(self.postprocessor_fn),
            collate_fn=_Collate(self.collate_fn),
            decollate_fn=_Decollate(self.decollate_fn),
        )

    @override
    def execute(
        self,
        data: Iterable[T_in],
        consumer_fn: ConsumerFn[tuple[int, W_out]],
        batch_size: int,
        num_workers: int = 0,
        auto_reclaim: bool = True,
    ):
        indexed_data = _Enumerate(data)

        super().execute(
            indexed_data,
            consumer_fn,
            batch_size,
            num_workers,
            auto_reclaim,
        )
