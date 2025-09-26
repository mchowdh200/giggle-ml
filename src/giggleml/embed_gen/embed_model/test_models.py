from collections.abc import Sequence
from typing import cast, final

import torch
from typing_extensions import override

from giggleml.embed_gen.embed_model import EmbedModel
from giggleml.utils.types import GenomicInterval


@final
class CountACGT(EmbedModel):
    """
    Developed for testing purposes. Has an arbitrary max sequence length of 10.
    Creates a 4 dimensional embedding vector for counts of ACGT
    respectively. Embedding conforms to f32 tensor.
    """

    wants = "sequences"

    def __init__(self, max_seq_len: int = 10):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = 4

    @override
    def collate(self, batch: Sequence[str]) -> Sequence[str]:
        for item in batch:
            if self.max_seq_len is not None and len(item) > self.max_seq_len:
                raise ValueError("Sequence exceeds max length; refusing to truncate.")
        return batch

    @override
    def forward(self, batch: Sequence[str]) -> torch.FloatTensor:
        results = list()

        for item in batch:
            counts = [0, 0, 0, 0]

            for char in item:
                if char == "A":
                    counts[0] += 1
                elif char == "C":
                    counts[1] += 1
                elif char == "G":
                    counts[2] += 1
                elif char == "T":
                    counts[3] += 1
            results.append(counts)
        return torch.FloatTensor(results)

    @override
    def __repr__(self):
        return f"CountACGT({self.max_seq_len})"


@final
class TrivialModel(EmbedModel):
    wants = "intervals"

    def __init__(self, max_seq_len: int = 10):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = 1

    @override
    def collate(self, batch: Sequence[GenomicInterval]) -> Sequence[GenomicInterval]:
        for item in batch:
            if self.max_seq_len is not None and len(item) > self.max_seq_len:
                raise ValueError("Sequence exceeds max length; refusing to truncate.")
        return batch

    @override
    def forward(self, batch: Sequence[GenomicInterval]) -> torch.FloatTensor:
        results = list()

        for item in batch:
            _, start, end = item
            results.append(end - start)

        return cast(torch.FloatTensor, torch.FloatTensor(results).unsqueeze(-1))

    @override
    def __repr__(self):
        return f"TrivialModel({self.max_seq_len})"
