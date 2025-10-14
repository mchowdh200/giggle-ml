from abc import ABC
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from giggleml.utils.types import lazy

# INFO: !! Currently all modules assume "embedding vectors" are float32.

# ===================
#    GenomicModel
# ===================


@lazy
class GenomicModel(ABC, nn.Module):
    wants: str  # Type of data this model accepts: "sequences" or "intervals"
    max_seq_len: int | None  # Maximum sequence length the model can handle
    embed_dim: int  # Dimension of the output embeddings
    embed_dtype: torch.dtype = torch.float32  # Data type of the output embeddings

    def collate(self, batch: Sequence[Any]) -> Any:
        """Pre-process the batch of inputs before it reaches the embed call"""
        return batch


# ===================
#    Region2Vec
# ===================


# @final
# class _RawRegion2VecModel(torch.nn.Module):
#     def __init__(self):
#         modelName = "databio/r2v-encode-hg38"
#         self.model = Region2VecExModel(modelName)
#         super().__init__()
#
#     @override
#     def forward(self, x: Any):
#         x = list(zip(*x))
#         x = [Region(*i) for i in x]
#         embed = self.model.encode(x)
#         return torch.tensor(embed)


# @final
# class Region2Vec(EmbedModel):
#     wants = "intervals"
#
#     def __init__(self):
#         self.maxSeqLen: int | None = None
#         self.embedDim: int = 100
#
#     @cached_property
#     def _model(self):
#         model = _RawRegion2VecModel()
#         model.eval()  # probably irrelevant with regard to R2V
#         return model
#
#     @override
#     def to(self, device: Device) -> Self:
#         # CPU only
#         return self
#
#     @override
#     def embed(self, batch: Sequence[GenomicInterval]):
#         return self._model(batch)
