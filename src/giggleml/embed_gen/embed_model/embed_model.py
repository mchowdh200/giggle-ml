from abc import ABC
from collections.abc import Sequence
from typing import Any, ClassVar

import torch
from torch import nn

from giggleml.utils.types import lazy

# INFO: !! Currently all modules assume "embedding vectors" are float32.

# ===================
#    EmbedModel
# ===================


@lazy
class EmbedModel(ABC, nn.Module):
    wants: ClassVar[str]  # Type of data this model accepts: "sequences" or "intervals"
    max_seq_len: ClassVar[int | None]  # Maximum sequence length the model can handle
    embed_dim: ClassVar[int]  # Dimension of the output embeddings
    embed_dtype: ClassVar[torch.dtype] = (
        torch.float32
    )  # Data type of the output embeddings

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
