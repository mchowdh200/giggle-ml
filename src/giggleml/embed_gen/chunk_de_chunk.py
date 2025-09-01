from collections.abc import Sequence
from typing import final

import torch
from torch import Tensor


# TODO: put the primary inference system on ChunkDeChunk
@final
class ChunkDeChunk:
    """
    Chunks strings up to a maxLen. Provides mechanism to "dechunk" (mean aggregate) a series
    of Tensor elements associated with the same original string. Holds impure values to correctly
    map the chunks in the dechunk phase back to original items. dechunk is fully vectorized
    and designed for GPU usage.
    """

    def __init__(self, chunk_max: int):
        self.max_len = chunk_max
        self._origin_ids: Tensor | None
        self._chunk_factor: Tensor | None

    def chunk(self, sequences: Sequence[str]) -> list[str]:
        # INFO: impurities left behind for self.dechunk(...)
        lengths = Tensor([len(x) for x in sequences])
        self._chunk_factor = torch.ceil(lengths / self.max_len).int()  # divide & ceil

        chunks = list()

        for seq in sequences:
            for i in range(0, len(seq), self.max_len):
                j = min(len(seq), i + self.max_len)
                chunks.append(seq[i:j])

        return chunks

    def dechunk(self, src: Tensor) -> Tensor:
        """
        Vectorized; adapts to the src.device
        """

        if self._chunk_factor is None:
            raise RuntimeError(
                "dechunk cannot be called without a prior, corresponding chunk call"
            )

        dev = src.device
        self._chunk_factor = self._chunk_factor.to(dev)
        chunk_amnt = self._chunk_factor.sum(dim=0)

        assert len(src.shape) == 2
        assert src.shape[0] == chunk_amnt

        original_amnt = len(self._chunk_factor)
        origin_ids = torch.arange(0, original_amnt, device=dev).repeat_interleave(
            self._chunk_factor, dim=0
        )

        sums = torch.zeros(original_amnt, *src.shape[1:], dtype=src.dtype, device=dev)
        sums.scatter_add_(0, origin_ids.unsqueeze(-1).expand_as(src), src)
        return sums / self._chunk_factor.unsqueeze(-1).expand_as(sums)
