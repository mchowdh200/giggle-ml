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

    def __init__(self, chunkMax: int):
        self.maxLen = chunkMax
        self._originIds: Tensor | None
        self._chunkFactor: Tensor | None

    def chunk(self, sequences: Sequence[str]) -> list[str]:
        # INFO: impurities left behind for self.dechunk(...)
        lengths = Tensor([len(x) for x in sequences])
        self._chunkFactor = torch.ceil(lengths / self.maxLen).int()  # divide & ceil

        chunks = list()

        for seq in sequences:
            for i in range(0, len(seq), self.maxLen):
                j = min(len(seq), i + self.maxLen)
                chunks.append(seq[i:j])

        return chunks

    def dechunk(self, src: Tensor) -> Tensor:
        """
        Vectorized; adapts to the src.device
        """

        if self._chunkFactor is None:
            raise RuntimeError(
                "dechunk cannot be called without a prior, corresponding chunk call"
            )

        dev = src.device
        self._chunkFactor.to(dev)
        chunkAmnt = self._chunkFactor.sum(dim=0)

        assert len(src.shape) == 2
        assert src.shape[0] == chunkAmnt

        originIds = torch.arange(
            0, len(self._chunkFactor), device=dev
        ).repeat_interleave(self._chunkFactor, dim=0)

        originalAmnt = len(self._chunkFactor)
        sums = torch.zeros(originalAmnt, *src.shape[1:], dtype=src.dtype, device=dev)
        sums.scatter_add_(0, originIds.unsqueeze(-1).expand_as(src), src)
        return sums / self._chunkFactor.unsqueeze(-1).expand_as(sums)
