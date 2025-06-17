import torch
from torch import Tensor

from giggleml.embedGen.chunkDeChunk import ChunkDeChunk


def testChunkDeChunk():
    cdc = ChunkDeChunk(3)
    data = cdc.chunk(["aaaBBBcccD", "bb"])
    assert data == ["aaa", "BBB", "ccc", "D", "bb"]
    embeds = Tensor([2, 2, 4, 4, 20]).repeat(3, 1).t()
    assert torch.all(torch.eq(cdc.dechunk(embeds), Tensor([[3, 3, 3], [20, 20, 20]])))
