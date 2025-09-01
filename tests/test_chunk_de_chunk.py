import torch
from torch import Tensor

from giggleml.embed_gen.chunk_de_chunk import ChunkDeChunk


def test_chunk_de_chunk():
    cdc = ChunkDeChunk(3)
    data = cdc.chunk(["aaaBBBcccD", "bb"])
    assert data == ["aaa", "BBB", "ccc", "D", "bb"]
    embeds = Tensor([2, 2, 4, 4, 20]).repeat(3, 1).t()
    assert torch.all(torch.eq(cdc.dechunk(embeds), Tensor([[3, 3, 3], [20, 20, 20]])))
