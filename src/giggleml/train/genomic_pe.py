import math

import torch
import torch.nn as nn


class SinusoidPE(nn.Module):
    """
    Builds the PE cache in fp64 -- it is safe to down cast after init.

    Standard sinusoidal PE. Conventionally we make the base 10-20x the size of max_len
    so that the fundamental only reaches a fraction of a quarter turn by the end of context.
    That maintains monotonicity and if sufficiently small, lets PE differences in the fundamental
    be approximately linear, a desirable property.

    When the PE dim can vary from d_model, a good choice is ceil(log2(max_len)) because each
    component of the PE tensor acts as a "clock" representing a bit of the input scalar. Rounding
    up to the nearest power of two is also reasonable here.
    """

    def __init__(self, d_pe: int = 32, base: float = 3e9):
        super().__init__()
        self.d_pe = d_pe
        self.base = base

        # We only register the tiny divisor term (32 numbers)
        # This ensures it saves with the model and moves to device automatically
        div_term = torch.exp(
            torch.arange(0, d_pe, 2, dtype=torch.float64) * (-math.log(base) / d_pe)
        )
        self.register_buffer("div_term", div_term)
        self.div_term: torch.Tensor  # assume torch assigned

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: [Batch, ...] LongTensor of absolute genomic coordinates
        """
        # 1. Cast indices to float64 for the calculation
        # Note: We compute on the fly, so we only allocate memory for the *active* batch
        pos = indices.unsqueeze(-1).to(torch.float64)

        # 2. Compute Sin/Cos
        # We assume indices are sparse, so this matrix is small (Batch_Size x 32)
        # This is negligible compute for a GPU
        pe = torch.zeros(
            *indices.shape, self.d_pe, device=indices.device, dtype=torch.float64
        )

        # Broadcast multiplication: [Batch, 1] * [d_model/2]
        term = pos * self.div_term

        pe[..., 0::2] = torch.sin(term)
        pe[..., 1::2] = torch.cos(term)

        # 3. Return in the same dtype as the rest of the model (usually bf16)
        # This adapts automatically to whatever mixed precision context you are in
        return pe.to(
            dtype=self.div_term.dtype
            if self.div_term.dtype != torch.float64  # avoid f64 embeddings getting out
            else torch.bfloat16
        )


class GenomicLocusEncoding(nn.Module):
    """
    Internally initializes intra-chromosomal PEs in fp64 -- requires downcast.

    Uses a standard absolute PE for the intra-chromosomal position.
    Uses a learnable lookup-table for chr indices. This table imposes
    no ordinal assumption on chr index encoding.

    Defaults to a max size per chromosome of 300M (it's more like 249M).
    Thus the base for the intra-chromosomal PE is 3e9, (~10x chr_max).
    Assumes 32 for the 25 chromosome indices (1-22, X, Y, MT).
    """

    def __init__(
        self, coord_dim: int = 32, chr_dim: int = 8, base: float = 3_000_000_000
    ):
        super().__init__()

        # 1. Categorical Encoding for Chromosome (1-22, X, Y, MT -> ~25 classes)
        # We use padding_idx=0 so chr_idx=0 maps to strict zeros if needed.
        # 8 dimensions is plenty to orthogonalize 25 categories.
        self.chr_embed: nn.Embedding = nn.Embedding(
            num_embeddings=32, embedding_dim=chr_dim, padding_idx=0
        )

        # 2. Continuous Fourier Encoding for Intra-Chromosomal Index
        self.coord_embed: SinusoidPE = SinusoidPE(d_pe=coord_dim, base=base)

        self.output_dim = coord_dim + chr_dim

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: [N, 2] LongTensor (chr_idx, coord)
        """
        chr_indices = indices[:, 0]
        coord_indices = indices[:, 1]

        # A. Look up Chromosome ID -> [N, a]
        c_emb = self.chr_embed(chr_indices)

        # B. Look up Coordinate PE -> [N, b > a]
        pos_emb = self.coord_embed(coord_indices)

        # C. Fuse -> [N, a + b]
        return torch.cat([c_emb, pos_emb], dim=-1)
