"""
C Model, deep sets architecture, hyenaDNA pre-processing
if giggle is a giggle, this is a chuckle

    (interval set -> sequence set --)-> sequence embeddings -> [C Model] -> (a single) set embedding

       | (hyenaDNA) FM                            | C Model Core
       |   could be memoized                      |   MLP, phi
       |                                          |
       |                                          |
    [ ACGT... | PE | lg(region_size) ] -> a-vec  --> b-vec  ---
    [ ACGT... | PE | lg(region_size) ] -> a-vec  --> b-vec   |
    [ ACGT... | PE | lg(region_size) ] -> a-vec  --> b-vec   | pool (mean)          MLP, rho
    [ ACGT... | PE | lg(region_size) ] -> a-vec  --> b-vec   |-------------> c-vec ----------> d-vec
    [ ACGT... | PE | lg(region_size) ] -> a-vec  --> b-vec   |
    [ ACGT... | PE | lg(region_size) ] -> a-vec  --> b-vec  ---
                |                      |
                |                      | layer norm
                |                      |   huge scale difference of FM tensors & PE can be hard on AdamW
                |
       (chr idx, coord) encoding
"""

from bisect import bisect_right
from collections.abc import Sequence
from typing import Any, final

import torch
import torch.nn as nn
from typing_extensions import override

from giggleml.models.genomic_model import GenomicModel
from giggleml.models.hyena_dna import HyenaDNA
from giggleml.train.genomic_pe import GenomicLocusEncoding
from giggleml.utils.autograd_aware_dist_ops import all_reduce_sum
from giggleml.utils.torch_utils import (
    get_rank,
    get_world_size,
    guess_device,
)


@final
class CModel(nn.Module):
    def __init__(
        self,
        size: str = "1k",
        phi_latent: int = 512,
        rho_latent: int = 128,
        phi_depth: int = 2,
        rho_depth: int = 2,
    ):
        super().__init__()
        self.size = size
        self.hyena_dna = HyenaDNA(size)
        self.phi_latent: int = phi_latent
        self.rho_latent: int = rho_latent
        self.phi_depth: int = phi_depth
        self.rho_depth: int = rho_depth
        self.final_embed_dim = rho_latent

        def block(x: int, z: int, depth: int):
            assert depth > 0
            y = round((x + z) / 2)

            yield nn.Linear(x, y)
            yield nn.Tanh()

            for _ in range(depth - 1):
                yield nn.Linear(y, y)
                yield nn.Tanh()

            yield nn.Linear(y, z)

        self.pe = GenomicLocusEncoding(32, 8)
        self.input_dim = self.hyena_dna.embed_dim + 8 + 32 + 1  # chr + coord + size
        self.phi = nn.Sequential(*block(self.input_dim, phi_latent, phi_depth))
        self.rho = nn.Sequential(*block(phi_latent, rho_latent, rho_depth))

        # align with hyena dna dtype
        self.dtype = torch.bfloat16
        self.to(dtype=self.dtype)

    def tokenize(self, batch: Sequence[Sequence[str]]) -> list[dict[str, torch.Tensor]]:
        return [self.hyena_dna.collate(item) for item in batch]

    def set_contents_forward(self, item: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward: only includes element-wise operations.
        Takes a single, tokenized item (tokenized sequence set).
        """

        # Step 1: Get sequence embeddings from HyenaDNA
        hdna_embeds = self.hyena_dna(item)

        # Step 2: Apply phi network to each sequence embedding
        with torch.autocast("cuda", dtype=torch.float32):
            # autocast performs the op in higher prec, but keeps weights low precision
            phi = self.phi(hdna_embeds).to(hdna_embeds.dtype)
            return phi

    def set_means_forward(self, batch: torch.Tensor) -> torch.Tensor:
        """takes a batch of set means"""
        return self.rho(batch)

    def distributed_embed(
        self, fm_batch: list[torch.Tensor], intervals: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Expects two same length lists of tensors corresponding to each set in the batch,
        for each entry we expect a tensor from each list:
        1) [N, fm_dim] embeddings from the foundation model (FM)
        2) [N, 3] embeddings representing (chr ID, start, end) pairs for each item
        """

        rank = get_rank()
        world_size = get_world_size()
        dev = guess_device()  # FIXME: this is brittle
        dtype = self.dtype
        pe_dim = self.pe.output_dim
        fm_dim = self.hyena_dna.embed_dim

        # Get on the common dtype
        fm_batch = [x.to(dtype=dtype) for x in fm_batch]
        # Positional encodings from interval coordinates
        pe_batch = [self.pe(x[:, :-1]) for x in intervals]

        # Calculate Global Partition Bounds
        #   We define the "Ideal Flat Buffer" dimensions without creating it yet
        total_samples = sum(t.size(0) for t in fm_batch)
        partition_size = total_samples // world_size

        # Calculate strict start/end indices for this rank
        rank_start = rank * partition_size
        rank_end = (
            total_samples if rank == world_size - 1 else rank_start + partition_size
        )
        local_size = rank_end - rank_start

        # Allocate the Contiguous Buffer on GPU
        #   This is our ONLY allocation.
        input_buffer = torch.empty(
            local_size, fm_dim + pe_dim + 1, device=dev, dtype=dtype
        )
        flat_set_ids = torch.empty(local_size, device=dev, dtype=torch.long)

        # Scatter: Copy from Ragged CPU -> Flat GPU
        #   We iterate the CPU sets and copy ONLY the parts that fall into our rank's window.
        #   This combines "linearization" and "host-to-device transfer" into one operation.

        # Pre-calculate offsets to find relevant tensors quickly
        set_lengths = [t.size(0) for t in fm_batch]
        set_offsets = torch.tensor([0] + set_lengths[:-1]).cumsum(0).tolist()

        # Find first and last tensor involving this rank
        first_tensor_idx = bisect_right(set_offsets, rank_start) - 1

        buffer_offset = 0
        current_idx = rank_start

        # Iterate only the relevant subset of input tensors
        idx = first_tensor_idx
        while buffer_offset < local_size and idx < len(fm_batch):
            src_tensor = fm_batch[idx]
            pe_tensor = pe_batch[idx]
            src_start_global = set_offsets[idx]

            # Calculate overlap between [src_tensor] and [rank_window]
            copy_start = max(0, current_idx - src_start_global)
            copy_end = min(src_tensor.size(0), rank_end - src_start_global)
            copy_len = copy_end - copy_start

            if copy_len > 0:
                # A. Copy Data (CPU Slice -> GPU Slice)
                input_buffer[buffer_offset : buffer_offset + copy_len, :fm_dim].copy_(
                    src_tensor[copy_start:copy_end], non_blocking=True
                )
                # Copy positional encodings
                input_buffer[buffer_offset : buffer_offset + copy_len, fm_dim:-1].copy_(
                    pe_tensor[copy_start:copy_end], non_blocking=True
                )
                # Append region size
                interval_tensor = intervals[idx][copy_start:copy_end]
                region_sizes = (
                    (interval_tensor[:, 2] - interval_tensor[:, 1]).float().log2()
                ).to(dtype=self.dtype)
                input_buffer[buffer_offset : buffer_offset + copy_len, -1].copy_(
                    region_sizes, non_blocking=True
                )

                # B. Fill Metadata (GPU Fill)
                # Extremely fast operation to broadcast the scalar 'idx'
                flat_set_ids[buffer_offset : buffer_offset + copy_len].fill_(idx)

                buffer_offset += copy_len
                current_idx += copy_len

            idx += 1

        # Idiomatic Processing
        local_output: torch.Tensor = self.phi(input_buffer)  # Entire partition at once?

        # Scatter Mean
        means = torch.zeros(len(fm_batch), self.phi_latent, device=dev, dtype=dtype)
        means.index_add_(0, flat_set_ids, local_output)
        means = all_reduce_sum(means)
        means /= torch.tensor(set_lengths, device=dev, dtype=dtype).unsqueeze(-1)

        # rho
        rho_output: torch.Tensor = self.rho(means)  # cheap
        return rho_output / rho_output.norm()  # normalization helps prevent cheating

    @override
    def train(self, mode: bool = True) -> "CModel":
        super().train(mode)
        self.hyena_dna.eval()
        return self

    def hot_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    @property
    def device(self):
        return next(self.hot_parameters()).device

    @override
    def to(self, *args: Any, **kwargs: Any) -> "CModel":
        self.hyena_dna.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    @override
    def __repr__(self) -> str:
        return (
            f"CModel(size={self.size}, "
            f"phi_latent={self.phi_latent}, "
            f"rho_latent={self.rho_latent}, "
            f"phi_depth={self.phi_depth}, "
            f"rho_depth={self.rho_depth})"
        )


@final
class RowCModel(GenomicModel):
    """
    Simplified CModel that only performs the element-wise (within the sets) operations
    """

    wants: str = "sequences"

    def __init__(self, cmodel: CModel):
        super().__init__()
        self.cmodel: CModel = cmodel
        self.hyena_dna: HyenaDNA = cmodel.hyena_dna
        self.max_seq_len: int | None = self.hyena_dna.max_seq_len
        self.embed_dim: int = self.hyena_dna.embed_dim
        self.embed_dtype: torch.dtype = self.hyena_dna.embed_dtype
        self.collate = HyenaDNA(cmodel.size).collate

    @override
    def forward(self, batch: dict[str, torch.Tensor]):
        return self.cmodel.set_contents_forward(batch)

    @override
    def train(self, mode: bool = True):
        self.cmodel.train(mode)
        return self
