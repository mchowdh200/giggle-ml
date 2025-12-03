"""
C Model, deep sets architecture, hyenaDNA pre-processing
if giggle is a giggle, this is a chuckle

    (interval set -> sequence set --)-> sequence embeddings -> [C Model] -> (a single) set embedding

          | hyenaDNA    | C Model Core
          |             |   MLP, phi
          |             |
          |             |
    ACGT... -> d1-vec  --> d2-vec  ---
    ACGT... -> d1-vec  --> d2-vec   |
    ACGT... -> d1-vec  --> d2-vec   | mean           MLP, rho
    ACGT... -> d1-vec  --> d2-vec   |------> d2-vec ----------> d3-vec
    ACGT... -> d1-vec  --> d2-vec   |
    ACGT... -> d1-vec  --> d2-vec  ---
                    |
                    |
       could be memoized

"""

from bisect import bisect_right
from collections.abc import Sequence
from typing import Any, final

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from typing_extensions import override

from giggleml.iter_utils.distributed_scatter_mean import (
    distributed_scatter_mean,
)
from giggleml.models.genomic_model import GenomicModel
from giggleml.models.hyena_dna import HyenaDNA
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

        hdna_latent = self.hyena_dna.embed_dim
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

        self.phi = nn.Sequential(*block(hdna_latent, phi_latent, phi_depth))
        self.rho = nn.Sequential(*block(phi_latent, rho_latent, rho_depth))

        # align with hyena dna dtype
        self.dtype = self.hyena_dna.embed_dtype
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

    def distributed_embed(self, input_sets: list[torch.Tensor]) -> torch.Tensor:
        rank = get_rank()
        world_size = get_world_size()
        dev = guess_device()

        # 1. Calculate Global Partition Bounds
        # We define the "Ideal Flat Buffer" dimensions without creating it yet
        total_samples = sum(t.size(0) for t in input_sets)
        partition_size = total_samples // world_size

        # Calculate strict start/end indices for this rank
        rank_start = rank * partition_size
        rank_end = (
            total_samples if rank == world_size - 1 else rank_start + partition_size
        )
        local_size = rank_end - rank_start

        # 2. Allocate the Contiguous Buffer on GPU
        # This is our ONLY allocation.
        flat_input = torch.empty(
            local_size, self.hyena_dna.embed_dim, device=dev, dtype=self.dtype
        )
        flat_set_ids = torch.empty(local_size, device=dev, dtype=torch.long)

        # 3. Scatter: Copy from Ragged CPU -> Flat GPU
        # We iterate the CPU sets and copy ONLY the parts that fall into our rank's window.
        # This combines "linearization" and "host-to-device transfer" into one operation.

        # Pre-calculate offsets to find relevant tensors quickly
        lengths = [t.size(0) for t in input_sets]
        offsets = torch.tensor([0] + lengths[:-1]).cumsum(0).tolist()

        # Find first and last tensor involving this rank
        first_tensor_idx = bisect_right(offsets, rank_start) - 1

        buffer_offset = 0
        current_idx = rank_start

        # Iterate only the relevant subset of input tensors
        idx = first_tensor_idx
        while buffer_offset < local_size and idx < len(input_sets):
            src_tensor = input_sets[idx]
            src_start_global = offsets[idx]

            # Calculate overlap between [src_tensor] and [rank_window]
            copy_start = max(0, current_idx - src_start_global)
            copy_end = min(src_tensor.size(0), rank_end - src_start_global)
            copy_len = copy_end - copy_start

            if copy_len > 0:
                # A. Copy Data (CPU Slice -> GPU Slice)
                flat_input[buffer_offset : buffer_offset + copy_len].copy_(
                    src_tensor[copy_start:copy_end], non_blocking=True
                )

                # B. Fill Metadata (GPU Fill)
                # Extremely fast operation to broadcast the scalar 'idx'
                flat_set_ids[buffer_offset : buffer_offset + copy_len].fill_(idx)

                buffer_offset += copy_len
                current_idx += copy_len

            idx += 1

        # 4. Idiomatic Processing
        local_output = self.phi(flat_input)  # Process entire partition at once?

        # 5. Scatter Mean
        means = distributed_scatter_mean(local_output, flat_set_ids)

        # 6. rho
        rho_outputs = self.rho(means) if get_rank() == 0 else None  # pyright: ignore[reportAssignmentType]
        rho_outputs: Tensor = dist.broadcast(rho_outputs, src=0)  # pyright: ignore[reportAssignmentType]

        return rho_outputs

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
