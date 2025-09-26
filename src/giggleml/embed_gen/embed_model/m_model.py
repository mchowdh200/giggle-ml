"""
M Model, deep sets architecture, hyenaDNA pre-processing (Optimized)

    (interval set -> sequence set --)-> sequence embeddings -> [M Model] -> (a single) set embedding

          | hyenaDNA    | M Model Core
          |             |   MLP, phi
          |             |   (!) activations NOT saved by default
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

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate, chain
from typing import final

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing_extensions import override

from giggleml.embed_gen.embed_model import EmbedModel, HyenaDNA
from giggleml.utils.torch_utils import freeze_model


@dataclass(frozen=True)
class MModelBatch:
    hyena_batch: dict[str, torch.Tensor]
    set_boundaries: torch.Tensor


@final
class MModel(EmbedModel):
    wants = "sequences"

    def __init__(
        self,
        hyena_dna: HyenaDNA,
        phi_hidden_dim_factor: int = 4,
        rho_hidden_dim_factor: int = 2,
        final_embed_dim_factor: int = 1,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.hyena_dna = freeze_model(hyena_dna)
        self.max_seq_len = self.hyena_dna.max_seq_len
        self.phi_hidden_dim_factor = phi_hidden_dim_factor  # for __repr__
        self.rho_hidden_dim_factor = rho_hidden_dim_factor  # for __repr__
        self.final_embed_dim_factor = final_embed_dim_factor  # for __repr__
        self.use_gradient_checkpointing = use_gradient_checkpointing

        hyena_embed_dim = self.hyena_dna.embed_dim
        phi_hidden_dim = phi_hidden_dim_factor * hyena_embed_dim
        rho_hidden_dim = rho_hidden_dim_factor * hyena_embed_dim
        final_embed_dim = final_embed_dim_factor * hyena_embed_dim
        self.embed_dim = final_embed_dim

        self.phi = nn.Sequential(
            nn.Linear(hyena_embed_dim, phi_hidden_dim),
            nn.Tanh(),
            nn.Linear(phi_hidden_dim, phi_hidden_dim),
            nn.Tanh(),
            nn.Linear(phi_hidden_dim, rho_hidden_dim),
        )

        self.rho_input_dim = rho_hidden_dim
        self.rho = nn.Sequential(
            nn.Linear(rho_hidden_dim, rho_hidden_dim),
            nn.Tanh(),
            nn.Linear(rho_hidden_dim, final_embed_dim),
        )

    @override
    def collate(self, batch: Sequence[Sequence[str]]) -> MModelBatch:
        """
        Flattens a batch of sequence sets into a single list for HyenaDNA
        and creates boundary indices to reconstruct the sets later.
        """
        all_sequences = list(chain.from_iterable(batch))
        set_sizes = [len(s) for s in batch]
        set_boundaries = torch.tensor(
            [0] + list(accumulate(set_sizes)), dtype=torch.long
        )

        if not all_sequences:
            device = next(self.phi.parameters()).device
            return MModelBatch(
                hyena_batch={
                    "input_ids": torch.empty(0, 0, dtype=torch.long, device=device)
                },
                set_boundaries=set_boundaries.to(device),
            )

        hyena_batch = self.hyena_dna.collate(all_sequences)
        device = hyena_batch["input_ids"].device

        return MModelBatch(
            hyena_batch=hyena_batch,
            set_boundaries=set_boundaries.to(device),
        )

    def _phi_forward_with_checkpointing(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training:
            # prevent None
            assert (activs := checkpoint(self.phi, x, use_reentrant=False))
            return activs
        return self.phi(x)

    @override
    def forward(self, batch: MModelBatch) -> torch.Tensor:
        with torch.set_grad_enabled(self.training):
            hyena_batch = batch.hyena_batch
            set_boundaries = batch.set_boundaries
            device = set_boundaries.device
            num_sets = len(set_boundaries) - 1

            if num_sets == 0:
                return torch.empty(
                    0, self.embed_dim, device=device, dtype=torch.float32
                )

            # If there are no sequences at all, create zero embeddings for each set.
            if hyena_batch["input_ids"].numel() == 0:
                mean_phi_embedding = torch.zeros(
                    num_sets, self.rho_input_dim, device=device, dtype=torch.float32
                )
                return self.rho(mean_phi_embedding).to(dtype=torch.float32)

            # Step 1: Get sequence embeddings from the foundational model.
            sequence_embeddings = self.hyena_dna(hyena_batch)

            # Step 2: Apply phi network to each sequence embedding.
            phi_embeddings = self._phi_forward_with_checkpointing(sequence_embeddings)

            # --- Vectorized Aggregation ---
            # Step 3: Compute the mean of phi_embeddings for each set in a single operation.
            set_sizes = set_boundaries[1:] - set_boundaries[:-1]

            # Create an index tensor mapping each embedding to its set ID.
            # e.g., for set sizes [2, 3], it creates tensor([0, 0, 1, 1, 1]).
            set_indices = torch.arange(num_sets, device=device).repeat_interleave(
                set_sizes
            )

            # Sum embeddings for each set using the highly efficient scatter_add_.
            summed_phi = torch.zeros(
                num_sets,
                phi_embeddings.shape[-1],
                device=device,
                dtype=phi_embeddings.dtype,
            )
            index_map = set_indices.unsqueeze(1).expand_as(phi_embeddings)
            summed_phi.scatter_add_(0, index_map, phi_embeddings)

            # Calculate mean. Clamping size to min=1 avoids division by zero for empty sets.
            # The sum for an empty set is 0, so 0 / 1 = 0, which is the correct mean.
            set_sizes_safe = set_sizes.clamp(min=1).unsqueeze(1)
            mean_phi_embedding = summed_phi / set_sizes_safe
            # --- End Vectorized Aggregation ---

            # Step 4: Apply rho network to the batched mean embeddings.
            final_embedding = self.rho(mean_phi_embedding)

            return final_embedding.to(dtype=torch.float32)

    @override
    def __repr__(self) -> str:
        return (
            f"MModel(hyena_dna={repr(self.hyena_dna)}, "
            f"phi_hidden_dim_factor={self.phi_hidden_dim_factor}, "
            f"rho_hidden_dim_factor={self.rho_hidden_dim_factor}, "
            f"final_embed_dim_factor={self.final_embed_dim_factor}, "
            f"use_gradient_checkpointing={self.use_gradient_checkpointing})"
        )
