"""
M Model, deep sets architecture, hyenaDNA pre-processing

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

from typing import final

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing_extensions import override

from giggleml.embed_gen.embed_model import HyenaDNA


@final
class MModel(HyenaDNA):
    wants = "sequences"

    def __init__(
        self,
        size: str = "1k",
        phi_hidden_dim_factor: int = 4,
        rho_hidden_dim_factor: int = 2,
        final_embed_dim_factor: int = 1,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__(size)
        self.phi_hidden_dim_factor = phi_hidden_dim_factor  # for __repr__
        self.rho_hidden_dim_factor = rho_hidden_dim_factor  # for __repr__
        self.final_embed_dim_factor = final_embed_dim_factor  # for __repr__
        self.use_gradient_checkpointing = use_gradient_checkpointing

        hyena_embed_dim = self.embed_dim
        phi_hidden_dim = phi_hidden_dim_factor * hyena_embed_dim
        rho_hidden_dim = rho_hidden_dim_factor * hyena_embed_dim
        final_embed_dim = final_embed_dim_factor * hyena_embed_dim

        self.phi = nn.Sequential(
            nn.Linear(hyena_embed_dim, phi_hidden_dim),
            nn.Tanh(),
            nn.Linear(phi_hidden_dim, phi_hidden_dim),
            nn.Tanh(),
            nn.Linear(phi_hidden_dim, rho_hidden_dim),
        )

        self.rho = nn.Sequential(
            nn.Linear(rho_hidden_dim, rho_hidden_dim),
            nn.Tanh(),
            nn.Linear(rho_hidden_dim, final_embed_dim),
        )

    @override
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # INFO: only includes element-wise operations

        # Step 1: Get sequence embeddings from HyenaDNA
        hdna_embeds = super().forward(batch)

        # Step 2: Apply phi network to each sequence embedding
        if self.use_gradient_checkpointing and self.training:
            # prevent None
            assert (activs := checkpoint(self.phi, hdna_embeds, use_reentrant=False))
            return activs
        return self.phi(hdna_embeds)

    @override
    def __repr__(self) -> str:
        return (
            f"MModel(size={self.size_type}, "
            f"phi_hidden_dim_factor={self.phi_hidden_dim_factor}, "
            f"rho_hidden_dim_factor={self.rho_hidden_dim_factor}, "
            f"final_embed_dim_factor={self.final_embed_dim_factor}, "
            f"use_gradient_checkpointing={self.use_gradient_checkpointing})"
        )
