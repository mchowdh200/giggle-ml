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

from collections.abc import Sequence
from typing import final

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing_extensions import override

from giggleml.data_wrangling.interval_dataset import IntervalDataset
from giggleml.embed_gen.batch_infer import BatchInfer
from giggleml.embed_gen.dex import Dex
from giggleml.embed_gen.embed_model import HyenaDNA
from giggleml.iter_utils.distributed_scatter_mean import distributed_scatter_mean_iter
from giggleml.iter_utils.set_flat_iter import SetFlatIter
from giggleml.utils.torch_utils import all_gather_concat, freeze_model


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
        self.final_embed_dim = final_embed_dim_factor * hyena_embed_dim

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
            nn.Linear(rho_hidden_dim, self.final_embed_dim),
        )

    @override
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # INFO: only includes element-wise operations

        # Step 1: Get sequence embeddings from HyenaDNA
        hdna_embeds = super().forward(batch)

        # Step 2: Apply phi network to each sequence embedding
        if self.use_gradient_checkpointing and self.training:
            # prevent None
            activs = checkpoint(self.phi, hdna_embeds, use_reentrant=False)
            assert activs is not None
            return activs
        return self.phi(hdna_embeds)

    def distributed_embed(
        self, data: Sequence[IntervalDataset], batch_size: int, sub_workers: int
    ) -> torch.Tensor:
        """
        Generate set-level embeddings using distributed processing.

        This method should be called on all ranks in a distributed setting.
        It processes interval datasets through the complete M Model pipeline:
        1. Freezes HyenaDNA parameters for efficiency
        2. Generates phi embeddings for all sequences using batch inference
        3. Computes set-level means using distributed scatter operations
        4. Applies rho network to produce final set embeddings

        Args:
            data: List of IntervalDataset objects to process
            batch_size: Batch size for inference operations
            sub_workers: Number of sub-workers for parallel processing

        Returns:
            Tensor of final set embeddings for this rank's portion of the data only
        """

        # 1. freeze HyenaDNA parameters
        freeze_model(super()._model)

        # 2. create element-wise embeddings
        phi_embeds = list()
        # BatchInfer is a Dex that handles genomic nuances like FASTA mapping & interval chunking
        BatchInfer(self, batch_size, sub_workers).raw(data, phi_embeds.extend)

        # 3. set-level mean
        structure = SetFlatIter(data)
        set_indices = structure.set_indices()
        set_means = torch.stack(
            [i for (_, i) in distributed_scatter_mean_iter(set_indices, phi_embeds)]
        )

        # 3. rho pass
        local_rho_embeds = list()
        rho_dex = Dex(self.rho)
        rho_dex.execute(set_means, local_rho_embeds.extend, batch_size, sub_workers)

        # 4. regroup
        total_rho_embeds = all_gather_concat(torch.stack(local_rho_embeds))
        total_ordering = rho_dex.simulate_global_concat(range(len(data)), batch_size)
        return total_rho_embeds[total_ordering]

    @override
    def train(self, mode: bool = True) -> "MModel":
        super().train(mode)
        super()._model.eval()
        return self

    def hot_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    @override
    def __repr__(self) -> str:
        return (
            f"MModel(size={self.size_type}, "
            f"phi_hidden_dim_factor={self.phi_hidden_dim_factor}, "
            f"rho_hidden_dim_factor={self.rho_hidden_dim_factor}, "
            f"final_embed_dim_factor={self.final_embed_dim_factor}, "
            f"use_gradient_checkpointing={self.use_gradient_checkpointing})"
        )
