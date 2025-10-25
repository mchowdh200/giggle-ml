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
from typing import Any, cast, final

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing_extensions import override

from giggleml.data_wrangling.interval_dataset import IntervalDataset
from giggleml.embed_gen.batch_infer import GenomicEmbedder, Idx
from giggleml.embed_gen.dex import Dex
from giggleml.iter_utils.distributed_scatter_mean import distributed_scatter_mean_iter
from giggleml.models.genomic_model import GenomicModel
from giggleml.models.hyena_dna import HyenaDNA
from giggleml.utils.torch_utils import all_gather_concat, freeze_model


@final
class MModel(nn.Module):
    """
    Note that this model takes in sets of sequence sets as input batches.
    The forward method, related procedures, including tokenize, have been
    designed to operate on single items (sequence set)-- not batches.
    This is due to its use in the parallel embedding pipeline which expects
    a flat stream of items, not non-homogeneous sets.
    """

    def __init__(
        self,
        size: str = "1k",
        phi_hidden_dim_factor: int = 4,
        rho_hidden_dim_factor: int = 2,
        final_embed_dim_factor: int = 1,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()

        self.hyena_dna = HyenaDNA(size)
        self.phi_hidden_dim_factor = phi_hidden_dim_factor  # for __repr__
        self.rho_hidden_dim_factor = rho_hidden_dim_factor  # for __repr__
        self.final_embed_dim_factor = final_embed_dim_factor  # for __repr__
        self.use_gradient_checkpointing = use_gradient_checkpointing

        hyena_embed_dim = self.hyena_dna.embed_dim
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

        # align with hyena dna dtype
        self.to(dtype=self.hyena_dna.embed_dtype)

    def tokenize(self, batch: Sequence[Sequence[str]]) -> list[dict[str, torch.Tensor]]:
        return [self.hyena_dna.collate(item) for item in batch]

    @override
    def forward(self, batch: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        # 1. phi pass
        phi_embeds = torch.cat(
            [self.set_contents_forward(item) for item in batch], dim=0
        )

        # 2. set-level mean

        dev = phi_embeds.device
        dtype: torch.dtype = phi_embeds.dtype

        set_sizes = torch.tensor(
            [len(item["input_ids"]) for item in batch],
            device=dev,
        )
        set_indices = torch.arange(0, len(batch), device=dev).repeat_interleave(
            set_sizes
        )

        set_means = torch.zeros(
            len(batch), phi_embeds.shape[1], device=dev, dtype=dtype
        ).scatter_add_(
            0, set_indices.unsqueeze(1).expand_as(phi_embeds), phi_embeds
        ) / set_sizes.unsqueeze(1)

        # 3. rho pass
        return self.set_means_forward(set_means)

    def set_contents_forward(self, item: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward: only includes element-wise operations.
        Takes a single, tokenized item (tokenized sequence set).
        """

        # Step 1: Get sequence embeddings from HyenaDNA
        hdna_embeds = self.hyena_dna(item)

        # Step 2: Apply phi network to each sequence embedding
        if self.use_gradient_checkpointing and self.training:
            # prevent None
            activs = checkpoint(self.phi, hdna_embeds, use_reentrant=False)
            assert activs is not None
            return activs

        return self.phi(hdna_embeds)

    def set_means_forward(self, batch: torch.Tensor) -> torch.Tensor:
        """takes a batch of set means"""
        return self.rho(batch)

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
        freeze_model(self.hyena_dna)

        # 2. create element-wise embeddings

        phi_embeds = list()
        phi_set_indices = list()

        def collect_outputs(block: Sequence[tuple[Idx, torch.Tensor]]):
            indices, embeds = zip(*block)
            phi_set_indices.extend(i for (i, _) in indices)
            phi_embeds.extend(embeds)

        # The RowMModel wraps a Dex that handles genomic nuances like FASTA mapping & interval chunking.
        # We have to conditionally unwrap self because tools like fabric like to wrap the model in
        # unpicklable shells.
        unwrapped_self = cast(
            "MModel", self.module if hasattr(self, "module") else self
        )
        GenomicEmbedder(RowMModel(unwrapped_self), batch_size, sub_workers).raw(
            data, collect_outputs
        )

        # 3. set-level mean
        set_means = torch.stack(
            [i for (_, i) in distributed_scatter_mean_iter(phi_set_indices, phi_embeds)]
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
        self.hyena_dna.eval()
        return self

    def hot_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    @property
    def device(self):
        return next(self.hot_parameters()).device

    @override
    def to(self, *args: Any, **kwargs: Any) -> "MModel":
        self.hyena_dna.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    @override
    def __repr__(self) -> str:
        return (
            f"MModel(size={self.hyena_dna.size_type}, "
            f"phi_hidden_dim_factor={self.phi_hidden_dim_factor}, "
            f"rho_hidden_dim_factor={self.rho_hidden_dim_factor}, "
            f"final_embed_dim_factor={self.final_embed_dim_factor}, "
            f"use_gradient_checkpointing={self.use_gradient_checkpointing})"
        )


@final
class RowMModel(GenomicModel):
    """
    Simplified MModel that only performs the element-wise (within the sets) operations
    """

    wants: str = "sequences"

    def __init__(self, mmodel: MModel):
        super().__init__()
        self.mmodel: MModel = mmodel
        self.hyena_dna: HyenaDNA = mmodel.hyena_dna
        self.max_seq_len: int | None = self.hyena_dna.max_seq_len
        self.embed_dim: int = self.hyena_dna.embed_dim
        self.embed_dtype: torch.dtype = self.hyena_dna.embed_dtype

    @override
    def collate(self, batch: Sequence[str]):
        return self.mmodel.tokenize([batch])

    @override
    def forward(self, batch: dict[str, torch.Tensor]):
        return self.mmodel.set_contents_forward(batch)

    @override
    def train(self, mode: bool = True):
        self.mmodel.train(mode)
        return self
