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

from collections.abc import Iterator, Sequence
from typing import Any, final

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from typing_extensions import override

from giggleml.embed_gen.dex import Dex
from giggleml.embed_gen.in_dex import InDex
from giggleml.iter_utils.distributed_scatter_mean import (
    distributed_scatter_mean,
)
from giggleml.iter_utils.set_flat_iter import SetFlatIter
from giggleml.models.genomic_model import GenomicModel
from giggleml.models.hyena_dna import HyenaDNA
from giggleml.utils.autograd_aware_dist_ops import all_gather_cat
from giggleml.utils.torch_utils import (
    get_rank,
    get_world_size,
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
        self.to(dtype=self.hyena_dna.embed_dtype)

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
        self, data: list[Tensor], batch_size: int, sub_workers: int
    ) -> Tensor:
        """
        Generate set-level embeddings using distributed processing.

        This method should be called on all ranks in a distributed setting.
        It processes interval datasets through the complete C Model pipeline:
        1. Freezes HyenaDNA parameters for efficiency
        2. Generates phi embeddings for all sequences using batch inference
        3. Computes set-level means using distributed scatter operations
        4. Applies rho network to produce final set embeddings

            data: List of IntervalDataset objects to process
            batch_size: Batch size for inference operations
            sub_workers: Number of sub-workers for parallel processing

        Returns:
            Tensor of final set embeddings for this rank's portion of the data only
        """
        assert sum(map(len, data)) > batch_size * (get_world_size() - 1)
        torch.set_float32_matmul_precision("medium")  # we're on half prec anyway

        # create element-wise embeddings

        sfi = SetFlatIter(data)
        flat_input: Iterator[Tensor] = iter(sfi)  # pyright: ignore[reportAssignmentType]
        true_indices = list(sfi.indices())

        phi_embeds: list[Tensor] = list()
        phi_set_indices: list[int] = list()

        with tqdm(
            total=round(len(sfi) / batch_size / get_world_size()),
            desc="phi",
            # FIXME:
            disable=(get_rank() != 9),
        ) as pbar:

            def collect_phi(block: Sequence[tuple[int, Tensor]]):
                flat_indices, embeds = zip(*block)
                phi_embeds.extend(embeds)
                set_indices = [true_indices[i][0] for i in flat_indices]
                phi_set_indices.extend(set_indices)
                pbar.update()

            # InDex allows stable output reordering even with num_workers != 0
            InDex(self.phi).execute(
                flat_input, collect_phi, batch_size, sub_workers, auto_reclaim=False
            )

        # set-level mean
        phi_tensor = torch.stack(phi_embeds)
        phi_indices_tensor = torch.tensor(phi_set_indices, device=phi_tensor.device)
        set_means = distributed_scatter_mean(phi_tensor, phi_indices_tensor)

        # rho pass

        local_rho_embeds = list()
        rho_dex = Dex(self.rho)

        def collect_rho(block: Sequence[torch.Tensor]):
            # block = [block.to("cpu", non_blocking=True) for block in block]
            local_rho_embeds.extend(block)

        batch_size = min(batch_size, len(data) // get_world_size())

        with torch.autocast("cuda", dtype=torch.float32):
            # Dex default collate is so fast that we can avoid sub-workers
            #   and must, because we can't easily transfer grad tensors between workers
            rho_dex.execute(
                set_means, collect_rho, batch_size, num_workers=0, auto_reclaim=False
            )

        # 4. regroup
        local_rho_embeds_tensor = torch.stack(local_rho_embeds)
        total_rho_embeds = all_gather_cat(local_rho_embeds_tensor)
        total_ordering = rho_dex.simulate_global_concat(range(len(data)), batch_size)
        return total_rho_embeds[total_ordering]

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
