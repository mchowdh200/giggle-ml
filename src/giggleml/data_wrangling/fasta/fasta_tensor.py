from collections.abc import Callable
from os import PathLike
from typing import cast

import torch
from torch import Tensor

from giggleml.data_wrangling.fasta.utils import Fasta, ensure_fa
from giggleml.utils.path_utils import is_path_like


def to_gpu_serializable(
    fasta: Fasta | PathLike, tokenizer: Callable[[str], Tensor]
) -> tuple[Tensor, Tensor, Tensor, dict[str, int]]:
    """
    Convert a fasta dict into GPU-serializable tensors.

    @param fasta: chromosome -> sequence mapping or path to fasta file
    @param tokenizer: function that takes a string and returns a tensor of tokens
    @return: (tokens, offsets, sizes, chr_to_idx) where:
        - tokens: concatenated tokens tensor
        - offsets: start indices for each chromosome in tokens tensor
        - sizes: size of each chromosome's token sequence
        - chr_to_idx: mapping from chromosome name to index
    """

    if is_path_like(fasta):
        fasta = ensure_fa(cast(PathLike, fasta))

    fasta = cast(Fasta, fasta)
    chr_names = list(fasta.keys())
    chr_to_idx = {name: idx for idx, name in enumerate(chr_names)}

    all_tokens = []
    offsets = []
    sizes = []
    current_offset = 0

    for chr_name in chr_names:
        sequence = fasta[chr_name]
        tokens = tokenizer(sequence)

        offsets.append(current_offset)
        sizes.append(len(tokens))
        all_tokens.append(tokens)
        current_offset += len(tokens)

    concatenated_tokens = torch.cat(all_tokens, dim=0)
    offsets_tensor = torch.tensor(offsets, dtype=torch.long)
    sizes_tensor = torch.tensor(sizes, dtype=torch.long)

    return concatenated_tokens, offsets_tensor, sizes_tensor, chr_to_idx
