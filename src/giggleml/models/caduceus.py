from collections.abc import Sequence
from logging import warning
from typing import cast, final, override

import torch
from torch import Tensor
from torch.types import Device
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer

from giggleml.models.genomic_model import GenomicModel

type CaduceusBatch = tuple[dict[str, Tensor], dict[str, Tensor]]


@final
class Caduceus(GenomicModel):
    """
    Caduceus is strictly cuda-only -- can't even run it cpu-side.
    """

    wants = "sequences"
    embed_dtype = torch.float32
    embed_dim = 256

    def __init__(self, size: str = "1k", stable_pad: bool = False):
        """
        The bi-mamba model is forced to always enumerate padding and it has not been designed
        to ignore it so the padding can not be entirely removed with masked mean pooling. As a result,
        when sequences are padded to form the tokenized batch, there is a small padding leak between
        sequences. This can be mitigated with batches of size one, but the padding leak almost certainly
        won't affect downstream results. Otherwise, and for a performance cost, this can also be mitigated
        with stable_pad=True.
        """

        super().__init__()

        if size == "1k":
            model_name = "kuleshov-group/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3"  # I stop reformat
            rev = "9865108985311772704c2ece0bac082153a6167b"
            self.max_seq_len = 1000
        elif size == "131k":
            model_name = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
            rev = "b0477522ac5d044ad03578aa724ec8e4bdbd405b"
            self.max_seq_len = int(131e3)
        else:
            raise ValueError(f"{size}, unknown size")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, revision=rev
        )
        self._model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, revision=rev
        )

        self.stable_pad = stable_pad
        self.collate = self._collate

    @property
    def device(self) -> Device:
        return next(self.parameters()).device

    @staticmethod
    def reverse_complement[T: str | Sequence[str]](item: T) -> T:
        """Gets the reverse complement of a DNA sequence."""

        complement_map = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}

        def _seq_rc(seq: str) -> str:
            seq_upper = seq.upper()
            rc_seq = "".join(
                complement_map.get(base, base) for base in reversed(seq_upper)
            )
            return rc_seq

        if isinstance(item, str):
            return _seq_rc(item)
        return cast(T, [_seq_rc(x) for x in item])

    def _collate(self, batch: Sequence[str]) -> CaduceusBatch:
        assert self.max_seq_len

        for item in batch:
            if len(item) > self.max_seq_len:
                # not raising because technically speaking, the model is capable of indefinite context
                warning(
                    f"Caduceus called with sequences greater than context limit. {len(item)} > {self.max_seq_len}"
                )
                break

        rc_batch = [Caduceus.reverse_complement(x) for x in batch]
        padding_strategy = "max_length" if self.stable_pad else True
        fwd_tokens = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=padding_strategy,
            max_length=self.max_seq_len if self.stable_pad else None,
            return_attention_mask=True,
        )
        rc_tokens = self.tokenizer(
            rc_batch,
            return_tensors="pt",
            padding=padding_strategy,
            max_length=self.max_seq_len if self.stable_pad else None,
            return_attention_mask=True,
        )
        return fwd_tokens, rc_tokens

    @override
    def forward(self, batch: CaduceusBatch) -> Tensor:
        # automatically move inputs to device
        fwd_batch = {k: v.to(self.device) for k, v in batch[0].items()}
        rc_batch = {k: v.to(self.device) for k, v in batch[1].items()}

        # model calls
        fwd = self._model(input_ids=fwd_batch["input_ids"], output_hidden_states=True)
        rc = self._model(input_ids=rc_batch["input_ids"], output_hidden_states=True)

        # final hidden state shape [batch size, max seq in batch, embed dim]
        fwd_hidden: Tensor = fwd.hidden_states[-1]
        rc_hidden: Tensor = rc.hidden_states[-1]

        def masked_mean_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.shape).float()
            sum_embeddings = torch.sum(hidden_states * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask

        fwd_pooled = masked_mean_pool(fwd_hidden, fwd_batch["attention_mask"])
        rc_pooled = masked_mean_pool(rc_hidden, rc_batch["attention_mask"])
        # this double call + sum implements RC-invariance.
        # I don't care about that property, but the model was trained with it.
        return (fwd_pooled + rc_pooled) / 2
