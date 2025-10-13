from collections.abc import Sequence
from functools import cached_property
from typing import Any

import torch
from torch._prims_common import DeviceLikeType
from torch.types import Device
from transformers import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModel
from typing_extensions import override

from giggleml.embed_gen.embed_model import EmbedModel
from giggleml.utils.types import lazy


@lazy
class HyenaDNA(EmbedModel):
    wants = "sequences"
    embed_dtype = torch.float16

    def __init__(self, size: str = "1k"):
        super().__init__()

        """
        Supported sizes: { 1k, 16k, 32k, 160k, 450k, 1m } corresponding to:
            hyenadna-tiny-1k-seqlen: 1024,
            hyenadna-small-32k-seqlen: 32768,
            hyenadna-medium-160k-seqlen: 160000,
            hyenadna-medium-450k-seqlen: 450000,  # T4 up to here
            hyenadna-large-1m-seqlen: 1_000_000,  # only A100 (paid tier)
        """

        details = {
            "1k": (
                1024,
                "LongSafari/hyenadna-tiny-1k-seqlen-hf",
                128,
                "e8c1eff",
            ),
            "16k": (
                16386,
                "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf",
                128,
                "d79fa37",
            ),
            "32k": (
                32768,
                "LongSafari/hyenadna-small-32k-seqlen-hf",
                256,
                "8fe770c",
            ),
            "160k": (
                160000,
                "LongSafari/hyenadna-medium-160k-seqlen-hf",
                256,
                "7ebf717",
            ),
            "450k": (
                450000,
                "LongSafari/hyenadna-medium-450k-seqlen-hf",
                256,
                "42dedd4",
            ),
            "1m": (
                1_000_000,
                "LongSafari/hyenadna-large-1m-seqlen-hf",
                256,
                "0a629ab",
            ),
        }

        if size not in details:
            raise ValueError(
                f"Unsupported size {size}.Supported sizes are {list(details.keys())}"
            )

        max_seq_len, name, e_dim, rev = details[size]

        self.rev: str = rev
        self.max_seq_len: int | None = max_seq_len
        self.checkpoint: str = name
        self.embed_dim: int = e_dim
        self.size_type: str = size  # used for __repr__ only

        self._device: Device | None = None

    @cached_property
    def _model(self):
        # WARN: HyenaDNA cannot be torch.compile(.)ed because the Hyena layers
        # use FFT which is fundamentally based on complex numbers. TorchInductor
        # does not support complex operators (4/30/2025)
        # model = torch.compile(model)
        return AutoModel.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=self.rev,
        ).to(self._device)

    @cached_property
    def _tokenizer(self):
        return AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code=True)

    @property
    def device(self) -> Device:
        return self._device

    @override
    def to(
        self, device: DeviceLikeType | None, *args: Any, **kwargs: Any
    ) -> "HyenaDNA":
        self._device = device
        return super().to(device, *args, **kwargs)

    @override
    def collate(self, batch: Sequence[str]) -> dict[str, torch.Tensor]:
        if self.max_seq_len is None:
            raise ValueError("How did this HyenaDNA instance get a None maxSeqLen?")

        for item in batch:
            if len(item) > self.max_seq_len:
                raise ValueError("Sequence exceeds max length; refusing to truncate.")

        batch = [item.upper() for item in batch]

        with torch.set_grad_enabled(self.training):
            # INFO: 1. tokenization

            # WARN: this was a legacy method:   tokenized = self.tokenizer.batch_encode_plus(
            tokenized = self._tokenizer(
                batch,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )

            inputs: dict[str, torch.Tensor] = {
                k: v.to(self.device, non_blocking=True) for k, v in tokenized.items()
            }

            # INFO: 2. create attention mask for masked mean pooling
            input_ids = inputs["input_ids"]
            seq_lens = torch.tensor(
                [len(item) for item in batch],
                dtype=torch.float32,
                device=self.device,
                requires_grad=False,
            )
            seq_lens = seq_lens.unsqueeze(dim=1)
            indices = torch.arange(
                self.max_seq_len,
                dtype=torch.float32,
                device=self.device,
                requires_grad=False,
            )
            mask = (indices < seq_lens).float()
            mask = mask.unsqueeze(-1)
            mask = mask.flip(dims=[1])  # because HyenaDNA pre-pads

            inputs["attention_mask"] = mask

            return inputs

    @override
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.set_grad_enabled(self.training):
            # INFO: 2. inference
            inputs = batch["input_ids"]

            hidden: torch.Tensor = self._model(
                input_ids=inputs
            ).last_hidden_state  # shape (batch, seqMax, eDim)

            batch_size, max_seq_len, hidden_dim = hidden.shape

            # sanity
            assert batch_size == len(inputs)
            assert max_seq_len == self.max_seq_len
            assert hidden_dim == self.embed_dim

            # INFO: 3. masked mean pooling using precomputed mask
            mask = batch["attention_mask"]
            mask = mask.expand(hidden.shape)

            # calculate sequence lengths from mask for normalization
            seq_lens = mask.sum(dim=1)[:, 0].unsqueeze(1)

            # zero values corresponding to padded regions
            hidden = hidden * mask
            # mean aggregation, removes seqMax dimension
            hidden = torch.sum(hidden, dim=1)
            # clamp ensures no divide by zero issue
            hidden /= torch.clamp(seq_lens, min=1e-9)

            return hidden.to(dtype=torch.float16)

    @override
    def __repr__(self):
        return f"HyenaDNA({self.size_type})"
