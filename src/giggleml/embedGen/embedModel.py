from abc import ABC
from collections.abc import Sequence
from functools import cached_property
from typing import Any, ClassVar, Protocol, Self, cast, final

import torch
from torch.types import Device
from transformers import AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.auto.modeling_auto import AutoModel
from typing_extensions import override

from ..utils.types import GenomicInterval

# INFO: !! Currently all modules assume "embedding vectors" are float32.

# ===================
#    EmbedModel
# ===================


class EmbedModel(ABC):
    wants: ClassVar[str]  # Type of data this model accepts: "sequences" or "intervals"
    maxSeqLen: int | None  # Maximum sequence length the model can handle
    embedDim: int  # Dimension of the output embeddings

    def embed(self, batch: Sequence[Any]) -> torch.FloatTensor:
        """Embed a batch of inputs and return tensor embeddings."""
        ...

    def to(self, device: Device) -> Self: ...


# ===================
#    HyenaDNA
# ===================


@final
class HyenaDNA(EmbedModel):
    wants = "sequences"

    def __init__(self, size: str = "1k"):
        """
        Supported sizes: { 1k, 32k, 160k, 450k, 1m } corresponding to:
            hyenadna-tiny-1k-seqlen: 1024,
            hyenadna-small-32k-seqlen: 32768,
            hyenadna-medium-160k-seqlen: 160000,
            hyenadna-medium-450k-seqlen: 450000,  # T4 up to here
            hyenadna-large-1m-seqlen: 1_000_000,  # only A100 (paid tier)
        """

        details = {
            "1k": (1024, "LongSafari/hyenadna-tiny-1k-seqlen-hf", 128),
            "32k": (32768, "LongSafari/hyenadna-small-32k-seqlen-hf", 256),
            "160k": (160000, "LongSafari/hyenadna-medium-160k-seqlen-hf", 256),
            "450k": (450000, "LongSafari/hyenadna-medium-450k-seqlen-hf", 256),
            "1m": (1_000_000, "LongSafari/hyenadna-large-1m-seqlen-hf", 256),
        }

        if size not in details:
            raise ValueError(
                f"Unsupported size {size}." f"Supported sizes are {list(details.keys())}"
            )

        maxSeqLen, name, eDim = details[size]
        self.maxSeqLen = maxSeqLen
        self.checkpoint = name
        self.embedDim: int = eDim
        self._device: Device = None

    @override
    def to(self, device: Device) -> Self:
        self._device = device
        return self._model.to(device)

    @cached_property
    def _model(self):
        model: Any = AutoModel.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()
        # WARN: HyenaDNA cannot be torch.compile(.)ed because the Hyena layers
        # use FFT which is fundamentally based on complex numbers. TorchInductor
        # does not support complex operators (4/30/2025)
        # model = torch.compile(model)
        model.to(self._device)
        return model

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code=True)

    @override
    def embed(self, batch: Sequence[str]) -> torch.FloatTensor:
        if self.maxSeqLen is None:
            raise ValueError("How did this HyenaDNA instance get a None maxSeqLen?")

        for item in batch:
            if len(item) > self.maxSeqLen:
                raise ValueError("Sequence exceeds max length; refusing to truncate.")

        with torch.no_grad():
            # INFO: 1. tokenization
            tokenized = self.tokenizer.batch_encode_plus(
                batch,
                max_length=self.maxSeqLen,
                padding="max_length",
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )

            dev = self._device
            inputs = {k: v.to(dev, non_blocking=True) for k, v in tokenized.items()}
            # INFO: 2. inference
            outputs = self._model(**inputs, output_hidden_states=True).hidden_states

        hidden: torch.Tensor = outputs[-1]  # shape (batch, seqMax, eDim)
        batchSize, maxSeqLen, hiddenDim = hidden.shape

        # sanity
        assert batchSize == len(batch)
        assert maxSeqLen == self.maxSeqLen
        assert hiddenDim == self.embedDim

        # INFO: 3. mean pooling

        # [ 1, 2, 3 ]
        seqLens = torch.Tensor([len(item) for item in batch]).to(dev)
        # -> [ [1], [2], [3] ]
        seqLens = seqLens.unsqueeze(dim=1)
        # [ 0 1 2 3 4 ... maxLen ]
        indices = torch.arange(maxSeqLen, device=dev)
        # [ [10000...], [11000...], [11100...] ]
        mask = (indices < seqLens).float()
        # to match the hidden dimension along the sequence length
        mask = mask.unsqueeze(-1).expand(hidden.shape)

        # zero values corresponding to padded regions
        hidden = hidden * mask
        # mean aggregation, removes seqMax dimension
        hidden = torch.sum(hidden, dim=1)
        # clamp ensures no divide by zero issue
        hidden /= torch.clamp(seqLens, min=1e-9)
        return hidden  # pyright: ignore[reportReturnType]


# ===================
#    Region2Vec
# ===================


# @final
# class _RawRegion2VecModel(torch.nn.Module):
#     def __init__(self):
#         modelName = "databio/r2v-encode-hg38"
#         self.model = Region2VecExModel(modelName)
#         super().__init__()
#
#     @override
#     def forward(self, x: Any):
#         x = list(zip(*x))
#         x = [Region(*i) for i in x]
#         embed = self.model.encode(x)
#         return torch.tensor(embed)


# @final
# class Region2Vec(EmbedModel):
#     wants = "intervals"
#
#     def __init__(self):
#         self.maxSeqLen: int | None = None
#         self.embedDim: int = 100
#
#     @cached_property
#     def _model(self):
#         model = _RawRegion2VecModel()
#         model.eval()  # probably irrelevant with regard to R2V
#         return model
#
#     @override
#     def to(self, device: Device) -> Self:
#         # CPU only
#         return self
#
#     @override
#     def embed(self, batch: Sequence[GenomicInterval]):
#         return self._model(batch)


# ===================
#    CountACGT
# ===================


@final
class CountACGT(EmbedModel):
    """
    Developed for testing purposes. Has an arbitrary max sequence length of 10.
    Creates a 4 dimensional embedding vector for counts of ACGT
    respectively. Embedding conforms to f32 tensor.
    """

    wants = "sequences"

    def __init__(self, maxSeqLen: int = 10):
        self.maxSeqLen = maxSeqLen
        self.embedDim = 4

    @override
    def to(self, device: Device):
        # CPU only
        return self

    @override
    def embed(self, batch: Sequence[str]):
        results = list()

        for item in batch:
            if self.maxSeqLen is not None and len(item) > self.maxSeqLen:
                raise ValueError("Sequence exceeds max length; refusing to truncate.")

            counts = [0, 0, 0, 0]

            for char in item:
                if char == "A":
                    counts[0] += 1
                elif char == "C":
                    counts[1] += 1
                elif char == "G":
                    counts[2] += 1
                elif char == "T":
                    counts[3] += 1
            results.append(counts)
        return torch.FloatTensor(results)


@final
class TrivialModel(EmbedModel):
    wants = "intervals"

    def __init__(self, maxSeqLen: int = 10):
        self.maxSeqLen = maxSeqLen
        self.embedDim = 1

    @override
    def to(self, device: Device):
        # CPU only
        return self

    @override
    def embed(self, batch: Sequence[GenomicInterval]) -> torch.FloatTensor:
        results = list()

        for item in batch:
            if self.maxSeqLen is not None and len(item) > self.maxSeqLen:
                raise ValueError("Sequence exceeds max length; refusing to truncate.")

            _, start, end = item
            results.append(end - start)

        return cast(torch.FloatTensor, torch.FloatTensor(results).unsqueeze(-1))
