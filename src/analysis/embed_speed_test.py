from collections.abc import Iterator
from typing import override

import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from giggleml.models.hyena_dna import HyenaDnaNoPooling
from giggleml.utils.print_utils import progress_logger


def fast():
    print("beginning")

    sample_count = int(60e6)
    edim = 256
    dtype = torch.float32
    batch_size = int(4e6)
    device = "cuda:0"

    print("using device", device)
    print("building inputs")
    input_samples = torch.rand(
        sample_count,
        1,
        dtype=dtype,
        pin_memory=True,
    ).repeat(1, edim)
    print("\t shape", input_samples.shape)

    outputs = torch.empty_like(input_samples, device=device)

    mlp = torch.nn.Sequential(
        torch.nn.Linear(edim, edim),
        torch.nn.Tanh(),
        torch.nn.Linear(edim, edim),
    ).to(device, dtype=input_samples.dtype)

    print("Warming up CUDA...")
    warmup_batch = torch.randn(batch_size, edim, device=device, dtype=dtype)
    mlp(warmup_batch)
    torch.cuda.synchronize()

    steps = range(0, len(input_samples), batch_size)

    with progress_logger(len(steps), "sprinting") as ckpt:
        with torch.no_grad():
            for start in steps:
                end = start + batch_size

                batch = input_samples[start:end]  # views into pinned memory are no-copy
                batch = batch.to(device, non_blocking=True)
                out = mlp(batch)  # keep on gpu
                outputs[start:end, :] = out

                ckpt()


# beginning
# using device cuda:0
# building inputs
#          shape torch.Size([60000000, 256])
# Warming up CUDA...
# [sprinting] 100.0% (15/15) | Elapsed: 7.1s | ETA: 0.0s
# [sprinting] Finished in 7.1s.

# ~8M/s on 1 A100-80G


def naive():
    print("beginning")

    sample_count = int(60e6)
    edim = 256
    dtype = torch.float32
    batch_size = int(4e6)
    device = "cuda:0"

    print("using device", device)
    print("building inputs")
    input_samples = torch.rand(
        sample_count,
        1,
        dtype=dtype,
        pin_memory=True,
    ).repeat(1, edim)
    print("\t shape", input_samples.shape)

    loader = DataLoader(
        TensorDataset(input_samples),
        batch_size,
    )

    outputs = torch.empty_like(input_samples, device=device)

    mlp = torch.nn.Sequential(
        torch.nn.Linear(edim, edim),
        torch.nn.Tanh(),
        torch.nn.Linear(edim, edim),
    ).to(device, dtype=input_samples.dtype)

    print("Warming up CUDA...")
    warmup_batch = torch.randn(batch_size, edim, device=device, dtype=dtype)
    mlp(warmup_batch)
    torch.cuda.synchronize()

    with progress_logger(len(loader), "sprinting") as ckpt:
        with torch.no_grad():
            for i, (batch,) in enumerate(loader):
                start = i * batch_size
                end = start + batch_size

                batch = batch.to(device)
                out = mlp(batch)  # keep on gpu
                outputs[start:end, :] = out

                ckpt()


# beginning
# using device cuda:0
# building inputs
#          shape torch.Size([60000000, 256])
# Warming up CUDA...
# [sprinting] 100.0% (15/15) | Elapsed: 04m 31s | ETA: 0.0s
# [sprinting] Finished in 04m 31s.

# ~200k/s


def with_dataloader_tensordataset():
    print("beginning")

    sample_count = int(60e6)
    edim = 256
    dtype = torch.float32
    batch_size = int(4e6)
    device = "cuda:0"

    print("using device", device)
    print("building inputs")
    input_samples = torch.rand(
        sample_count,
        1,
        dtype=dtype,
        pin_memory=True,
    ).repeat(1, edim)
    print("\t shape", input_samples.shape)

    loader = DataLoader(
        TensorDataset(input_samples),
        batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    outputs = torch.empty_like(input_samples, device=device)

    mlp = torch.nn.Sequential(
        torch.nn.Linear(edim, edim),
        torch.nn.Tanh(),
        torch.nn.Linear(edim, edim),
    ).to(device, dtype=input_samples.dtype)

    print("Warming up CUDA...")
    warmup_batch = torch.randn(batch_size, edim, device=device, dtype=dtype)
    mlp(warmup_batch)
    torch.cuda.synchronize()

    with progress_logger(len(loader), "sprinting") as ckpt:
        with torch.no_grad():
            for i, (batch,) in enumerate(loader):
                start = i * batch_size
                end = start + batch_size

                batch = batch.to(device, non_blocking=True)
                out = mlp(batch)  # keep on gpu
                outputs[start:end, :] = out

                ckpt()


# beginning
# using device cuda:0
# building inputs
#          shape torch.Size([60000000, 256])
# Warming up CUDA...
# [sprinting] 100.0% (15/15) | Elapsed: 04m 53s | ETA: 0.0s
# [sprinting] Finished in 04m 53s.

# 200k/s


class MyTensorDataset(IterableDataset):
    def __init__(self, tensor: torch.Tensor) -> None:
        super().__init__()
        self.tensor = tensor

    @override
    def __iter__(self) -> Iterator[torch.Tensor]:
        yield from self.tensor

    def __len__(self) -> int:
        return len(self.tensor)


def with_dataloader():
    print("beginning")

    sample_count = int(60e6)
    edim = 256
    dtype = torch.float32
    batch_size = int(4e6)
    device = "cuda:0"

    print("using device", device)
    print("building inputs")
    input_samples = torch.rand(
        sample_count,
        1,
        dtype=dtype,
        pin_memory=True,
    ).repeat(1, edim)
    print("\t shape", input_samples.shape)

    loader = DataLoader(
        MyTensorDataset(input_samples),
        batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    outputs = torch.empty_like(input_samples, device=device)

    mlp = torch.nn.Sequential(
        torch.nn.Linear(edim, edim),
        torch.nn.Tanh(),
        torch.nn.Linear(edim, edim),
    ).to(device, dtype=input_samples.dtype)

    print("Warming up CUDA...")
    warmup_batch = torch.randn(batch_size, edim, device=device, dtype=dtype)
    mlp(warmup_batch)
    torch.cuda.synchronize()

    with progress_logger(len(input_samples) // batch_size, "sprinting") as ckpt:
        with torch.no_grad():
            for i, batch in enumerate(loader):
                start = i * batch_size
                end = start + batch_size

                batch = batch.to(device, non_blocking=True)
                out = mlp(batch)  # keep on gpu
                outputs[start:end, :] = out

                ckpt()


# beginning
# using device cuda:0
# building inputs
#          shape torch.Size([60000000, 256])
# Warming up CUDA...
# [sprinting] 100.0% (15/15) | Elapsed: 02m 28s | ETA: 0.0s
# [sprinting] Finished in 02m 28s.

# 400k/s


def hdna():
    print("beginning")

    sample_count = int(20e3)
    edim = 128
    batch_size = int(256)
    device = "cuda:0"
    model = HyenaDnaNoPooling("16k").to(device).eval()
    seq_len = 16386

    print("using device", device)
    print("building inputs")
    # akin to pretokenization which appears to have been the main bottleneck
    input_samples = torch.full(
        (sample_count, seq_len), 7, dtype=torch.int64, pin_memory=True
    )
    input_samples[:, -1] = 1  # [SEP] token
    print(f"\t shape [{len(input_samples)}, {len(input_samples[0])}")
    outputs = torch.empty(sample_count, edim, device=device, dtype=torch.float16)

    with torch.inference_mode():
        print("Warming up CUDA...")
        model({"input_ids": input_samples[0].unsqueeze(0).to(device)})
        torch.cuda.synchronize()

        steps = range(0, len(input_samples), batch_size)

        with progress_logger(len(steps), "sprinting") as ckpt:
            for start in steps:
                end = start + batch_size

                raw = input_samples[start:end].to(device, non_blocking=True)
                batch = {"input_ids": raw}
                out = model(batch)[:, -1, :]  # keep on gpu
                outputs[start:end].copy_(out, non_blocking=True)

                ckpt()

            torch.cuda.synchronize()


# beginning
# using device cuda:0
# building inputs
#          shape [20000, 16386
# Warming up CUDA...
# /Users/siwa3657/opt/miniforge3/envs/gml3/lib/python3.12/site-packages/huggingface_hub/file_download.py:942: FutureWarning: `resume_download` is deprecated and will be removed
#  in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
#   warnings.warn(
# /Users/siwa3657/opt/miniforge3/envs/gml3/lib/python3.12/site-packages/transformers/utils/generic.py:309: FutureWarning: `torch.utils._pytree._register_pytree_node` is depreca
# ted. Please use `torch.utils._pytree.register_pytree_node` instead.
#   _torch_pytree._register_pytree_node(
# /Users/siwa3657/opt/miniforge3/envs/gml3/lib/python3.12/site-packages/transformers/utils/generic.py:309: FutureWarning: `torch.utils._pytree._register_pytree_node` is depreca
# ted. Please use `torch.utils._pytree.register_pytree_node` instead.
#   _torch_pytree._register_pytree_node(
# [sprinting] 100.0% (79/79) | Elapsed: 55.0s | ETA: 0.0s
# [sprinting] Finished in 55.0s.

# 364 items/s


if __name__ == "__main__":
    # fast()
    # naive()
    # with_dataloader_tensordataset()
    # with_dataloader()
    hdna()
    pass
