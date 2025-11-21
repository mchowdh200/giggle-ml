from collections.abc import Iterator
from typing import override

import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

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


if __name__ == "__main__":
    # fast()
    # with_dataloader_tensordataset()
    # with_dataloader()
    pass
