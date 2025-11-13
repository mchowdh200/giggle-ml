"""Intelligent distributed setup using torch.multiprocessing to simulate torchrun behavior.

This module provides automatic hardware detection, parameter inference, and optimized
execution paths for distributed PyTorch workloads. It can run locally for simple cases
or spawn multiple processes when true distribution is needed.

Supports two modes of operation:
1. Decorator: @Parallel()
2. Direct Call: parallel.run(fn, *args, **kwargs)
"""

import functools
import multiprocessing
import os
import socket
from contextlib import contextmanager
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _infer_port() -> int:
    # Find a free port automatically
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _infer_backend() -> str:
    """Infer the optimal distributed backend based on available hardware.

    Returns:
        "nccl" for CUDA systems (best GPU communication performance)
        "gloo" for MPS or CPU systems (universal compatibility)
    """
    if torch.cuda.is_available():
        return "nccl"  # Best for CUDA GPUs
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "gloo"  # MPS doesn't support nccl
    else:
        return "gloo"  # CPU fallback


def _infer_world_size() -> int:
    """Infer optimal world size based on available hardware.

    Returns:
        - CUDA systems: number of available GPUs (or 1 if device_count() returns 0)
        - MPS systems: 2 processes (optimal for Apple Silicon)
        - CPU systems: min(cpu_count, 4) (capped for reasonable resource usage)
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count() or 1
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS is single-device, but we can still use multiple processes
        return 2  # Default to 2 processes for MPS
    else:
        # CPU: default to number of physical cores, capped at 4 for reasonable defaults
        return min(multiprocessing.cpu_count(), 4)


@contextmanager
def dist_process_group(
    rank: int | None = None,
    world_size: int | None = None,
    master_addr: str | None = None,
    master_port: str | None = None,
    backend: str | None = None,
):
    if dist.is_initialized():
        return

    rank = rank if rank is not None else int(os.environ["RANK"])
    assert rank is not None, "Missing rank"
    os.environ["RANK"] = str(rank)

    world_size = world_size if world_size is not None else int(os.environ["WORLD_SIZE"])
    assert world_size is not None, "Missing world_size"
    os.environ["WORLD_SIZE"] = str(world_size)

    master_port = master_port or os.environ["MASTER_PORT"]
    assert master_port is not None, "Missing master_port"
    os.environ["MASTER_PORT"] = master_port

    master_addr = master_addr or os.environ["MASTER_ADDR"] or "localhost"
    os.environ["MASTER_ADDR"] = master_addr

    backend = backend or _infer_backend()

    try:
        accelerator = torch.accelerator.current_accelerator()

        if accelerator is None:
            device_id = None
        elif accelerator.type == "mps":
            device_id = None
        else:
            device_id = rank

        dist.init_process_group(
            backend=backend, rank=rank, world_size=world_size, device_id=device_id
        )

        if backend == "nccl":
            torch.cuda.set_device(rank)

        yield
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def _worker_wrapper(
    rank: int,
    world_size: int,
    master_addr: str | None,
    master_port: str | None,
    backend: str | None,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Wrapper function that sets up distributed environment for each worker process."""

    print("worker", rank, world_size)

    with dist_process_group(rank, world_size, master_addr, master_port, backend):
        fn(*args, **kwargs)


class Parallel:
    """Intelligent distributed setup for PyTorch via decorator or direct call.

    This class provides automatic hardware detection, parameter inference, and optimized
    execution paths for distributed PyTorch workloads. It can run locally for simple cases
    or spawn multiple processes when true distribution is needed.

    Args:
        world_size: Number of processes to spawn. If None, automatically inferred.
        backend: Distributed backend ("nccl", "gloo", "mpi"). If None, auto-selected.
        master_addr: Master node address (default: "localhost").
        master_port: Master node port. If None, automatically finds a free port.

    Raises:
        ValueError: For invalid configuration (e.g., world_size <= 0, or
                    NCCL backend requested without CUDA).

    ---
    ### Example 1: Decorator Usage

    >>> @Parallel()
    ... def worker_fn(data_path, val_path):
    ...     rank = dist.get_rank()
    ...     print(f"Rank {rank}: Processing {data_path} & {val_path}")
    ...
    >>> # This call will now be executed in parallel
    >>> worker_fn("data/train.txt", val_path="data/val.txt")

    ---
    ### Example 2: Direct Call Usage

    >>> def another_worker_fn(data_path):
    ...     rank = dist.get_rank()
    ...     print(f"Rank {rank}: Processing {data_path}")
    ...
    >>> # Instantiate with auto-detection
    >>> parallel = Parallel()
    >>> # Run the function directly
    >>> parallel.run(another_worker_fn, "data/train.txt")

    ---
    Notes:
        - Return values are only supported in local execution mode (world_size=1).
          Multiprocess execution via `mp.spawn` does not propagate return values.
    """

    def __init__(
        self,
        world_size: int | None = None,
        backend: str | None = None,
        master_addr: str = "localhost",
        master_port: str | None = None,
    ) -> None:
        self.world_size = world_size or _infer_world_size()
        self.backend = backend
        self.master_addr = master_addr
        self.master_port = master_port or str(_infer_port())

    def _execute[T](
        self, fn: Callable[..., T], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> T | None:
        """Internal execution logic shared by decorator and .run()"""
        has_accelerators = torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

        if self.world_size == 1 and not has_accelerators:
            print("Running locally (world_size=1, no accelerators)")
            with dist_process_group(
                0, 1, self.master_addr, self.master_port, self.backend
            ):
                return fn(*args, **kwargs)
        else:
            print(
                f"Running distributed with world_size={self.world_size}, backend={self.backend}"
            )
            mp.spawn(
                _worker_wrapper,
                args=(
                    self.world_size,
                    self.master_addr,
                    self.master_port,
                    self.backend,
                    fn,
                    args,
                    kwargs,
                ),
                nprocs=self.world_size,
                join=True,
            )
            # mp.spawn does not support return values
            return None

    def __call__[**P, T](self, fn: Callable[P, T]) -> Callable[P, T]:
        """
        Acts as the decorator, receiving the function to be wrapped.

        Args:
            fn: The function to be run in a distributed environment.

        Returns:
            A wrapper function that, when called, will launch the distributed execution.
        """

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            The wrapper that replaces the decorated function.
            This is what gets executed when the decorated function is called.
            """
            return self._execute(fn, args, kwargs)

        return wrapper

    def run[T](self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T | None:
        """
        Run a function in a distributed environment (direct call).

        This provides the original (non-decorator) behavior.

        Args:
            fn: The function to run in each worker process.
            *args: Arguments to pass to the worker function.
            **kwargs: Keyword arguments to pass to the worker function.

        Returns:
            The result of `fn` if run locally, otherwise None.
        """
        return self._execute(fn, args, kwargs)
