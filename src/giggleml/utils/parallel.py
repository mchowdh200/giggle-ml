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
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


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


def _setup_dist_env(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: str,
    backend: str,
) -> None:
    """Set up distributed environment variables and initialize process group."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def _cleanup_dist() -> None:
    """Clean up the distributed process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _worker_wrapper(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: str,
    backend: str,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Wrapper function that sets up distributed environment for each worker process."""
    _setup_dist_env(rank, world_size, master_addr, master_port, backend)

    try:
        fn(*args, **kwargs)
    finally:
        _cleanup_dist()


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
        # Infer parameters if not provided
        if world_size is None:
            world_size = _infer_world_size()

        if backend is None:
            backend = _infer_backend()

        if master_port is None:
            # Find a free port automatically
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                master_port = str(s.getsockname()[1])

        # Validation guards
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")

        if world_size > 1024:
            raise ValueError(
                f"world_size too large: {world_size}. Maximum supported is 1024"
            )

        if backend not in {"gloo", "nccl", "mpi"}:
            raise ValueError(
                f"Unsupported backend: {backend}. Must be one of: gloo, nccl, mpi"
            )

        # Backend-specific validation
        if backend == "nccl" and not torch.cuda.is_available():
            raise ValueError("NCCL backend requires CUDA, but CUDA is not available")

        self.world_size = world_size
        self.backend = backend
        self.master_addr = master_addr
        self.master_port = master_port

    def _execute[T](
        self, fn: Callable[..., T], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> T | None:
        """Internal execution logic shared by decorator and .run()"""
        has_accelerators = torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

        if self.world_size == 1 and not has_accelerators:
            print("Running locally (world_size=1, no accelerators)")
            _setup_dist_env(0, 1, self.master_addr, self.master_port, self.backend)

            try:
                result = fn(*args, **kwargs)
            finally:
                _cleanup_dist()
            return result
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
