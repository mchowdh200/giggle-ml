"""Intelligent distributed setup using torch.multiprocessing to simulate torchrun behavior.

This module provides automatic hardware detection, parameter inference, and optimized
execution paths for distributed PyTorch workloads. It can run locally for simple cases
or spawn multiple processes when true distribution is needed.
"""

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
    *args: Any,
) -> None:
    """Wrapper function that sets up distributed environment for each worker process."""
    _setup_dist_env(rank, world_size, master_addr, master_port, backend)
    
    try:
        fn(*args)
    finally:
        _cleanup_dist()


class Parallel:
    """Intelligent distributed setup using torch.multiprocessing to simulate torchrun behavior.

    This class provides automatic hardware detection, parameter inference, and optimized
    execution paths for distributed PyTorch workloads. It can run locally for simple cases
    or spawn multiple processes when true distribution is needed.

    Execution Modes:
        - Local execution: world_size=1 + no accelerators → runs in current process
        - Multiprocess: world_size>1 or accelerators present → spawns worker processes

    Auto-Detection Features:
        - Hardware-aware backend selection (NCCL for CUDA, GLOO for CPU/MPS)
        - Intelligent world_size inference based on available GPUs/cores
        - Automatic port allocation to avoid conflicts
        - Comprehensive parameter validation with helpful error messages

    Args:
        world_size: Number of processes to spawn. If None, automatically inferred:
            - CUDA systems: number of available GPUs (torch.cuda.device_count())
            - MPS systems: 2 processes (optimal for Apple Silicon)
            - CPU systems: min(cpu_count, 4) (capped for performance)
        backend: Distributed backend to use. If None, automatically selected:
            - CUDA systems: "nccl" (optimal for GPU communication)
            - MPS/CPU systems: "gloo" (universal compatibility)
            - Supported options: "gloo", "nccl", "mpi"
        master_addr: Master node address for coordination (default: "localhost").
        master_port: Master node port. If None, automatically finds a free port.

    Raises:
        ValueError: If world_size <= 0, world_size > 1024, invalid backend,
                   or NCCL backend requested without CUDA.

    Examples:
        Basic usage with auto-detection:
        >>> def worker_fn(data_path):
        ...     rank = dist.get_rank()
        ...     world_size = dist.get_world_size()
        ...     print(f"Rank {rank}/{world_size}: Processing {data_path}")
        ...     # Your distributed code here...
        ...
        >>> # Automatically detects: backend, world_size, execution mode
        >>> parallel = Parallel()
        >>> parallel(worker_fn, "data/train.txt")

        Manual configuration:
        >>> # Force specific configuration
        >>> parallel = Parallel(world_size=4, backend="gloo")
        >>> parallel(worker_fn, "data/train.txt")

        CPU-only single process (runs locally):
        >>> # On CPU-only system with world_size=1 → local execution
        >>> parallel = Parallel(world_size=1)
        >>> parallel(worker_fn, "data/test.txt")

    Notes:
        - Worker functions have full access to torch.distributed API
        - Each process gets a unique rank from 0 to world_size-1
        - Distributed environment is automatically initialized and cleaned up
        - Local execution mode eliminates multiprocessing overhead when possible
        - All collective operations (all_reduce, all_gather, etc.) work in both modes
        - For multiprocess mode, the function must be importable (not defined in __main__)
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

    def __call__(self, fn: Callable[..., Any], *args: Any) -> None:
        """Run a function in a distributed environment.

        Args:
            fn: The function to run in each worker process. Will be called with the
                provided args in each worker. Must be pickle-able for multiprocess mode.
            *args: Arguments to pass to the worker function.
        """
        # Check if we can run locally instead of spawning processes
        has_accelerators = torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

        if self.world_size == 1 and not has_accelerators:
            print("Running locally (world_size=1, no accelerators)")
            _setup_dist_env(0, 1, self.master_addr, self.master_port, self.backend)
            
            try:
                fn(*args)
            finally:
                _cleanup_dist()
        else:
            print(f"Running distributed with world_size={self.world_size}, backend={self.backend}")
            # Spawn worker processes
            mp.spawn(
                _worker_wrapper,
                args=(self.world_size, self.master_addr, self.master_port, self.backend, fn, *args),
                nprocs=self.world_size,
                join=True,
            )


