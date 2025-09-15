import os
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from time import time
from typing import Callable, final, overload

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch import multiprocessing as mp
from torch.utils.data import DataLoader

from giggleml.data_wrangling import fasta
from giggleml.data_wrangling.unified_dataset import UnifiedDataset

from ..data_wrangling.interval_dataset import IntervalDataset
from ..utils.guess_device import guess_device
from ..utils.types import GenomicInterval
from .block_distributed_sampler import BlockDistributedSampler
from .embed_model import EmbedModel


@final
class FastaCollate:
    def __init__(self, fasta: Path):
        self.fasta = fasta

    def __call__(self, batch: Sequence[GenomicInterval]):
        return fasta.map(batch, self.fasta)


def pass_collate(batch: Sequence[GenomicInterval]):
    return batch


@final
class BatchInfer:
    def __init__(
        self,
        model: EmbedModel,
        batch_size: int,
        worker_count: int,
        sub_worker_count: int,
    ):
        """
        @param workerCount: should be <= gpu count
        @param subWorkerCount: corresponds to pytorch::DataLoader::num_worker
        argument -- used to prepare subprocesses for batch construction. Can be
        zero.
        """

        if worker_count == 0:
            raise ValueError("No workers; no work.")

        self.model = model
        self.batch_size = batch_size
        self.worker_count = worker_count
        self.sub_worker_count = sub_worker_count
        self.embed_dim = model.embed_dim

    @overload
    def _infer_loop(self, rank: int, data_loader: DataLoader) -> Tensor: ...

    @overload
    def _infer_loop(
        self, rank: int, data_loader: DataLoader, out_file: np.memmap
    ) -> None: ...

    def _infer_loop(
        self, rank: int, data_loader: DataLoader, out_file: np.memmap | None = None
    ) -> Tensor | None:
        """inference loop."""
        rprint = lambda *args: print(f"[{rank}]:", *args)

        time0 = time()
        next_idx = 0
        in_memory = out_file is None
        in_memory_results = []

        for i, batch in enumerate(data_loader):
            outputs = self.model.embed(batch).to("cpu")
            final_idx = next_idx + len(outputs)

            if in_memory:
                in_memory_results.append(outputs)
            else:
                assert out_file is not None
                assert final_idx <= len(out_file)
                out_file[next_idx:final_idx] = outputs.detach().numpy()

            next_idx += len(outputs)

            if i % 50 == 0:
                rprint(f"Batch: {i + 1}\t/ {len(data_loader)}")
            if i % 150 == 1 and rank == 0:
                elapsed = time() - time0
                eta = timedelta(
                    seconds=((elapsed / (i + 1)) * len(data_loader)) - elapsed
                )
                elapsed_dt = timedelta(seconds=elapsed)
                rprint(f"== {str(elapsed_dt)}, ETA: {str(eta)}")

        rprint(f"Batch: {len(data_loader)}\t/ {len(data_loader)}")

        if in_memory:
            return (
                torch.cat(in_memory_results, dim=0)
                if in_memory_results
                else torch.tensor([])
            )
        else:
            # "close" the memmap
            assert out_file is not None
            out_file.flush()
            del out_file
            return None

    def _worker(
        self,
        rank: int,
        datasets: Sequence[IntervalDataset],
        out_paths: Sequence[str] | None,
        post: Sequence[Callable[[np.memmap], None]] | None,
    ) -> list[Tensor] | None:
        device = guess_device(rank)

        if rank == 0:
            print("Starting inference.")

        rprint = lambda *args: print(f"[{rank}]:", *args)

        fasta: Path | None = None

        for dataset in datasets:
            dataset_fasta = dataset.associated_fasta_path

            if fasta is None:
                fasta = dataset_fasta
            elif fasta != dataset_fasta:
                # TODO: this constraint could be avoided
                raise ValueError("Expecting all datasets to have same fasta path")

        master_dataset = UnifiedDataset[GenomicInterval](datasets)

        model = self.model.to(device)
        model.to(device)
        wants_seq = self.model.wants == "sequences"
        e_dim = self.model.embed_dim

        if wants_seq:
            if fasta is None:
                raise ValueError("Unable to map to fasta; missing associatedFastaPath")
            collate = FastaCollate(fasta)
        else:
            collate = pass_collate

        # INFO: In-memory only case:
        # processes the entire input disregarded potential concurrent workers
        in_memory = out_paths is None

        if in_memory:
            data_loader = DataLoader(
                master_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.sub_worker_count,
                persistent_workers=self.sub_worker_count != 0,
                collate_fn=collate,
            )

            embeddings = self._infer_loop(rank, data_loader)

            # Split embeddings back into separate tensors for each dataset
            results = []
            start_idx = 0
            for dataset in datasets:
                end_idx = start_idx + len(dataset)
                results.append(embeddings[start_idx:end_idx])
                start_idx = end_idx

            return results

        # INFO: Distributed memmap case:
        # assumes a series of concurrent workers

        block_sampler = BlockDistributedSampler(master_dataset, self.worker_count, rank)
        sample_count = len(block_sampler)
        offset = block_sampler.lower * e_dim * 4  # (4 bytes per float32)
        master_out_path = Path(out_paths[0]).parent / "wip.tmp.npy"
        master_out_file = np.memmap(
            master_out_path, np.float32, "r+", offset, shape=(sample_count, e_dim)
        )

        data_loader = DataLoader(
            master_dataset,
            batch_size=self.batch_size,
            sampler=block_sampler,
            shuffle=False,
            pin_memory=True,
            num_workers=self.sub_worker_count,
            persistent_workers=self.sub_worker_count != 0,  # (usually True) -- crucial
            collate_fn=collate,
        )

        try:
            # INFO: init process group

            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12356"
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                rank=rank,
                world_size=self.worker_count,
                # it seems to be about 1.2 seconds per batch 5/27/2025
                timeout=timedelta(
                    seconds=len(master_dataset) / self.worker_count * 0.5
                ),
                device_id=(device if device.type == "cuda" else None),
            )

            if not dist.is_initialized():
                raise RuntimeError("Process group could not initialized")

            dist.barrier()

            # INFO: big step

            self._infer_loop(
                rank,
                data_loader,
                master_out_file,
            )

            dist.barrier()  # INFO: inference complete.
            #                 now beginning separation of outputs
            if rank == 0:
                print("Starting post-processing.")

            # we are splitting up embeds from masterOutFile into separate memmaps
            # based on the distribution of datasets. Each worker is responsible for
            # the datasets they implicitly completed.

            set_idx_start = (
                master_dataset.list_idx_of(block_sampler.lower)
                if block_sampler.lower < len(master_dataset)
                else len(master_dataset.lists)
            )
            set_idx_end = (
                master_dataset.list_idx_of(block_sampler.upper)
                if block_sampler.upper < len(master_dataset)
                else len(master_dataset.lists)
            )

            #                                     (1)     (2)     (3)
            # for datasets full of intervals,   [----] [-------] [---]
            # the sampler can split datasets:   |==========|=========|
            # so memmap init works like:        (rank1)(    rank2    )

            # since sometimes a rank might need to write some intervals preparred by
            # rank-1, we need to remap masterOutFile
            master_out_file.flush()
            del master_out_file
            master_size = (
                master_dataset.sums[set_idx_end] - master_dataset.sums[set_idx_start]
            )
            master_out_file = np.memmap(
                master_out_path,
                np.float32,
                "r",
                offset=master_dataset.sums[set_idx_start] * e_dim * 4,
                shape=(master_size, e_dim),
            )

            for set_idx in range(set_idx_start, set_idx_end):
                size = len(datasets[set_idx])
                out_path = out_paths[set_idx]

                # reorganize data
                i = master_dataset.sums[set_idx] - master_dataset.sums[set_idx_start]
                content = master_out_file[i : i + size]
                mmap = np.memmap(out_path, np.float32, "w+", shape=(size, e_dim))
                mmap[:] = content
                mmap.flush()

                if post is not None:
                    # post processing
                    post_call = post[set_idx]
                    post_call(mmap)

                if rank == 0:
                    if (100 * set_idx // len(out_paths)) % 10 == 0:  # roughly every 10%
                        rprint(f"{set_idx + 1} / {len(out_paths)}")

            rprint(f"{len(out_paths)} / {len(out_paths)}")
            dist.barrier()
        finally:
            dist.destroy_process_group()

        return None

    @overload
    def batch(
        self,
        datasets: Sequence[IntervalDataset],
        out_paths: Sequence[str],
        post: Sequence[Callable[[np.memmap], None]] | None = None,
    ) -> None: ...

    @overload
    def batch(
        self,
        datasets: Sequence[IntervalDataset],
    ) -> list[Tensor]: ...

    def batch(
        self,
        datasets: Sequence[IntervalDataset],
        out_paths: Sequence[str] | None = None,
        post: Sequence[Callable[[np.memmap], None]] | None = None,
    ) -> list[Tensor] | None:
        """
        @param post: Is a list of processing Callable to apply to completed memmaps after inference is completed.
        """

        in_memory = out_paths is None

        if in_memory:
            if out_paths is not None or post is not None:
                raise ValueError(
                    "In-memory mode does not support outPaths or post processing"
                )
            return self._worker(0, datasets, None, None)

        assert len(datasets) == len(out_paths)

        e_dim = self.model.embed_dim
        total_len = sum([len(dataset) for dataset in datasets])
        master_out = Path(out_paths[0]).parent.mkdir(parents=True, exist_ok=True)
        master_out = Path(out_paths[0]).parent / "wip.tmp.npy"
        mm_total = np.memmap(master_out, np.float32, "w+", shape=(total_len, e_dim))
        mm_total[:] = 0
        mm_total.flush()
        del mm_total

        # big operation
        args = (datasets, out_paths, post)
        mp.spawn(
            self._worker, args=args, nprocs=self.worker_count, join=True
        )  # spawn method

        # working file space
        os.remove(master_out)
        return None
