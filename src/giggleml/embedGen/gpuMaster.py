import os
from collections.abc import Iterable, Sequence
from typing import Callable, final

import numpy as np
import torch
import torch.distributed as dist
from torch import multiprocessing as mp
from torch.utils.data import DataLoader

from giggleml.dataWrangling.unifiedDataset import UnifiedDataset

from ..dataWrangling import fasta
from ..dataWrangling.intervalDataset import IntervalDataset
from ..utils.guessDevice import guessDevice
from ..utils.types import GenomicInterval, ListLike
from .blockDistributedSampler import BlockDistributedSampler
from .embedModel import EmbedModel


@final
class FastaCollate:
    def __init__(self, fasta: str):
        self.fasta = fasta

    def __call__(self, batch: Sequence[GenomicInterval]):
        return fasta.map(batch, self.fasta)


def passCollate(batch: Sequence[GenomicInterval]):
    return batch


@final
class GpuMaster:
    def __init__(self, model: EmbedModel, batchSize: int, workerCount: int, subWorkerCount: int):
        """
        @param workerCount: should be <= gpu count
        @param subWorkerCount: corresponds to pytorch::DataLoader::num_worker
        argument -- used to prepare subprocesses for batch construction. Can be
        zero.
        """

        if workerCount == 0:
            raise ValueError("No workers; no work.")

        self.model = model
        self.batchSize = batchSize
        self.workerCount = workerCount
        self.subWorkerCount = subWorkerCount
        self.embedDim = model.embedDim

    def _inferLoop(self, rank: int, dataLoader: DataLoader, outFile: np.memmap):
        """inference loop."""
        rprint = lambda *args: print(f"[{rank}]:", *args)

        nextIdx = 0

        for i, batch in enumerate(dataLoader):
            outputs = self.model.embed(batch).to("cpu", non_blocking=True)
            finalIdx = nextIdx + len(outputs)

            assert finalIdx <= len(outFile)

            if i >= 4:
                break

            outFile[nextIdx:finalIdx] = outputs.numpy()
            nextIdx += len(outputs)

            if i % 50 == 0:
                rprint(f"Batch: {i + 1}\t/ {len(dataLoader)}")

        # "close" the memmap
        outFile.flush()
        del outFile

    def _worker(
        self,
        rank: int,
        datasets: Sequence[IntervalDataset],
        outPaths: Sequence[str],
        post: Sequence[Callable[[np.memmap], None]] | None,
    ):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12356"
        backendType = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backendType, rank=rank, world_size=self.workerCount)

        if not dist.is_initialized():
            raise RuntimeError("Process group could not initialized")

        if rank == 0:
            print("Starting inference.")

        rprint = lambda *args: print(f"[{rank}]:", *args)

        device = guessDevice(rank)
        model = self.model.to(device)
        model.to(device)
        wantsSeq = self.model.wants == "sequences"
        eDim = self.model.embedDim

        for i, dataset, outPath in zip(range(len(datasets)), datasets, outPaths):
            if wantsSeq:
                if dataset.associatedFastaPath is None:
                    raise ValueError("Unable to map to fasta; missing associatedFastaPath")
                collate = FastaCollate(dataset.associatedFastaPath)
            else:
                collate = passCollate

            blockSampler = BlockDistributedSampler(dataset, self.workerCount, rank)
            sampleCount = len(blockSampler)
            offset = blockSampler.lower * eDim * 4  # (4 bytes per float32)
            outFile = np.memmap(outPath, np.float32, "r+", offset, shape=(sampleCount, eDim))

            dataLoader = DataLoader(
                dataset,
                batch_size=self.batchSize,
                sampler=blockSampler,
                shuffle=False,
                pin_memory=True,
                num_workers=self.subWorkerCount,
                persistent_workers=self.subWorkerCount != 0,  # (usually True) -- crucial
                collate_fn=collate,
            )

            self._inferLoop(
                rank,
                dataLoader,
                outFile,
            )

            if rank == 0:
                rprint(f"\t- Dataset {i+1} / {len(datasets)}")

        dist.barrier()  # INFO: inference complete

        if rank == 0:
            if post is not None:
                print("Starting post-processing.")

                for i, (dataset, outPath, postCall) in enumerate(zip(datasets, outPaths, post)):
                    if (100 * i // len(outPaths)) % 10 == 0:  # roughly every 10%
                        rprint(f"{i+1} / {len(outPaths)}")

                    size = len(dataset)
                    outFile = np.memmap(outPath, np.float32, "r", shape=(size, eDim))
                    postCall(outFile)

                rprint(f"{len(outPaths)} / {len(outPaths)}")

        dist.barrier()
        dist.destroy_process_group()

    def batch(
        self,
        datasets: Sequence[ListLike],
        outPaths: Sequence[str],
        post: Sequence[Callable[[np.memmap], None]] | None = None,
    ):
        """
        @param post: Is a list of processing Callable to apply to completed memmaps after inference is completed.
        """

        assert len(datasets) == len(outPaths)

        eDim = self.model.embedDim

        # necessary to initialize output space before workers map to specific regions
        for dataset, outPath in zip(datasets, outPaths):
            outFile = np.memmap(outPath, np.float32, "w+", shape=(len(dataset), eDim))
            outFile.flush()
            del outFile

        # big operation
        args = (datasets, outPaths, post)
        mp.spawn(self._worker, args=args, nprocs=self.workerCount, join=True)  # spawn method
