import os
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from time import time
from typing import Callable, final

import numpy as np
import torch
import torch.distributed as dist
from torch import multiprocessing as mp
from torch.utils.data import DataLoader

from giggleml.dataWrangling import fasta
from giggleml.dataWrangling.unifiedDataset import UnifiedDataset

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
class BatchInfer:
    def __init__(
        self, model: EmbedModel, batchSize: int, workerCount: int, subWorkerCount: int
    ):
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

        time0 = time()
        nextIdx = 0

        with torch.no_grad():
            for i, batch in enumerate(dataLoader):
                outputs = self.model.embed(batch).to("cpu")
                finalIdx = nextIdx + len(outputs)

                assert finalIdx <= len(outFile)

                outFile[nextIdx:finalIdx] = outputs.numpy()
                nextIdx += len(outputs)

                if i % 50 == 0:
                    rprint(f"Batch: {i + 1}\t/ {len(dataLoader)}")
                if i % 150 == 1 and rank == 0:
                    elapsed = time() - time0
                    eta = timedelta(
                        seconds=((elapsed / (i + 1)) * len(dataLoader)) - elapsed
                    )
                    elapsedDt = timedelta(seconds=elapsed)
                    rprint(f"== {str(elapsedDt)}, ETA: {str(eta)}")

        rprint(f"Batch: {len(dataLoader)}\t/ {len(dataLoader)}")

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
        device = guessDevice(rank)

        if rank == 0:
            print("Starting inference.")

        rprint = lambda *args: print(f"[{rank}]:", *args)

        fasta: str | None = None

        for dataset in datasets:
            datasetFasta = dataset.associatedFastaPath
            if fasta is None:
                fasta = datasetFasta
            elif fasta != datasetFasta:
                # TODO: this constraint could be avoided
                raise ValueError("Expecting all datasets to have same fasta path")

        masterDataset = UnifiedDataset[GenomicInterval](datasets)

        model = self.model.to(device)
        model.to(device)
        wantsSeq = self.model.wants == "sequences"
        eDim = self.model.embedDim

        if wantsSeq:
            if fasta is None:
                raise ValueError("Unable to map to fasta; missing associatedFastaPath")
            collate = FastaCollate(fasta)
        else:
            collate = passCollate

        # with torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(f"logs/rank_{rank}"),
        #     with_stack=True,
        # ) as prof:
        blockSampler = BlockDistributedSampler(masterDataset, self.workerCount, rank)
        sampleCount = len(blockSampler)
        offset = blockSampler.lower * eDim * 4  # (4 bytes per float32)
        masterOutPath = Path(outPaths[0]).parent / "wip.tmp.npy"
        masterOutFile = np.memmap(
            masterOutPath, np.float32, "r+", offset, shape=(sampleCount, eDim)
        )

        dataLoader = DataLoader(
            masterDataset,
            batch_size=self.batchSize,
            sampler=blockSampler,
            shuffle=False,
            pin_memory=True,
            num_workers=self.subWorkerCount,
            persistent_workers=self.subWorkerCount != 0,  # (usually True) -- crucial
            collate_fn=collate,
        )

        # INFO: init process group

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12356"
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=rank,
            world_size=self.workerCount,
            # it seems to be about 1.2 seconds per batch 5/27/2025
            timeout=timedelta(seconds=len(masterDataset) / self.workerCount * 0.5),
            device_id=(device if device.type == "cuda" else None),
        )

        if not dist.is_initialized():
            raise RuntimeError("Process group could not initialized")

        dist.barrier()

        # INFO: big step

        self._inferLoop(
            rank,
            dataLoader,
            masterOutFile,
        )

        dist.barrier()  # INFO: inference complete.
        #                 now beginning separation of outputs
        if rank == 0:
            print("Starting post-processing.")

        # we are splitting up embeds from masterOutFile into separate memmaps
        # based on the distribution of datasets. Each worker is responsible for
        # the datasets they implicitly completed.

        setIdxStart = (
            masterDataset.listIdxOf(blockSampler.lower)
            if blockSampler.lower < len(masterDataset)
            else len(masterDataset.lists)
        )
        setIdxEnd = (
            masterDataset.listIdxOf(blockSampler.upper)
            if blockSampler.upper < len(masterDataset)
            else len(masterDataset.lists)
        )

        #                                     (1)     (2)     (3)
        # for datasets full of intervals,   [----] [-------] [---]
        # the sampler can split datasets:   |==========|=========|
        # so memmap init works like:        (rank1)(    rank2    )

        # since sometimes a rank might need to write some intervals preparred by
        # rank-1, we need to remap masterOutFile
        masterOutFile.flush()
        del masterOutFile
        masterSize = masterDataset.sums[setIdxEnd] - masterDataset.sums[setIdxStart]
        masterOutFile = np.memmap(
            masterOutPath,
            np.float32,
            "r",
            offset=masterDataset.sums[setIdxStart] * eDim * 4,
            shape=(masterSize, eDim),
        )

        for setIdx in range(setIdxStart, setIdxEnd):
            size = len(datasets[setIdx])
            outPath = outPaths[setIdx]

            # reorganize data
            i = masterDataset.sums[setIdx] - masterDataset.sums[setIdxStart]
            content = masterOutFile[i : i + size]
            mmap = np.memmap(outPath, np.float32, "w+", shape=(size, eDim))
            mmap[:] = content
            mmap.flush()

            if post is not None:
                # post processing
                postCall = post[setIdx]
                postCall(mmap)

            if rank == 0:
                if (100 * setIdx // len(outPaths)) % 10 == 0:  # roughly every 10%
                    rprint(f"{setIdx + 1} / {len(outPaths)}")

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
        totalLen = sum([len(dataset) for dataset in datasets])
        masterOut = Path(outPaths[0]).parent.mkdir(parents=True, exist_ok=True)
        masterOut = Path(outPaths[0]).parent / "wip.tmp.npy"
        mmTotal = np.memmap(masterOut, np.float32, "w+", shape=(totalLen, eDim))
        mmTotal[:] = 0
        mmTotal.flush()
        del mmTotal

        # big operation
        args = (datasets, outPaths, post)
        mp.spawn(
            self._worker, args=args, nprocs=self.workerCount, join=True
        )  # spawn method

        # working file space
        os.remove(masterOut)
