import os
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from time import time
from typing import Callable, final, overload

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

        for i, batch in enumerate(dataLoader):
            outputs = self.model.embed(batch).to("cpu")
            finalIdx = nextIdx + len(outputs)

            assert finalIdx <= len(outFile)

            outFile[nextIdx:finalIdx] = outputs.detach().numpy()
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

    def _inferLoop_inmemory(self, rank: int, dataLoader: DataLoader) -> np.ndarray:
        """Inference loop that returns results in memory."""
        rprint = lambda *args: print(f"[{rank}]:", *args)
        time0 = time()

        all_outputs = []
        for i, batch in enumerate(dataLoader):
            outputs = self.model.embed(batch).to("cpu")
            all_outputs.append(outputs.detach().numpy())

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

        if not all_outputs:
            return np.empty((0, self.embedDim), dtype=np.float32)

        return np.concatenate(all_outputs, axis=0)

    def _worker(
        self,
        rank: int,
        datasets: Sequence[IntervalDataset],
        outPaths: Sequence[str] | None,
        post: Sequence[Callable[[np.memmap], None]] | None,
        world_size_override: int | None = None,
    ) -> np.ndarray | None:
        device = guessDevice(rank)
        in_memory = outPaths is None
        world_size = world_size_override or self.workerCount

        if rank == 0:
            print("Starting inference.")

        rprint = lambda *args: print(f"[{rank}]:", *args)

        fasta_path: str | None = None

        for dataset in datasets:
            datasetFasta = dataset.associatedFastaPath
            if fasta_path is None:
                fasta_path = datasetFasta
            elif fasta_path != datasetFasta:
                # TODO: this constraint could be avoided
                raise ValueError("Expecting all datasets to have same fasta path")

        masterDataset = UnifiedDataset[GenomicInterval](datasets)

        model = self.model.to(device)
        wantsSeq = self.model.wants == "sequences"
        eDim = self.model.embedDim

        if wantsSeq:
            if fasta_path is None:
                raise ValueError("Unable to map to fasta; missing associatedFastaPath")
            collate = FastaCollate(fasta_path)
        else:
            collate = passCollate

        pg_managed_here = False
        if not dist.is_initialized():
            pg_managed_here = True
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12356"
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=len(masterDataset) / world_size * 0.5),
                device_id=(device if device.type == "cuda" else None),
            )

        try:
            if not dist.is_initialized():
                raise RuntimeError("Process group could not be initialized")

            dist.barrier()

            blockSampler = BlockDistributedSampler(masterDataset, world_size, rank)
            dataLoader = DataLoader(
                masterDataset,
                batch_size=self.batchSize,
                sampler=blockSampler,
                shuffle=False,
                pin_memory=True,
                num_workers=self.subWorkerCount,
                persistent_workers=self.subWorkerCount != 0,
                collate_fn=collate,
            )

            if in_memory:
                if rank == 0:
                    print("Starting in-memory inference.")
                embeddings = self._inferLoop_inmemory(rank, dataLoader)
                dist.barrier()
                return embeddings
            else:
                assert outPaths is not None
                sampleCount = len(blockSampler)
                offset = blockSampler.lower * eDim * 4
                masterOutPath = Path(outPaths[0]).parent / "wip.tmp.npy"
                masterOutFile = np.memmap(
                    masterOutPath, np.float32, "r+", offset, shape=(sampleCount, eDim)
                )

                self._inferLoop(rank, dataLoader, masterOutFile)
                dist.barrier()

                if rank == 0:
                    print("Starting post-processing.")

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

                masterOutFile.flush()
                del masterOutFile
                masterSize = (
                    masterDataset.sums[setIdxEnd] - masterDataset.sums[setIdxStart]
                )
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
                    i = masterDataset.sums[setIdx] - masterDataset.sums[setIdxStart]
                    content = masterOutFile[i : i + size]
                    mmap = np.memmap(outPath, np.float32, "w+", shape=(size, eDim))
                    mmap[:] = content
                    mmap.flush()

                    if post is not None:
                        postCall = post[setIdx]
                        postCall(mmap)

                    if rank == 0 and (100 * setIdx // len(outPaths)) % 10 == 0:
                        rprint(f"{setIdx + 1} / {len(outPaths)}")

                rprint(f"{len(outPaths)} / {len(outPaths)}")
                dist.barrier()
                return None
        finally:
            if pg_managed_here and dist.is_initialized():
                dist.destroy_process_group()

    @overload
    def batch(
        self,
        datasets: Sequence[ListLike],
        outPaths: None = None,
        post: Sequence[Callable[[np.memmap], None]] | None = None,
    ) -> list[np.ndarray]: ...

    @overload
    def batch(
        self,
        datasets: Sequence[ListLike],
        outPaths: Sequence[str],
        post: Sequence[Callable[[np.memmap], None]] | None = None,
    ) -> None: ...

    def batch(
        self,
        datasets: Sequence[ListLike],
        outPaths: Sequence[str] | None = None,
        post: Sequence[Callable[[np.memmap], None]] | None = None,
    ) -> list[np.ndarray] | None:
        if outPaths is None:
            if not dist.is_initialized():
                raise RuntimeError(
                    "Process group must be initialized for in-memory batch inference."
                )

            rank = dist.get_rank()
            world_size = dist.get_world_size()

            if self.workerCount != world_size:
                print(
                    f"Warning: workerCount ({self.workerCount}) does not match distributed world size ({world_size}). Using world size."
                )
                self.workerCount = world_size

            my_embeddings = self._worker(
                rank, datasets, None, None, world_size_override=world_size
            )

            gathered_embeddings = [None] * world_size
            dist.all_gather_object(gathered_embeddings, my_embeddings)

            master_embeddings = np.concatenate(gathered_embeddings, axis=0)
            masterDataset = UnifiedDataset[GenomicInterval](datasets)
            result_arrays = []
            for i in range(len(datasets)):
                start_index = masterDataset.sums[i]
                end_index = masterDataset.sums[i + 1]
                dataset_embedding = master_embeddings[start_index:end_index]
                result_arrays.append(dataset_embedding)

            return result_arrays
        else:
            assert len(datasets) == len(outPaths)
            eDim = self.model.embedDim
            totalLen = sum([len(dataset) for dataset in datasets])
            masterOut = Path(outPaths[0]).parent
            masterOut.mkdir(parents=True, exist_ok=True)
            masterOut_path = masterOut / "wip.tmp.npy"
            mmTotal = np.memmap(
                masterOut_path, np.float32, "w+", shape=(totalLen, eDim)
            )
            mmTotal[:] = 0
            mmTotal.flush()
            del mmTotal

            args = (datasets, outPaths, post)
            mp.spawn(self._worker, args=args, nprocs=self.workerCount, join=True)

            os.remove(masterOut_path)
            return None

