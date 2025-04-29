import os
from collections.abc import Iterable, Sequence
from typing import final

import numpy as np
import torch
import torch.distributed as dist
from torch import multiprocessing as mp
from torch.cuda import is_available
from torch.utils.data import DataLoader, Dataset

from dataWrangling import fasta
from dataWrangling.intervalDataset import IntervalDataset
from embedGen.blockDistributedSampler import BlockDistributedSampler
from embedModel import EmbedModel
from utils.guessDevice import guessDevice
from utils.types import MmapF32, SizedDataset


@final
class GpuMaster:
    def __init__(self, model: EmbedModel, batchSize: int, workerCount: int):
        self.model = model
        self.batchSize = batchSize
        self.workerCount = workerCount
        self.embedDim = model.embedDim

    def _inferLoop(self, rank, dataLoader, outPath):
        """inference loop."""
        sampleCount = len(dataLoader.sampler)
        rprint = lambda *args: print(f"[{rank}]:", *args)

        # remove outFile if exists
        if os.path.exists(outPath):
            os.remove(outPath)

        outFile = np.memmap(
            outPath, dtype="float32", mode="w+", shape=(sampleCount, self.embedDim)
        )
        nextIdx = 0

        for i, batch in enumerate(dataLoader):
            outputs = self.model.embed(batch).cpu()
            outFile[nextIdx : nextIdx + len(outputs)] = outputs.numpy()
            nextIdx += len(outputs)
            rprint(f"Batch: {i + 1}\t/ {len(dataLoader)}")

        # "close" the memmap
        outFile.flush()
        del outFile

    def _worker(self, rank: int, datasets: Sequence[IntervalDataset], outPaths: Sequence[str]):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12356"
        backendType = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backendType, rank=rank, world_size=self.workerCount)

        if not dist.is_initialized():
            raise RuntimeError("Process group could not initialized")
        if not torch.accelerator.is_available():
            raise RuntimeError("Hardware acceleration not available")

        device = guessDevice(rank)
        model = self.model.to(device)
        model.to(device)

        for i, dataset, outPath in zip(range(len(datasets)), datasets, outPaths):
            if os.path.exists(outPath):
                continue
            if self.model.wants == "sequences":
                dataset = fasta.map(dataset)

            sampler = BlockDistributedSampler(dataset, num_replicas=self.workerCount, rank=rank)
            dataLoader = DataLoader(
                dataset,
                batch_size=self.batchSize,
                sampler=sampler,
                shuffle=False,
                pin_memory=True,
                num_workers=0,
            )

            outPath += "." + str(rank)
            self._inferLoop(rank, dataLoader, outPath + "__")
            # it is now finished, rename x__ -> x
            os.rename(outPath + "__", outPath)

            if rank == 0:
                print(f"\t- Finished {i+1} / {len(datasets)}")

        dist.barrier()
        dist.destroy_process_group()

    def batch(
        self, datasets: Sequence[SizedDataset], outPaths: Sequence[str]
    ) -> Iterable[MmapF32]:
        assert len(datasets) == len(outPaths)

        # big operation
        args = (datasets, outPaths)
        mp.spawn(self._worker, args=args, nprocs=self.workerCount, join=True)  # spawn method

        # TODO: aggregation can also be parallelized
        for dataset, outPath in zip(datasets, outPaths):
            outShape = (len(dataset), self.embedDim)

            if os.path.exists(outPath):
                # INFO: The overall system is designed to ignore results that
                # have already been created.
                yield np.memmap(outPath, dtype="float32", mode="r", shape=outShape)
                continue

            outFile = np.memmap(outPath, dtype="float32", mode="w+", shape=outShape)
            nextIdx = 0

            for rank in range(self.workerCount):
                rankOutPath = outPath + "." + str(rank)

                rankFile = np.memmap(rankOutPath, dtype="float32", mode="r")
                rankFile = rankFile.reshape((-1, self.embedDim))
                outFile[nextIdx : nextIdx + len(rankFile)] = rankFile

                nextIdx += len(rankFile)
                del rankFile
                os.remove(outPath + "." + str(rank))

            outFile.flush()
            yield outFile
