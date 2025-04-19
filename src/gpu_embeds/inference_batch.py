import os
from typing import List

import numpy as np
import torch
import torch.distributed as dist
from torch import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

from gpu_embeds.block_distributed_sampler import BlockDistributedSampler
from gpu_embeds.hyenadna_backend import prepare_model


class BatchInferHyenaDNA:
    def __init__(self, embedDim=128, useMeanAggregation=True, maxSeqLen=512):
        # TODO: maxSeqLen should be inferred
        self.embedDim = embedDim
        self.useMeanAggregation = useMeanAggregation

    def prepare_model(self, rank, device):
        return prepare_model(rank, device)

    def item_to_device(self, item, device):
        return item.to(device, non_blocking=True)

    def infer_loop(self, rank, model, device, dataLoader, outPath):
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

        with torch.inference_mode():
            for i, batch in enumerate(dataLoader):
                print(type(batch))
                # tuple case is used for datasets that perform chunking
                # on long inputs, chunkGroups is used to map chunks
                # back to original sequences
                if type(batch) is tuple:
                    inputIds, chunkGroups = batch
                    inputIds = self.item_to_device(inputIds, device)
                    chunkGroups = self.item_to_device(chunkGroups, device)

                    embeds = model(inputIds)

                    if self.useMeanAggregation:
                        chunkEmbeds = embeds.mean(dim=1)

                    # Average chunk embeddings by original sequence
                    outputs = scatter_mean(chunkEmbeds, chunkGroups, dim=0)
                else:
                    input = self.item_to_device(batch, device)
                    outputs = model(batch).cpu()

                    # mean aggregation, flatten batch dimension
                    if self.useMeanAggregation:
                        outputs = torch.mean(outputs, dim=1)

                outFile[nextIdx : nextIdx + len(outputs)] = outputs
                nextIdx += len(outputs)
                rprint(f"Batch: {i + 1}\t/ {len(dataLoader)}")

        # "close" the memmap
        outFile.flush()
        del outFile

    def worker(self, rank, worldSize, batchSize, datasets, outPaths):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12356"
        backendType = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backendType, rank=rank, world_size=worldSize)

        if not dist.is_initialized():
            raise "Failed to initialize distributed backend"

        # TODO: does not work on mig partitions
        # TODO: not quite functional for gpu-cpu mix
        if torch.cuda.is_available() and rank < torch.cuda.device_count():
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        model = self.prepare_model(rank, device)
        model.to(device)
        model.eval()

        for i, dataset, outPath in zip(range(len(datasets)), datasets, outPaths):
            if os.path.exists(outPath):
                continue

            sampler = BlockDistributedSampler(
                dataset, num_replicas=worldSize, rank=rank
            )

            dataLoader = DataLoader(
                dataset,
                batch_size=batchSize,
                sampler=sampler,
                shuffle=False,
                pin_memory=True,
                collate_fn=dataset.collate_fn,
                num_workers=0,
            )

            outPath += "." + str(rank)
            self.infer_loop(rank, model, device, dataLoader, outPath + "__")
            # it is now finished, rename x__ -> x
            os.rename(outPath + "__", outPath)

            if rank == 0:
                print(f"\t- Finished {i+1} / {len(datasets)}")

        dist.barrier()
        dist.destroy_process_group()

    def batchInfer(
        self, datasets: List[Dataset], outPaths: List[str], batchSize=16, worldSize=None
    ):
        assert len(datasets) == len(outPaths)
        # big operation

        args = (worldSize, batchSize, datasets, outPaths)
        mp.spawn(self.worker, args=args, nprocs=worldSize, join=True)  # spawn method

        # TODO: aggregation can also be parallelized
        for dataset, outPath in zip(datasets, outPaths):
            if os.path.exists(outPath):
                continue

            outFile = np.memmap(
                outPath, dtype="float32", mode="w+", shape=(len(dataset), self.embedDim)
            )
            nextIdx = 0

            for rank in range(worldSize):
                rankOutPath = outPath + "." + str(rank)

                rankFile = np.memmap(rankOutPath, dtype="float32", mode="r")
                rankFile = rankFile.reshape((-1, self.embedDim))
                outFile[nextIdx : nextIdx + len(rankFile)] = rankFile

                nextIdx += len(rankFile)
                del rankFile
                os.remove(outPath + "." + str(rank))

            outFile.flush()
