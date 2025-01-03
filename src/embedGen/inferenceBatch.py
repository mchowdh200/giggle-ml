import multiprocessing as mp
import os
import tracemalloc
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from embedGen.blockDistributedSampler import BlockDistributedSampler
from embedGen.hyenadnaBackend import prepare_model

doMemorySnapshots = False


class BatchInferHyenaDNA:
    def __init__(self, embedDim=128, useDDP=True, useMeanAggregation=True):
        self.embedDim = embedDim
        self.useDDP = useDDP
        self.useMeanAggregation = useMeanAggregation


    def prepare_model(self, rank, device):
        return prepare_model(rank, device)


    def item_to_device(self, item, device):
        return item.to(device)


    def infer_loop(self, rank, worldSize, model, device, dataLoader, outPath):
        """embedGen loop."""
        sampleCount = len(dataLoader.sampler)
        rprint = lambda *args: print(f"[{rank}]:", *args)

        # remove outFile if exists
        if os.path.exists(outPath):
            os.remove(outPath)

        outFile = np.memmap(outPath, dtype='float32', mode='w+',
                            shape=(sampleCount, self.embedDim))
        print("Allocated memmap.")
        nextIdx = 0

        with torch.inference_mode():
            tracemalloc.start()
            for i, input in enumerate(dataLoader):
                input = self.item_to_device(input, device)
                output = model(input).cpu()

                # mean aggregation, flatten batch dimension
                if self.useMeanAggregation:
                    output = torch.mean(output, dim=1)

                outFile[nextIdx:nextIdx + len(output)] = output
                nextIdx += len(output)

                if doMemorySnapshots:
                    if i % 10 == 0:
                        snapshot = tracemalloc.take_snapshot()
                        current, peak = tracemalloc.get_traced_memory()
                        top_stats = snapshot.statistics('lineno')
                        print(f"Process {mp.current_process().name}:")
                        print(
                            "\t==>  ", f"Peak memory usage: {peak / 10**6:.2f} MB")
                        for stat in top_stats[:5]:
                            print('\t-', stat)
                rprint(f"Batch: {i}\t/ {len(dataLoader)}")

        # "close" the memmap
        outFile.flush()
        del outFile


    def worker(self, rank, worldSize, batchSize, datasets, outPaths):
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12356'
        # backendType = 'nccl' if torch.cuda.is_available() else 'gloo'
        # dist.init_process_group(backendType, rank=rank, world_size=worldSize)

        # if not dist.is_initialized():
        #     raise "Failed to initialize distributed backend"

        # TODO: does not work on mig partitions
        device = None
        devIds = None

        # TODO: not quite functional for gpu-cpu mix
        if torch.cuda.is_available() and rank < torch.cuda.device_count():
            device = torch.device(f'cuda:{rank}')
            devIds = [rank]
        else:
            device = torch.device('cpu')

        model = self.prepare_model(rank, device)
        model.to(device)

        # if self.useDDP:
        #     model = DDP(model, device_ids=devIds)
        model.eval()

        for i, dataset, outPath in zip(range(len(datasets)), datasets, outPaths):
            if os.path.exists(outPath):
                continue

            sampler = BlockDistributedSampler(
                dataset, num_replicas=worldSize, rank=rank)
            dataLoader = DataLoader(dataset, batch_size=batchSize,
                                    sampler=sampler, shuffle=False)

            outPath += "." + str(rank)
            self.infer_loop(rank, worldSize, model,
                            device, dataLoader, outPath + '__')
            # it is now finished, rename x__ -> x
            os.rename(outPath + '__', outPath)

            if rank == 0:
                print(f"\t- Finished {i+1} / {len(datasets)}")



    def batchInfer(self, datasets: List[Dataset], outPaths: List[str], batchSize=16, worldSize=None):
        assert len(datasets) == len(outPaths)

        device = None
        if torch.cuda.is_available():
            print("We're using", torch.cuda.device_count(), "GPUs!")
            device = 'cuda'
            if worldSize is None:
                worldSize = torch.cuda.device_count()
        else:
            device = 'cpu'
            if worldSize is None:
                worldSize = 4
            print(f"We're using {worldSize} CPUs.")

        # big operation

        workers = list()
        args = (worldSize, batchSize, datasets, outPaths)
        mp.set_start_method('spawn')

        for rank in range(worldSize):
            p = mp.Process(target=self.worker, args=(rank, *args))
            p.start()
            workers.append(p)

        [p.join() for p in workers]

        # TODO: aggregation can also be parallelized
        for dataset, outPath in zip(datasets, outPaths):
            if os.path.exists(outPath):
                continue

            outFile = np.memmap(outPath, dtype='float32', mode='w+',
                                shape=(len(dataset), self.embedDim))
            nextIdx = 0

            for rank in range(worldSize):
                rankOutPath = outPath + "." + str(rank)

                rankFile = np.memmap(rankOutPath, dtype='float32', mode='r')
                rankFile = rankFile.reshape((-1, self.embedDim))
                outFile[nextIdx:nextIdx + len(rankFile)] = rankFile

                nextIdx += len(rankFile)
                del rankFile
                os.remove(outPath + "." + str(rank))

            outFile.flush()
