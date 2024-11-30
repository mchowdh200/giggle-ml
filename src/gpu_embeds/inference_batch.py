import numpy as np
import torch
import tracemalloc
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import os
from gpu_embeds.block_distributed_sampler import BlockDistributedSampler
from gpu_embeds.hyenadna_wrapper import prepare_model


doMemorySnapshots = True


class BatchInferHyenaDNA:
    def __init__(self, embedDim=128, useDDP=True, useMeanAggregation=True):
        self.embedDim = embedDim
        self.useDDP = useDDP
        self.useMeanAggregation = useMeanAggregation

    def prepare_model(self, rank, device):
        return prepare_model(rank, device)

    def item_to_device(self, item, device):
        item = item.to(device)

    def infer_loop(self, rank, worldSize, model, device, dataLoader, outPath):
        """inference loop."""
        sampleCount = len(dataLoader.sampler)
        rprint = lambda *args: print(f"[{rank}]:", *args)
        outFile = np.memmap(outPath, dtype='float32', mode='w+',
                            shape=(sampleCount, self.embedDim))
        nextIdx = 0

        with torch.inference_mode():
            tracemalloc.start()
            for i, input in enumerate(dataLoader):
                # TODO: .to call on cpu side string tuples
                input = self.item_to_device(input, device)
                # execute model, retrieve embeddings
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

    def worker(self, rank, worldSize, batchSize, dataset, outFile):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        backendType = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backendType, rank=rank, world_size=worldSize)

        sampler = BlockDistributedSampler(
            dataset, num_replicas=worldSize, rank=rank)
        dataLoader = DataLoader(dataset, batch_size=batchSize,
                                sampler=sampler, shuffle=False)

        # TODO: does not work on mig partitions
        device = None
        devIds = None
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{rank}')
            devIds = [rank]
        else:
            device = torch.device('cpu')

        model = self.prepare_model(rank, device)
        model.to(device)
        if self.useDDP:
            model = DDP(model, device_ids=devIds)
        model.eval()

        outFile += "." + str(rank)
        self.infer_loop(rank, worldSize, model,
                        device, dataLoader, outFile)

    def batchInfer(self, dataset, outPath, batchSize=16, worldSize=None):
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
        mp.set_start_method('spawn', force=True)
        args = (worldSize, batchSize, dataset, outPath)
        mp.spawn(self.worker, args=args, nprocs=worldSize)

        # aggregate results into a single file and delete others

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
