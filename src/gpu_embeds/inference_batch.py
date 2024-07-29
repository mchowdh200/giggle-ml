# @title Batch example
"""
Let's say you want to do inference on a dataset to grab a lot of embeddings,
you can just loop thru a dataloader like this.

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import tracemalloc
import json
import os
import subprocess
import transformers
from transformers import PreTrainedModel, AutoModelForCausalLM, PretrainedConfig
import numpy as np

from gpu_embeds.standalone_hyenadna import HyenaDNAModel
from gpu_embeds.standalone_hyenadna import CharacterTokenizer
from gpu_embeds.block_distributed_sampler import BlockDistributedSampler
from gpu_embeds.standalone_hyenadna import CharacterTokenizer

# helper 1


def inject_substring(orig_str):
    """Hack to handle matching keys between models trained with and without
    gradient checkpointing."""

    # modify for mixer keys
    pattern = r"\.mixer"
    injection = ".mixer.layer"

    modified_string = re.sub(pattern, injection, orig_str)

    # modify for mlp keys
    pattern = r"\.mlp"
    injection = ".mlp.layer"

    modified_string = re.sub(pattern, injection, modified_string)

    return modified_string

# helper 2


def load_weights(scratch_dict, pretrained_dict, checkpointing=False):
    """Loads pretrained (backbone only) weights into the scratch state dict."""

    # loop thru state dict of scratch
    # find the corresponding weights in the loaded model, and set it

    # need to do some state dict "surgery"
    for key, value in scratch_dict.items():
        if 'backbone' in key:
            # the state dicts differ by one prefix, '.model', so we add that
            key_loaded = 'model.' + key
            # breakpoint()
            # need to add an extra ".layer" in key
            if checkpointing:
                key_loaded = inject_substring(key_loaded)
            try:
                scratch_dict[key] = pretrained_dict[key_loaded]
            except:
                raise Exception('key mismatch in the state dicts!')

    # scratch_dict has been updated
    return scratch_dict


class HyenaDNAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    base_model_prefix = "hyenadna"

    def __init__(self, config):
        pass

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    @classmethod
    def from_pretrained(cls,
                        path,
                        model_name,
                        download=False,
                        config=None,
                        device='cpu',
                        use_head=False,
                        n_classes=2,
                        ):
        # first check if it is a local path
        pretrained_model_name_or_path = os.path.join(path, model_name)
        if os.path.isdir(pretrained_model_name_or_path) and download == False:
            if config is None:
                config = json.load(
                    open(os.path.join(pretrained_model_name_or_path, 'config.json')))
        else:
            hf_url = f'https://huggingface.co/LongSafari/{model_name}'

            subprocess.run(
                f'rm -rf {pretrained_model_name_or_path}', shell=True)
            command = f'mkdir -p {path} && cd {path} && git lfs install && git clone {hf_url}'
            subprocess.run(command, shell=True)

            if config is None:
                config = json.load(
                    open(os.path.join(pretrained_model_name_or_path, 'config.json')))

        scratch_model = HyenaDNAModel(
            **config, use_head=use_head, n_classes=n_classes)  # the new model format
        loaded_ckpt = torch.load(
            os.path.join(pretrained_model_name_or_path, 'weights.ckpt'),
            map_location=torch.device(device)
        )

        # need to load weights slightly different if using gradient checkpointing
        if config.get("checkpoint_mixer", False):
            checkpointing = config["checkpoint_mixer"] == True or config["checkpoint_mixer"] == True
        else:
            checkpointing = False

        # grab state dict from both and load weights
        state_dict = load_weights(scratch_model.state_dict(
        ), loaded_ckpt['state_dict'], checkpointing=checkpointing)

        # scratch model has now been updated
        scratch_model.load_state_dict(state_dict)
        print("Loaded pretrained weights ok!")
        return scratch_model

#########################################################################


def prepareModel(rank, device, barrier):
    '''
    this selects which backbone to use, and grabs weights/ config from HF
    4 options:
      'hyenadna-tiny-1k-seqlen'   # fine-tune on colab ok
      'hyenadna-small-32k-seqlen'
      'hyenadna-medium-160k-seqlen'  # inference only on colab
      'hyenadna-medium-450k-seqlen'  # inference only on colab
      'hyenadna-large-1m-seqlen'  # inference only on colab
    '''

    # select model
    # use None if training from scratch
    pretrained_model_name = 'hyenadna-medium-160k-seqlen'

    max_lengths = {
        'hyenadna-tiny-1k-seqlen': 1024,
        'hyenadna-small-32k-seqlen': 32768,
        'hyenadna-medium-160k-seqlen': 160000,
        'hyenadna-medium-450k-seqlen': 450000,  # T4 up to here
        'hyenadna-large-1m-seqlen': 1_000_000,  # only A100 (paid tier)
    }

    # we need these for the decoder head, if using
    use_head = False
    n_classes = 2  # not used for embeddings only

    # you can override with your own backbone config here if you want,
    # otherwise we'll load the HF one in None
    backbone_cfg = None

    # instantiate the model (pretrained here)
    if pretrained_model_name not in ['hyenadna-tiny-1k-seqlen',
                                     'hyenadna-small-32k-seqlen',
                                     'hyenadna-medium-160k-seqlen',
                                     'hyenadna-medium-450k-seqlen',
                                     'hyenadna-large-1m-seqlen']:
        raise ValueError(
            f"Invalid pretrained model name: {pretrained_model_name}")

    if rank != 0:
        barrier.wait()

    model = HyenaDNAPreTrainedModel.from_pretrained(
        './checkpoints',
        pretrained_model_name,
        download=(rank == 0),
        config=backbone_cfg,
        device=device,
        use_head=use_head,
        n_classes=n_classes)

    # ensure only rank 0 can download the model
    if rank == 0:
        barrier.wait()

    return model


def infer_loop(rank, worldSize, batchSize, model, device, dataLoader, outPath):
    """inference loop."""
    rprint = lambda *args: print(f"[{rank}]:", *args)
    outFile = np.memmap(outPath, dtype='float32', mode='w+',
                        shape=(len(dataLoader) * batchSize, 256))
    # TODO: assumption regarding shape does not apply in case of uneven batch sizes

    with torch.inference_mode():
        tracemalloc.start()
        for i, input in enumerate(dataLoader):
            input = input.to(device)
            # execute model, retrieve embeddings
            output = model(input).cpu()
            # mean aggregation, flatten batch dimension
            output = torch.mean(output, dim=1)

            for j, single in enumerate(output):
                outFile[i * batchSize + j] = single

            if i % 10 == 0:
                snapshot = tracemalloc.take_snapshot()
                current, peak = tracemalloc.get_traced_memory()
                top_stats = snapshot.statistics('lineno')
                print(f"Process {mp.current_process().name}:")
                print("\t==>  ", f"Peak memory usage: {peak / 10**6:.2f} MB")
                for stat in top_stats[:5]:
                    print('\t-', stat)
            rprint(f"Batch: {i}\t/ {len(dataLoader)}")

    # "close" the memmap
    outFile.flush()
    del outFile


def worker(rank, worldSize, batchSize, dataset, outFile, barrier):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    backendType = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backendType, rank=rank, world_size=worldSize)

    # TODO: (from_fasta.py) Size of the batch embeds is 500x256 regardless of batch size?
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

    model = prepareModel(rank, device, barrier)
    model.to(device)
    model = DDP(model, device_ids=devIds)
    model.eval()

    outFile += "." + str(rank)
    infer_loop(rank, worldSize, batchSize, model, device, dataLoader, outFile)


# TODO: stop hardcoding embedding dimensionality
def batchInfer(dataset, outPath, batchSize=16, worldSize=None):
    device = None
    if torch.cuda.is_available():
        print("We're using", torch.cuda.device_count(), "GPUs!")
        device = 'cuda'
        if worldSize is None:
            worldSize = torch.cuda.device_count()
    else:
        device = 'cpu'
        if worldSize is None:
            worldSize = 1
        print(f"We're using {worldSize} CPUs.")

    # big operation
    mp.set_start_method('spawn', force=True)
    with mp.Pool(worldSize) as pool:
        with mp.Manager() as manager:
            barrier = manager.Barrier(worldSize)
            baseArgs = (worldSize, batchSize, dataset, outPath, barrier)
            args = [(rank,) + baseArgs for rank in range(worldSize)]
            # TODO: no longer need map: remove pool
            results = list(pool.starmap(worker, args))

    # aggregate results into a single file and delete others

    outFile = np.memmap(outPath, dtype='float32', mode='w+',
                        shape=(len(dataset), 256))
    nextIdx = 0

    for rank in range(worldSize):
        rankOutPath = outPath + "." + str(rank)

        rankFile = np.memmap(rankOutPath, dtype='float32', mode='r')
        rankFile = rankFile.reshape((-1, 256))
        outFile[nextIdx:nextIdx + len(rankFile)] = rankFile

        nextIdx += len(rankFile)
        del rankFile
        os.remove(outPath + "." + str(rank))

    outFile.flush()
