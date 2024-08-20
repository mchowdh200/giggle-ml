import torch
import torch.nn as nn
from data_wrangling.list_dataset import ListDataset
from gpu_embeds.inference_batch import BatchInferHyenaDNA
import numpy as np
import os


class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def forward(self, x):
        return x


class BatchInfTesting(BatchInferHyenaDNA):
    def __init__(self):
        embedDim = 10
        super(BatchInfTesting, self).__init__(
            embedDim, useDDP=False, useMeanAggregation=False)

    def prepare_model(self, rank, device):
        return IdentityModel()


def test_inference_batch():
    batchInf = BatchInfTesting()
    workers = 2
    batchSize = 3
    count = 25
    outPath = "./test_out.npy"

    inputs = torch.zeros((count, 10), dtype=torch.float32)

    for i in range(count):
        inputs[i] = torch.tensor([i] * 10, dtype=torch.float32)

    dataset = ListDataset(inputs)
    batchInf.batchInfer(dataset, outPath, batchSize, workers)

    outputs = np.memmap(outPath, dtype='float32', mode='r')
    outputs = outputs.reshape((-1, 10))
    inputs = inputs.numpy()

    os.remove(outPath)
    assert np.allclose(inputs, outputs)
