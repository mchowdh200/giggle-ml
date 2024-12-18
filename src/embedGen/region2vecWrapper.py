import torch.nn as nn
from geniml.io import Region
from geniml.region2vec import Region2VecExModel as BackingModel
from torch import tensor

from embedGen.inferenceBatch import BatchInferHyenaDNA


def prepareModel(rank, device):
    pass


class Region2VecModel(nn.Module):
    def __init__(self):
        modelName = "databio/r2v-encode-hg38"
        self.model = BackingModel(modelName)
        super(Region2VecModel, self).__init__()

    def forward(self, x):
        x = list(zip(*x))
        x = [Region(*i) for i in x]
        embed = self.model.encode(x)
        return tensor(embed)


class R2VBatchInf(BatchInferHyenaDNA):
    def __init__(self):
        embedDim = 100
        super(R2VBatchInf, self).__init__(
            embedDim, useDDP=False, useMeanAggregation=False)

    def prepare_model(self, rank, device):
        return Region2VecModel()

    def item_to_device(self, item, device):
        # TODO: moving regions to GPU not yet supported
        return item
