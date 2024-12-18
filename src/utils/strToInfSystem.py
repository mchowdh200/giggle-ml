from embedGen.inferenceBatch import BatchInferHyenaDNA


def getInfSystem(s=snakemake.config.batchInference.preferModel):
    if s == 'hyenaDNA':
        return BatchInferHyenaDNA()
    # TODO:
    # elif s == 'region2Vec':
    #     return R2VBatchInf()
    else:
        raise "Unknown inference model"