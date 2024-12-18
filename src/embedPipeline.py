import os
from datetime import datetime
from pathlib import Path

from data_wrangling.seq_datasets import TokenizedDataset, FastaDataset, BedDataset
from data_wrangling.transform_dataset import TransformDataset
import intervalTransforms as Transform
from utils.strToInfSystem import getInfSystem


# TODO: notion of config item "preferModel" should be "preferPipeline"


def embedAfterTransform(
        queryIntervals,
        sampleIntervals,
        queryEmbeds,
        sampleEmbeds,
        infoFile,
        expName,
        k,
        padLength,
        querySwellFactor,
        queryChunkAmnt,
        sampleSwellFactor,
        sampleChunkAmnt,
        queryTranslation,
):
    refGenome = snakemake.config.referenceGenome
    conf = snakemake.config.batchInference
    batchSize = conf.batchSize
    workers = conf.workes
    bufferSize = conf.bufferSize
    inputsInMemory = conf.inputsInMemory
    preferModel = conf.preferModel

    sampleIntervals = TransformDataset(
        backingDataset=BedDataset(
            sampleIntervals,
            inMemory=inputsInMemory,
            bufferSize=bufferSize),
        transforms=[
            Transform.Swell(swellFactor=sampleSwellFactor),
            Transform.Chunk(chunkAmnt=sampleChunkAmnt)
        ])

    queryIntervals = TransformDataset(
        backingDataset=BedDataset(
            queryIntervals,
            inMemory=inputsInMemory,
            bufferSize=bufferSize),
        transforms=[
            Transform.Translate(offset=queryTranslation),
            Transform.Swell(swellFactor=querySwellFactor),
            Transform.Chunk(chunkAmnt=queryChunkAmnt)
        ])

    sampleTokens = TokenizedDataset(
        FastaDataset(refGenome, sampleIntervals),
        padToLength=padLength)

    queryTokens = TokenizedDataset(
        FastaDataset(refGenome, queryIntervals),
        padToLength=padLength)

    Path(sampleEmbeds).parent.mkdir(parents=True, exist_ok=True)
    Path(queryEmbeds).parent.mkdir(parents=True, exist_ok=True)

    getInfSystem(preferModel).batchInfer(
        [sampleTokens, queryTokens],
        [sampleEmbeds, queryEmbeds],
        batchSize,
        workers)

    print("Finished embedding.")

    # write config information to info.md
    Path(infoFile).parent.mkdir(parents=True, exist_ok=True)
    with open(infoFile, "w") as f:
        f.write("Date: " + str(datetime.now()) + "\n")
        f.write("Parameters:\n")
        f.write(f"\texpName: {expName}\n")
        f.write(f"\tk: {k}\n")
        f.write(f"\tpadLength: {padLength}\n")
        f.write(f"\tquerySwellFactor: {querySwellFactor}\n")
        f.write(f"\tqueryChunkAmnt: {queryChunkAmnt}\n")
        f.write(f"\tsampleSwellFactor: {sampleSwellFactor}\n")
        f.write(f"\tsampleChunkAmnt: {sampleChunkAmnt}\n")
        f.write(f"\tqueryTranslation: {queryTranslation}\n")

    # TODO: this should be generalized as to not rely on query/sample pairs


def embedSets(refGenome, bedFiles, embedPaths, seqMinLen, seqMaxLen):
    conf = snakemake.config.batchInference
    batchSize = conf.batchSize
    workers = conf.workes
    bufferSize = conf.bufferSize
    inputsInMemory = conf.inputsInMemory

    sourceDatasets = list()
    outPaths = list()

    print("Starting inference on", len(ids), "bed files.")
    for bedFile, i in enumerate(bedFiles):
        embedFile = embedPaths[bedFile]
        outPaths.append(embedFile)

        dataset = TokenizedDataset(
            FastaDataset(
                refGenome,
                BedDataset(
                    bedFile,
                    inputsInMemory,
                    bufferSize=bufferSize,
                    maxLen=seqMaxLen
                )
            ),
            padToLength=seqMaxLen
        )

        sourceDatasets.append(dataset)

    getInfSystem().batchInfer(sourceDatasets, outPaths, batchSize, workers)


# from intervalTransforms import BasicTransforms


# class EmbedPipeline:
#     def __init__(self, inferConf, refGenome):
#         self.inferConf = inferConf
#         self.refGenome = refGenome
#
#     def simple(self, bedPaths: list[str], embedDir: str):
#         pass
#
#     def transformBed(self, bedPaths: list[str], embedDir: str, transforms=BasicTransforms(), embedFileSuffix=""):
#         conf = self.inferConf
#         batchSize = conf.batchSize
#         workers = conf.workes
#         bufferSize = conf.bufferSize
#         inputsInMemory = conf.inputsInMemory
#         rowsLimit = conf.rowsLimit
#         maxSeqLen = conf.maxSeqLen
#         preferModel = conf.preferModel
#         refGenome = self.refGenome
#
#         if preferModel == "hyenaDNA":
#             infSystem = BatchInferHyenaDNA()
#         # TODO: elif preferModel == "intervalTransformer":
#         else:
#             raise f"{preferModel} is not a recognized batch inference system."
#
#         sourceDatasets = list()
#         outPaths = list()
#
#         if len(bedPaths) > 1:
#             print("Starting embed_gen on", len(bedPaths), "bed files.")
#
#         for bedPath in bedPaths:
#             name = Path(bedPath).stem
#             embedFile = os.path.join(embedDir, name + ".npy")
#             outPaths.append(embedFile)
#
#             dataset = TokenizedDataset(
#                 FastaDataset(
#                     refGenome,
#                     BedDataset(
#                         bedPath,
#                         inputsInMemory,
#                         bufferSize=bufferSize,
#                         rowsLimit=rowsLimit,
#                         maxLen=maxSeqLen
#                     )
#                 ),
#                 padToLength=maxSeqLen
#             )
#
#             sourceDatasets.append(dataset)
#
#         infSystem.batchInfer(sourceDatasets, outPaths, batchSize, workers)
#
#
