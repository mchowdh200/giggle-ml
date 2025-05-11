"""
This workflow involves creating embeddings based on transformations of the
ENCFF478KET_small dataset. Each variation is used to benchmark the new
embedding-based system against (legacy) giggle.
"""

from giggleml.intervalTransformer import *
from giggleml.giggle import giggleBenchmark

experiments = {
    "all-chunk": {
        "k": 300,
        "querySwellFactor": 1,
        "queryChunkAmnt": 3,
        "sampleSwellFactor": 1,
        "sampleChunkAmnt": 3,
        "queryTranslation": 0
    },
    "all-thirds": {
        "k": 300,
        "querySwellFactor": 3,
        "queryChunkAmnt": 3,
        "sampleSwellFactor": 3,
        "sampleChunkAmnt": 3,
        "queryTranslation": 0
    },
    "query-chunk-highk": {
        "k": 300,
        "querySwellFactor": 1,
        "queryChunkAmnt": 3,
        "sampleSwellFactor": 1,
        "sampleChunkAmnt": 1,
        "queryTranslation": 0
    },
    "query-chunk": {
        "k": 100,
        "querySwellFactor": 1,
        "queryChunkAmnt": 3,
        "sampleSwellFactor": 1,
        "sampleChunkAmnt": 1,
        "queryTranslation": 0
    },
    "query-shrink-light": {
        "k": 90,
        "querySwellFactor": 0.9,
        "queryChunkAmnt": 1,
        "sampleSwellFactor": 1,
        "sampleChunkAmnt": 1,
        "queryTranslation": 0
    },
    "query-swell-light": {
        "k": 110,
        "querySwellFactor": 1.1,
        "queryChunkAmnt": 1,
        "sampleSwellFactor": 1,
        "sampleChunkAmnt": 1,
        "queryTranslation": 0
    },
    "query-swell": {
        "k": 200,
        "querySwellFactor": 2,
        "queryChunkAmnt": 1,
        "sampleSwellFactor": 1,
        "sampleChunkAmnt": 1,
        "queryTranslation": 0
    },
    "query-thirds": {
        "k": 100,
        "querySwellFactor": 3,
        "queryChunkAmnt": 3,
        "sampleSwellFactor": 1,
        "sampleChunkAmnt": 1,
        "queryTranslation": 0
    },
    "straight-highpad": {
        "k": 100,
        "querySwellFactor": 1,
        "queryChunkAmnt": 1,
        "sampleSwellFactor": 1,
        "sampleChunkAmnt": 1,
        "queryTranslation": 0
    },
    "straight": {
        "k": 100,
        "querySwellFactor": 1,
        "queryChunkAmnt": 1,
        "sampleSwellFactor": 1,
        "sampleChunkAmnt": 1,
        "queryTranslation": 0
    },
    "translate": {
        "k": 100,
        "querySwellFactor": 1,
        "queryChunkAmnt": 1,
        "sampleSwellFactor": 1,
        "sampleChunkAmnt": 1,
        "queryTranslation": 10
    }
}

expNames = list(experiments.keys())
expDataset = f"{DATA}/ENCFF478KET_small"


rule allGiggleBenchmark:
  input:
    overlapPlot = expand(f"{config.experimentsDir}/giggleBench/{{expName}}/recallByOverlap.png", expName=expNames),
    geOverlapPlot = expand(f"{config.experimentsDir}/giggleBench/{{expName}}/recallByGEOverlap.png", expName=expNames)


rule giggleBenchmark:
  """
  Responsibilities:
  --- for a particular experiment configuration ---
    1. Transform input intervals accordingly
    2. Create embeddings
    3. Create "modern" giggle results
    4. Compare results with legacy results

  The interval transforms and giggleBenchmark(.) dependence on
  `IntervalTransformer`s is due to an attempt to improve recall by transforming
  intervals before they are embedded and operated on by the vector database. To
  perform the benchmark after the vector database, results must be
  back-translated to their original intervals for comparison.
  """
  input:
    queryIntervals = f"{expDataset}/query.bed",
    sampleIntervals = f"{expDataset}/sample.bed",
    legacyResults = f"{expDataset}/giggleHits.bed",
    fasta = f"{HG}/hg38.fa"
  output:
    queryEmbeds = f"{expDataset}/embeds/{{expName}}/query.npy",
    queryEmbedsMeta = f"{expDataset}/embeds/{{expName}}/query.npy.meta",
    sampleEmbeds = f"{expDataset}/embeds/{{expName}}/sample.npy",
    sampleEmbedsMeta = f"{expDataset}/embeds/{{expName}}/sample.npy.meta",
    infoFile = f"{config.experimentsDir}/giggleBench/{{expName}}/info.txt",

    overlapPlot = f"{config.experimentsDir}/giggleBench/{{expName}}/recallByOverlap.png",
    geOverlapPlot = f"{config.experimentsDir}/giggleBench/{{expName}}/recallByGEOverlap.png"
  params:
    exp = lambda wildcards: experiments[wildcards.expName],
    model = HyenaDNA('1k')
  run:
    # 1. transform intervals according to the experiment configuration
    queryBed = BedDataset(input.queryIntervals, associatedFastaPath=input.fasta)
    sampleBed = BedDataset(input.sampleIntervals, associatedFastaPath=input.fasta)
    queryIT = IntervalTransformer(queryBed, [
      Translate(params.exp['queryTranslation']),
      Swell(params.exp['querySwellFactor']),
      Split(params.exp['queryChunkAmnt'])
    ])
    sampleIT = IntervalTransformer(sampleBed, [
      Swell(params.exp['sampleSwellFactor']),
      Split(params.exp['sampleChunkAmnt'])
    ])

    # 2. create embeddings
    batchSize = 4096
    workerCount = 4
    subWorkerCount = 2
    pipeline = EmbedPipeline(params.model, batchSize, workerCount, subWorkerCount)
    queryEmbeds: MmapF32 = pipeline.embed(queryIT.newDataset, output.queryEmbeds).data
    sampleEmbeds: MmapF32 = pipeline.embed(sampleIT.newDataset, output.sampleEmbeds).data

    # 3. write an info file
    with open(output.infoFile, "w") as f:
      f.write(f'date: {str(datetime.datetime.today())}\n')
      f.write("experiment: " + str({
        "name": wildcards.expName,
        **params.exp,
      }) + "\n")
      f.write(f'model: {str(params.model)}\n')

      # 4. perform benchmark
      hitCount, groundTruth = giggleBenchmark(
        queryIT,
        sampleIT,
        queryEmbeds,
        sampleEmbeds,
        input.legacyResults,
        output.overlapPlot,
        output.geOverlapPlot,
        params.exp['k']
      )

      f.write(f'hitCount: {str(hitCount)}\n')
      f.write(f'groundTruth: {str(groundTruth)}\n')
