roadmapEpigBedPattern = DATA + "/roadmap_epigenomics/beds/{id}.bed"
roadmapEpigBedNames, = glob_wildcards(roadmapEpigBedPattern)


rule roadmapEpigenomicsEmbeds:
    input:
        refGenome = HG + "/hg19.fa",
        bedFiles = expand(roadmapEpigBedPattern, id=roadmapEpigBedNames),
    output:
        embeds = expand(DATA + "/roadmap_epigenomics/embeds/{id}.npy",
                        id=roadmapEpigBedNames),
        # associated metadata should also be generated
        meta = expand(DATA + "/roadmap_epigenomics/embeds/{id}.npy.meta",
                      id=roadmapEpigBedNames),
        # the region set collection also produces a "master" mean embed dict
        master = DATA + "/roadmap_epigenomics/embeds/master.pickle"
    params:
        model = HyenaDNA("32k")
    run:
        batchSize = 64
        subWorkerCount = 4

        inputData = [
          BedDataset(bed, associatedFastaPath=input.refGenome)
          for bed in input.bedFiles
        ]

        # big job
        EmbedPipeline(params.model, batchSize, subWorkers=subWorkerCount) \
          .embed(intervals=inputData, out=output.embeds)
        # little job (produce $master)
        meanEmbedDict.build(output.embeds, output.master)
        print("Complete.")
