import numpy as np

import embed_tests.tests as tests
from data_wrangling.seq_datasets import BedDataset, FastaDataset, TokenizedDataset
from utils.strToInfSystem import getInfSystem


def advancedTests(intervals, embeds, embedsPath, swtFigPath, limit):
    refGenome = snakemake.config.referenceGenome
    fastaDs = FastaDataset(refGenome, intervals)
    tokDs = TokenizedDataset(fastaDs)

    # TODO: only god knows why the reverse order causes a crash
    tests.swt(embedsPath, fastaDs, swtFigPath, considerLimit=limit)
    print()
    tests.rct(embeds, tokDs)


def runTests(intervalsPath, embedsPath, limit, swtFigPath):
    intervals = BedDataset(intervalsPath, rowsLimit=limit, inMemory=True)

    # TODO: embed dimensionality needs to be included in embedding file
    embeds = np.memmap(embedsPath, dtype=np.float32, mode="r")
    embedDim = getInfSystem().embedDim
    embeds = embeds.reshape(-1, embedDim)

    print()
    advancedTests(intervals, embeds, embedsPath, swtFigPath, limit)
    print()
    tests.ctt(embeds)
    print()
    tests.gdst(embeds, intervals)
    print()
    tests.npt(embeds, intervals)


