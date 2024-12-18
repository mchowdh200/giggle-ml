sys.path.append('./src')
from snakemake import expand, directory

# configfile: "config.yaml"

rule synthesizeSequences:
    params:
        seqLenMin = int(1e3),
        seqLenMax = int(25e3),
        seqPerUniverse = int(2e5),
        seed = 31415
    output:
        outFiles = [ "./data/synthetic/intervals.bed" ],
        fastaOut = "./data/synthetic/ref.fa"
    run:
        from src.synthesis import synthesize
        synthesize(output.fastaOut, output.outFiles, *params)

