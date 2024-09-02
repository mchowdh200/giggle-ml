from geniml.io import Region
from geniml.region2vec import Region2VecExModel

model = Region2VecExModel("databio/r2v-encode-hg38")
bed = "data/synthetic/universe_0.bed"
embeds = model.encode(bed)

print(embeds)
