# Tests Explanation

## Cluster Tendency Test (CTT)
This test investigates the spatial distribution within a dataset comprising embeddings, treating these embeddings as discrete points. 
The CTT aims to ascertain the distances between pairs of points sampled from the dataset. If DT, the summation of all the distances,
is lesser than DR, the summation of all the distances between a point and its nearest neighbor, the CTT will be large indicating clustering 
tendencies. However, it remains ambiguous whether this test offers insights specifically relevant to the dataset under consideration. Even in 
the case of a randomly generated dataset, discernible clustering tendencies may emerge. The utility of the CTT may be enhanced by incorporating 
a priori knowledge regarding the expected clustering patterns within the genome, providing a meaningful point of comparison for analysis.

## Genome Distance Scaling Test (GDST)
The GDST computes and contrasts the genomic distance and embedding distance between two specific regions. Given that genomic proximity often corresponds to similarity in biological function, the genomic distance (GD) can serve as a somewhat imprecise indicator of shared biological functionalities, on average. It is acknowledged that functional similarities may exist between regions despite genomic separation, and such regions may also exhibit proximity in embedding space due to their biological affinity. Nevertheless, it is generally anticipated that the GD will be smaller for regions with greater similarity compared to randomly selected regions. Notably, during the embedding training phase, the model is not provided with genomic location information. Hence, any observed correlation between genomic distance and embedding distance reflects the biological insights acquired through the training process.

## Neighborhood Preserving Test (NPT) and Reconstruction Test (RT)
These assessments necessitate employing binary representations of genome data alongside embedding data. In the RT evaluation, the comparison 
between the output and observed data gauges the alignment of embeddings with the binary dataset. Conversely, the NPT examines the extent to 
which regions that are proximate in genome space maintain their adjacency in the embedding space. However, prior studies have demonstrated that 
binary embeddings are ineffective in capturing intricate biological nuances. Consequently, utilizing binary representations as a metric for 
evaluating embedding efficiency appears counterproductive.

## Continuous Comparison Test (CCT)
This test uses Monte Carlo simulations to calculate the probability of intersections between genomic intervals and the similarity of embeddings (vectors) using cosine similarity. It reads interval data and embedding vectors from specified files, generates random points within the intervals to check for overlaps, and compares embeddings to see if their similarity exceeds a given threshold. The goal is to probabilistically assess two types of biological data interactions: genomic intervals and embedding vectors. Specifically, it aims to determine the likelihood of intervals from different genomic datasets overlapping on the same chromosome and assess the similarity between high-dimensional embeddings using cosine similarity. By employing Monte Carlo simulations, the code provides estimates of these probabilities based on random sampling, allowing for statistical inference about the relationships and similarities within biological datasets.


# Libraries
Install numpy, scipy, joblib, and scikit-learn.

# Usage
## CTT
**--num_iterations**: Number of times to run each comparison.

​python path/to/cluster_tendency_test.py path/to/data_embeddings.txt --num_iterations 10

## GDST
**--num_regions**: to control the number of regions to use from the dataset. 

**--embedding_files**: genome embeddings. 

**--regions_file**: Interval file with chrom, start and end coordinates.

​python path/to/region_distance_analysis.py --num_regions 100 --embeddings_file path/to/embeddings.txt --regions_file path/to/regions.txt

## NPT and RT
**-k**: number of nearest neighbors. 

**-b**: binary representation. 

**-q**: genome embeddings.

**-n**: Number of parallel jobs.

​python NPT.py -k 3 -b path/to/genome_data.txt -q path/to/embedding_data.txt -n 2

**-f**: Optionally, you can provide the number of folds for K-fold cross-validation for RT 

python RT.py -b path/to/binary_embeddings.txt -q path/to/query_embeddings.txt -f 5

## CCT
python script.py --interval_file path/to/interval_file.bed --embedding_file path/to/embedding_file.txt 
