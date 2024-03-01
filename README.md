# Zero-shot/Few-shot learning
- Zero-shot learning refers to the ability of large pre-trained models to perform tasks they were not explicitly trained on
- For example, Bert/GPT-based models were trained on large corpora of text with language modeling objectives, but they can generalize to new, unseen tasks well with either minimal training (few-shot) or no training (zero-shot)
![visualization of S-BERT label and text embeddings](https://joeddav.github.io/blog/images/zsl/tsne_no_projection.png)

# In context
- In the context of our goals, we can make use of large DNA foundation models pre-trained on language modeling objectives
- To create an embedding for a region set we can create a set of vectors from a bed file and a reference genome
	- Then we can do mean aggregation to create a single region set vector.
![[zero shot .png]]

# Possible models to test
- There are various large models out there that we could test to gauge the efficacy of a zero-shot approach
- **Enformer (Oct 2021)
	- Conv input stage followed by transformer (used to reduce sequence length before transormer to maximize context length)
	- Prediction of various values of genomic assays like ChIP-seq, etc.
- **Deep RNA (Sept 2023)** (they say they'll release the model upon peer review?)
	- Conv input stage -> transformer
	- Comes from industry so maybe they wont release it.
- **Geneformer (May 2023)**
	- pre-trained on a large-scale corpus of about 30  million single-cell transcriptomes
- **DNABERT2 (June 2023)**
	- Transformer trained on human reference genome with language modeling objective.
	- Uses flash attention and tokenizer tricks to enable faster inference on longer sequences
- **Hyena DNA (June 2023)**
	- Long-range convolution-based model capable of handling very long sequence lengths
	- Can handle even up to 1Mbp (with batch size 1 on 80GB A100)
- **Mamba (Dec 2023)**
	- State space sequence model also able to handle very long sequences

# Initial Metrics to test
Taken from 
[1] *Zheng, G., Rymuza, J., Gharavi, E., LeRoy, N. J., Zhang, A., & Sheffield, N. C. (2023). Methods for evaluating unsupervised vector representations of genomic regions. bioRxiv, 2023-08.*

### Cluster tendency test (CTT)
- How well a set of embeddings can form clusters
- Useful for both individual regions and region sets
### Reconstruction test (RCT)
- Evaluates information preserved in an embedding by testing whether an embedding can reconstruct the training data.
- Probably easy for individual regions and harder for region sets.
### Genome distance scaling test (GDST)
- Calculates how the embedding distance between two regions scales with genome distance.
- Unlikely to use this for region sets -- just single regions.
- Use monte carlo simulation ...
### Neighborhood preserving test (NPT)
- Calculates how much the neighborhood of a region in the genome space overlaps with the neighborhood of the same region in the embedding space.
- Again, this is mostly useful for single regions.

# Datasets to test
From [1],
> We collected 690 BED files from the ENCODE Uniform TFBS composite track1 as a region set collection to generate and evaluate region embeddings.

We can expand on this later to something larger scale.
- check the giggle paper

# Other possible experiments or applications
- Comparison with giggle index searches
- Retrieval augmented machine learning models for other types of tasks.
	- Eg. a sequence model that has a KNN data structure that retrieves nearest neighbors and uses them as input features.
# Going beyond zero-shot
![[feedforward aggregation 1.png]]![[transformer agg 1.png]]