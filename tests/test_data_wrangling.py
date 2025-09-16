import numpy as np
import torch
from pathlib import Path

from giggleml.data_wrangling import fasta
from giggleml.data_wrangling.interval_dataset import BedDataset
from giggleml.data_wrangling.list_dataset import ListDataset
from giggleml.data_wrangling.unified_dataset import UnifiedDataset


def test_bed_fasta_parsing():
    bed = BedDataset("tests/test.bed", "tests/test.fa")
    assert bed[0] == ("chr1", 0, 1)
    assert bed[1] == ("chr1", 0, 40)
    assert bed[4] == ("chr1", 15, 25)
    assert bed[5] == ("chr1", 10, 30)
    assert bed[6] == ("chr2", 10, 30)
    assert bed[-1] == ("chr3", 5, 10)
    assert bed[-2] == ("chr3", 0, 5)

    fa = fasta.map(bed)
    assert len(fa) == len(bed)
    assert fa[-1] == "TTTTT"
    assert fa[-2] == "AAAAA"


# def testSlidingWindowDataset():
#     # TODO: SlidingWindowDataset requires further tests.
#     # there's weird stuff going on
#
#     content = [
#         "Steaks are best hot!",
#         "123456789",
#     ]
#     dataset = SlidingWindowDataset(content, 0.5)
#     expecting = [
#         "Steaks are",
#         "s are best",
#         " best hot!",
#         "1234",
#         "3456",
#         "5678",
#     ]
#
#     for i, expect in enumerate(expecting):
#         assert expect == dataset[i]


def test_unified_dataset():
    sizes = [5, 20, 8, 13]
    lists = [list(range(i)) for i in sizes]
    datasets = [ListDataset(items) for items in lists]
    uni_set = UnifiedDataset[int](datasets)
    expect = np.concatenate(lists)

    assert len(uni_set) == len(expect)

    for i, item in enumerate(expect):
        assert item == uni_set[i]

    np.random.seed(42)
    walk = list(range(len(expect)))
    np.random.shuffle(walk)

    for i in walk:
        # tests random access
        assert expect[i] == uni_set[i]


def test_to_gpu_serializable_with_dict():
    """Test to_gpu_serializable with fasta dict input"""
    fasta_dict = {'chr1': 'ATCG', 'chr2': 'GCTA'}
    
    def tokenizer(seq):
        mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        return torch.tensor([mapping[c] for c in seq], dtype=torch.long)
    
    tokens, offsets, sizes, chr_to_idx = fasta.to_gpu_serializable(fasta_dict, tokenizer)
    
    # Check shapes and types
    assert isinstance(tokens, torch.Tensor)
    assert isinstance(offsets, torch.Tensor)
    assert isinstance(sizes, torch.Tensor)
    assert isinstance(chr_to_idx, dict)
    
    # Check values
    expected_tokens = torch.tensor([0, 1, 2, 3, 3, 2, 1, 0], dtype=torch.long)
    expected_offsets = torch.tensor([0, 4], dtype=torch.long)
    expected_sizes = torch.tensor([4, 4], dtype=torch.long)
    expected_chr_to_idx = {'chr1': 0, 'chr2': 1}
    
    assert torch.equal(tokens, expected_tokens)
    assert torch.equal(offsets, expected_offsets)
    assert torch.equal(sizes, expected_sizes)
    assert chr_to_idx == expected_chr_to_idx


def test_to_gpu_serializable_with_path():
    """Test to_gpu_serializable with file path input"""
    def tokenizer(seq):
        mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        return torch.tensor([mapping[c] for c in seq], dtype=torch.long)
    
    # Test with string path
    tokens_str, offsets_str, sizes_str, chr_to_idx_str = fasta.to_gpu_serializable("tests/test.fa", tokenizer)
    
    # Test with Path object
    path_obj = Path("tests/test.fa")
    tokens_path, offsets_path, sizes_path, chr_to_idx_path = fasta.to_gpu_serializable(path_obj, tokenizer)
    
    # Check that both approaches produce equivalent results
    assert torch.equal(tokens_str, tokens_path)
    assert torch.equal(offsets_str, offsets_path)
    assert torch.equal(sizes_str, sizes_path)
    assert chr_to_idx_str == chr_to_idx_path
    
    # Check basic properties
    assert isinstance(tokens_str, torch.Tensor)
    assert len(chr_to_idx_str) > 0  # Should have chromosomes
    assert len(offsets_str) == len(sizes_str)  # Should be aligned
    assert len(offsets_str) == len(chr_to_idx_str)  # One per chromosome
