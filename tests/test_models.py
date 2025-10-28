from pathlib import Path
from unittest.mock import patch

import torch

from giggleml.data_wrangling import fasta
from giggleml.data_wrangling.interval_dataset import MemoryIntervalDataset
from giggleml.models.hyena_dna import HyenaDNA
from giggleml.models.c_model import CModel
from giggleml.utils.parallel import Parallel
from giggleml.utils.torch_utils import guess_device


def test_hyena_dna():
    model = HyenaDNA("1k")
    model.to(guess_device())
    results = model(model.collate(["ACGTTGCA", "ACT"])).cpu()

    assert len(results) == 2

    for embed in results:
        assert len(embed) == model.embed_dim
        assert embed.dtype == torch.float16

    brief = results[:, ::10]
    brief = torch.round(brief * 10)

    # # INFO: IF using mean aggregation
    # expect = torch.Tensor(
    #     [
    #         [2.0, 6.0, -16.0, -1.0, -9.0, -5.0, 10.0, 7.0, -1.0, 1.0, -3.0, 9.0, -11.0],
    #         [2.0, 6.0, -16.0, -1.0, -9.0, -5.0, 10.0, 7.0, -1.0, 1.0, -3.0, 9.0, -11.0],
    #     ]
    # )

    expect = torch.Tensor(
        [
            [-7.0, 3.0, 7.0, 8.0, -9.0, 5.0, 7.0, 6.0, -2.0, -10.0, -0.0, 3.0, 1.0],
            [-4.0, 7.0, 1.0, -0.0, -9.0, 14.0, 9.0, 8.0, -6.0, -9.0, -5.0, 1.0, 7.0],
        ]
    ).to(torch.float16)

    assert torch.allclose(brief, expect, atol=1.0)


# INFO: CountACGT Model already tested implicitly in testDataWrangling.py


def test_region2_vec():
    # TODO: impl R2V test
    ...


# INFO: ----------------
#         CModel
# ----------------------


def test_cmodel_init():
    model = CModel(
        size="1k",
        phi_hidden_dim_factor=2,
        rho_hidden_dim_factor=1,
        final_embed_dim_factor=1,
        use_gradient_checkpointing=False,
    )

    # Test that CModel is now a standalone nn.Module
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, "hyena_dna")
    assert isinstance(model.hyena_dna, HyenaDNA)

    assert model.phi_hidden_dim_factor == 2
    assert model.rho_hidden_dim_factor == 1
    assert model.final_embed_dim_factor == 1
    assert not model.use_gradient_checkpointing

    hyena_embed_dim = model.hyena_dna.embed_dim
    assert model.final_embed_dim == hyena_embed_dim

    assert len(model.phi) == 5
    assert len(model.rho) == 3


def test_cmodel_set_contents_forward():
    """Test the set_contents_forward method (element-wise operations)."""
    model = CModel(size="1k", use_gradient_checkpointing=False)
    device = guess_device()
    model.to(device).half()
    model.eval()

    # Test set_contents_forward directly with a batch dict
    with patch.object(model.hyena_dna, "forward") as mock_hyena_forward:
        mock_hdna_output = torch.randn(
            2, model.hyena_dna.embed_dim, dtype=torch.float16, device=device
        )
        mock_hyena_forward.return_value = mock_hdna_output

        batch_dict = {"input_ids": torch.tensor([[1, 2, 3], [1, 2, 0]], device=device)}

        result = model.set_contents_forward(batch_dict)

        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 2
        assert (
            result.shape[1] == model.rho_hidden_dim_factor * model.hyena_dna.embed_dim
        )
        assert result.device.type == device.type


def test_cmodel_set_means_forward():
    """Test the set_means_forward method (rho network)."""
    model = CModel(size="1k")
    device = guess_device()
    model.to(device).half()
    model.eval()

    # Create mock set means tensor
    batch_size = 3
    rho_input_dim = model.rho_hidden_dim_factor * model.hyena_dna.embed_dim
    set_means = torch.randn(
        batch_size, rho_input_dim, dtype=torch.float16, device=device
    )

    result = model.set_means_forward(set_means)

    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == batch_size
    assert result.shape[1] == model.final_embed_dim
    assert result.device.type == device.type


def test_cmodel_gradient_checkpointing_flag():
    """Test that gradient checkpointing flag is properly stored."""
    # Test with gradient checkpointing disabled
    model_no_gc = CModel(size="1k", use_gradient_checkpointing=False)
    assert not model_no_gc.use_gradient_checkpointing

    # Test with gradient checkpointing enabled
    model_gc = CModel(size="1k", use_gradient_checkpointing=True)
    assert model_gc.use_gradient_checkpointing

    # Test that model structure is correct regardless of checkpointing flag
    assert hasattr(model_gc, "phi")
    assert hasattr(model_gc, "rho")
    assert isinstance(model_gc.phi, torch.nn.Sequential)
    assert isinstance(model_gc.rho, torch.nn.Sequential)


def test_cmodel_train_mode():
    model = CModel(size="1k")

    model.train(True)
    assert model.training
    # HyenaDNA should be kept in eval mode even when CModel is in training mode
    assert not model.hyena_dna.training

    model.train(False)
    assert not model.training
    assert not model.hyena_dna.training


def test_cmodel_tokenize():
    """Test the tokenize method that wraps HyenaDNA.collate."""
    model = CModel(size="1k")

    sequences = ["ACGT", "TTGG"]
    result = model.tokenize([sequences])[0]

    assert "input_ids" in result
    assert isinstance(result["input_ids"], torch.Tensor)


def test_cmodel_hot_parameters():
    model = CModel(size="1k")

    total_params = list(model.parameters())
    hot_params = list(model.hot_parameters())

    assert all(p.requires_grad for p in hot_params)
    assert len(hot_params) <= len(total_params)


def test_cmodel_distributed_embed_structure():
    """Test that distributed_embed has the correct signature and basic structure."""
    model = CModel(size="1k", rho_hidden_dim_factor=2)

    # Check that the method exists and has correct signature
    assert hasattr(model, "distributed_embed")
    assert callable(getattr(model, "distributed_embed"))

    # This test verifies the method exists and would be called correctly
    # without actually executing the complex distributed logic
    import inspect

    sig = inspect.signature(model.distributed_embed)
    params = list(sig.parameters.keys())
    assert "data" in params
    assert "batch_size" in params
    assert "sub_workers" in params
    assert len(params) == 3  # self is not included in inspect.signature


def test_row_cmodel():
    """Test the RowCModel helper class."""
    from giggleml.models.c_model import RowCModel

    cmodel = CModel(size="1k")
    row_model = RowCModel(cmodel)

    assert row_model.wants == "sequences"
    assert hasattr(row_model, "cmodel")
    assert hasattr(row_model, "hyena_dna")
    assert row_model.hyena_dna == cmodel.hyena_dna
    assert row_model.embed_dim == cmodel.hyena_dna.embed_dim


def test_cmodel_repr():
    model = CModel(
        size="1k",
        phi_hidden_dim_factor=3,
        rho_hidden_dim_factor=2,
        final_embed_dim_factor=1,
        use_gradient_checkpointing=True,
    )

    repr_str = repr(model)
    # The repr method accesses self.size_type, which should come from hyena_dna
    expected = (
        f"CModel(size={model.hyena_dna.size_type}, phi_hidden_dim_factor=3, rho_hidden_dim_factor=2, "
        "final_embed_dim_factor=1, use_gradient_checkpointing=True)"
    )
    assert repr_str == expected


def test_cmodel_call():
    model = CModel("1k")
    device = guess_device()
    model.to(device)
    assert model.device == device
    assert model.hyena_dna.device == device

    tokens = model.tokenize([["ACGTTGCA", "ACT"]])
    # move tokens to device
    tokens = [{k: v.to(device) for (k, v) in item.items()} for item in tokens]
    results = model(tokens).cpu()

    assert len(results) == 1

    for embed in results:
        assert len(embed) == model.final_embed_dim
        assert embed.dtype == torch.float16


def _test_cmodel_dist_embed():
    model: CModel = CModel("1k").cpu()
    data_shape = (7, 3, 5)
    intervals = [
        [
            (
                "chr4",
                data_shape[2] * (data_shape[1] * i + j),
                data_shape[2] * (data_shape[1] * i + j) + data_shape[2],
            )
            for j in range(data_shape[1])
        ]
        for i in range(data_shape[0])
    ]
    datasets = [
        MemoryIntervalDataset(group, associated_fasta_path=Path("tests/test.fa"))
        for group in intervals
    ]

    distributed_results = model.distributed_embed(
        data=datasets,
        batch_size=4,
        sub_workers=2,
    ).cpu()

    sequences = [fasta.map(group, Path("tests/test.fa")) for group in intervals]
    direct_results = model(model.tokenize(sequences)).cpu()

    # we can't actually test that these results are equal until we train the model
    # or provide a deterministic parameter initialization

    assert distributed_results.shape == direct_results.shape
    # assert torch.allclose(distributed_results, direct_results, atol=1e-2)


def test_cmodel_dist_embed():
    Parallel(world_size=2).run(_test_cmodel_dist_embed)
