import torch

from giggleml.embed_gen.embed_model import HyenaDNA
from giggleml.utils.guess_device import guess_device


def test_hyena_dna():
    model = HyenaDNA("1k")
    model.to(guess_device())
    results = model.embed(model.collate(["ACGTTGCA", "ACT"])).cpu()

    assert len(results) == 2

    for embed in results:
        assert len(embed) == model.embed_dim
        assert embed.dtype == torch.float32

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
    )

    assert torch.allclose(brief, expect, atol=1.0)


# INFO: CountACGT Model already tested implicitly in testDataWrangling.py


def test_region2_vec():
    # TODO: impl R2V test
    ...
