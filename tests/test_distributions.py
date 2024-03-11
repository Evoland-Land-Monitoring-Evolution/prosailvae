import torch
from einops import repeat

from prosailvae.utils.TruncatedNormal import TruncatedNormal, test_kl_tntn, test_kl_u


def test_sample_truncated_gaussian() -> None:
    batch_size = 10
    n_samples = 7
    lows = torch.zeros(batch_size)
    highs = torch.ones(batch_size)
    locs = torch.rand(batch_size)
    scales = torch.exp(torch.randn(batch_size))
    dist = TruncatedNormal(locs, scales, lows, highs)
    samples = dist.rsample(torch.Size((n_samples,)))
    assert (samples >= repeat(lows, "b -> n b", n=n_samples)).all()
    assert (samples <= repeat(highs, "b -> n b", n=n_samples)).all()


def test_KL_truncated_gaussian() -> None:
    test_kl_tntn()
    test_kl_u()
