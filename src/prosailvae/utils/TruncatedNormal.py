import math
from numbers import Number

import torch
from torch.distributions import Distribution, Uniform, constraints, register_kl
from torch.distributions.utils import broadcast_all

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    Credits to https://github.com/toshas/torch_truncnorm/blob/main/TruncatedNormal.py
    for the implementation backbone
    """

    arg_constraints = {
        "low": constraints.real,
        "high": constraints.real,
    }
    has_rsample = True

    def __init__(self, loc, scale, low, high, validate_args=None):
        self.loc, self.scale, self.low, self.high = broadcast_all(loc, scale, low, high)
        self.low = torch.nan_to_num(self.low, nan=math.nan)
        self.high = torch.nan_to_num(self.high, nan=math.nan)
        self.scaled_low = (self.low - self.loc) / self.scale
        self.scaled_high = (self.high - self.loc) / self.scale
        if (
            isinstance(low, Number)
            and isinstance(high, Number)
            and isinstance(loc, Number)
            and isinstance(scale, Number)
        ):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)
        if self.low.dtype != self.high.dtype:
            raise ValueError("Truncation bounds types are different")
        if self.loc.dtype != self.scale.dtype:
            raise ValueError("Loc and scale types are different")
        if any(
            (self.low >= self.high)
            .view(
                -1,
            )
            .tolist()
        ):
            raise ValueError("Incorrect truncation range")
        eps = torch.finfo(self.low.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self.little_phi_a = self._normal_pdf(self.scaled_low)
        self.little_phi_b = self._normal_pdf(self.scaled_high)
        self._big_phi_a = self._normal_cdf(self.scaled_low)
        self._big_phi_b = self._normal_cdf(self.scaled_high)
        self.Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._logZ = self.Z.log()
        self._log_scale = scale.log()
        self._lpbb_m_lpaa_d_Z = (
            self.little_phi_b * self.scaled_high - self.little_phi_a * self.scaled_low
        ) / self.Z
        self._mean = (
            -(self.little_phi_b - self.little_phi_a) / self.Z * self.scale + self.loc
        )
        self._variance = (
            1
            - self._lpbb_m_lpaa_d_Z
            - ((self.little_phi_b - self.little_phi_a) / self.Z) ** 2
        ) * self.scale**2
        self._entropy = (
            CONST_LOG_SQRT_2PI_E
            + self._logZ
            - 0.5 * self._lpbb_m_lpaa_d_Z
            + self._log_scale
        )

    @constraints.dependent_property
    def support(self):
        """
        Domain of the Truncated Normal
        """
        return constraints.interval(self.low, self.high)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self.Z

    @staticmethod
    def _normal_pdf(x):
        return (-(x**2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _normal_cdf(x: torch.Tensor):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _normal_icdf(x: torch.Tensor):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value: torch.Tensor):
        """
        Cumulative Distribution Function
        """
        if self._validate_args:
            self._validate_sample(value)
        return (
            (self._normal_cdf((value - self.loc) / self.scale) - self._big_phi_a)
            / self.Z
        ).clamp(0, 1)

    def icdf(self, value: torch.Tensor):
        """
        Inverse Cumulative Distribution Function
        """
        return (
            self._normal_icdf(self._big_phi_a + value * self.Z) * self.scale + self.loc
        )

    def log_prob(self, value: torch.Tensor):
        """
        Log PDF
        """
        if self._validate_args:
            self._validate_sample(value)
        return (
            CONST_LOG_INV_SQRT_2PI
            - self._logZ
            - (((value - self.loc) / self.scale) ** 2) * 0.5
            - self._log_scale
        )

    def pdf(self, value: torch.Tensor):
        """
        Probability Density Function
        """
        return self.log_prob(value).exp()

    def rsample(self, sample_shape: torch.Size | None = None):
        """
        Reparametrized sampled (differentiable w.r.t. distribution parameters)
        """
        if sample_shape is None:
            sample_shape = torch.Size()
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.low.device).uniform_(
            self._dtype_min_gt_0, self._dtype_max_lt_1
        )
        return self.icdf(p)


@register_kl(TruncatedNormal, TruncatedNormal)
def kl_truncated_normal_truncated_normal(p, q):
    """Kullback-Leibler divergence between two truncated normal
    distributions on the same interval

    """
    assert isinstance(p, TruncatedNormal) and isinstance(q, TruncatedNormal)
    assert torch.isclose(p.low, q.low).all()
    assert torch.isclose(p.high, q.high).all()
    sq_sigma1 = p.scale.pow(2)
    sq_sigma2 = q.scale.pow(2)
    kl_pq = -1 / 2
    kl_pq += (q.scale * q.Z).log() - (p.scale * p.Z).log()
    kl_pq += (
        -(p.scaled_low * p.little_phi_a - p.scaled_high * p.little_phi_b)
        / (2 * p.Z)
        * (1 - sq_sigma1 / sq_sigma2)
    )
    kl_pq += (p.loc - q.loc).pow(2) / 2 / sq_sigma2
    kl_pq += sq_sigma1 / 2 / sq_sigma2
    kl_pq += (
        (p.loc - q.loc) * p.scale / p.Z / sq_sigma2 * (p.little_phi_a - p.little_phi_b)
    )
    return kl_pq


@register_kl(TruncatedNormal, Uniform)
def kl_truncated_normal_uniform(p, q=None):
    """Kullback-Leibler divergence between a truncated normal
    distribution and a uniform distribution on the same interval

    """
    assert isinstance(p, TruncatedNormal)
    if q is not None:
        assert isinstance(q, Uniform)
        assert torch.isclose(p.low, q.low).all()
        assert torch.isclose(p.high, q.high).all()
    kl_pq = (
        -torch.log(p.scale.float() * p.Z.float())
        - torch.log(torch.tensor(2 * math.pi)) / 2
        - 1 / 2
    )
    kl_pq += -(p.scaled_low * p.little_phi_a - p.scaled_high * p.little_phi_b) / (
        2 * p.Z
    )
    kl_pq += torch.log(p.high - p.low)
    return kl_pq


def numerical_kl(density_p: torch.Tensor, density_q: torch.Tensor, dim: int = 0, dx=1):
    """
    Numerical Kullback-Leibler divergence from densities.
    """
    assert density_p.size() == density_q.size()
    kl_pq = (density_p * density_p.log()).sum(dim) - (density_p * density_q.log()).sum(
        dim
    )
    return kl_pq * dx


def test_kl_tntn():
    """
    Test of KL divergence between 2 truncated normal distributions.
    """
    mu_1 = torch.tensor(1.0)
    mu_2 = torch.tensor(2.0)
    sigma_1 = torch.tensor(1.0)
    sigma_2 = torch.tensor(0.5)
    low = torch.tensor(0.0)
    high = torch.tensor(3.0)
    p = TruncatedNormal(loc=mu_1, scale=sigma_1, low=low, high=high)
    q_tn = TruncatedNormal(loc=mu_2, scale=sigma_2, low=low, high=high)

    kl_pq_tntn_cls = kl_truncated_normal_truncated_normal(p, q_tn)

    dx = torch.tensor(0.0001)
    support = torch.arange(low.item(), high.item(), dx.item())
    density_p = p.pdf(support)
    density_q_tn = q_tn.pdf(support)
    kl_pq_tntn_num = numerical_kl(density_p, density_q_tn, dim=0, dx=dx)
    assert torch.isclose(kl_pq_tntn_cls, kl_pq_tntn_num, atol=1e-4)


def test_kl_u():
    """
    Test of KL divergence between 2 truncated normal distributions.
    """
    mu_1 = torch.tensor(1.0)
    sigma_1 = torch.tensor(1.0)
    low = torch.tensor(0.0)
    high = torch.tensor(3.0)
    p = TruncatedNormal(loc=mu_1, scale=sigma_1, low=low, high=high)
    q_u = Uniform(low=low, high=high)

    kl_pq_u_cls = kl_truncated_normal_uniform(p, q_u)

    dx = torch.tensor(0.0001)
    support = torch.arange(low.item(), high.item(), dx.item())[:-1]
    density_p = p.pdf(
        support
    )  # Exclude last value for uniform density (high bound exluded)
    density_q_u = q_u.log_prob(support).exp()

    kl_pq_u_num = numerical_kl(density_p, density_q_u, dim=0, dx=dx)
    assert torch.isclose(kl_pq_u_cls, kl_pq_u_num, atol=1e-4)


if __name__ == "__main__":
    test_kl_tntn()
    test_kl_u()
