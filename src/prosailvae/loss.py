from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LossConfig:
    """
    Dataclass to hold loss config.
    """

    supervised: bool = False
    beta_kl: float = 0.0
    beta_index: float = 0.0
    beta_cyclical: float = 0.0
    snap_cyclical: bool = False
    loss_type: str = "diag_nll"
    lat_loss_type: str = ""
    reconstruction_bands_coeffs: list[int] | None = None
    lat_idx: torch.Tensor = torch.tensor([])
    pvae: bool = False


def get_nll_dimensions(loss_type):
    simple_losses_1d = ["diag_nll", "hybrid_nll", "lai_nll"]
    if loss_type in simple_losses_1d:
        return 2, 1
    elif loss_type == "spatial_nll":
        return 1, 2
    else:
        raise NotImplementedError


def gaussian_nll(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma2: torch.Tensor,
    eps: float = 1e-6,
    device: str = "cpu",
    sum_dim: int = 1,
    feature_indexes: None | list[int] = None,
):
    """
    Gaussian Negative Log-Likelihood
    """
    eps = torch.tensor(eps).to(device)
    if feature_indexes is None:
        return (
            (torch.square(x - mu) / torch.max(sigma2, eps))
            + torch.log(torch.max(sigma2, eps))
        ).sum(sum_dim)
    else:
        loss = []
        for idx in feature_indexes:
            if len(sigma2.size()) != 0:
                idx_sigma2 = torch.max(sigma2.select(dim=sum_dim, index=idx), eps)
            else:
                idx_sigma2 = sigma2
            idx_loss = (
                (
                    torch.square(
                        x.select(dim=sum_dim, index=idx)
                        - mu.select(dim=sum_dim, index=idx)
                    )
                    / idx_sigma2
                )
                + torch.log(idx_sigma2)
            ).unsqueeze(sum_dim)
            loss.append(idx_loss)
        loss = torch.cat(loss, dim=sum_dim).sum(sum_dim)
        return loss


def gaussian_nll_loss(
    tgt, recs, sample_dim=2, feature_dim=1, feature_indexes: list[int] | None = None
):
    if len(recs.size()) < 3:
        raise ValueError("recs needs a batch, a feature and a sample dimension")
    elif recs.size(sample_dim) == 1:
        rec_err_var = torch.tensor(0.0001).to(
            tgt.device
        )  # constant variance, enabling computation even with 1 sample
        rec_mu = recs
    else:
        rec_err_var = recs.var(sample_dim, keepdim=True)  # .unsqueeze(sample_dim)
        rec_mu = recs.mean(sample_dim, keepdim=True)  # .unsqueeze(sample_dim)
        # if feature_dim > sample_dim: # if feature dimension is after sample dimension,
        #     # reducing it because sample dimension disappeared
        #     feature_dim = feature_dim - 1
    return gaussian_nll(
        tgt.unsqueeze(sample_dim),
        rec_mu,
        rec_err_var,
        sum_dim=feature_dim,
        feature_indexes=feature_indexes,
    ).mean()


def pvae_gaussian_nll_loss(
    tgt,
    rec_mu,
    rec_err_var,
    sample_dim=2,
    feature_dim=1,
    feature_indexes: list[int] | None = None,
):
    return gaussian_nll(
        tgt.unsqueeze(sample_dim),
        rec_mu,
        rec_err_var,
        sum_dim=feature_dim,
        feature_indexes=feature_indexes,
    ).mean()


class NLLLoss(nn.Module):
    """
    nn.Module Loss for NLL
    """

    def __init__(
        self,
        loss_type: str | None = None,
        sample_dim=2,
        feature_dim=1,
        feature_indexes: list[int] | None = None,
    ) -> None:
        """

        'feature_indexes' allows to indicate which features are
        taken into account for the loss computatino. For ex. [1 2 3] ->
        only takes into account B3 B4 B5 (0 is B2)

        """
        super().__init__()
        if loss_type is not None:
            sample_dim, feature_dim = get_nll_dimensions(loss_type)
        self.sample_dim = sample_dim
        self.feature_dim = feature_dim
        self.feature_indexes = feature_indexes

    def forward(self, targets, inputs):
        return gaussian_nll_loss(
            targets,
            inputs,
            sample_dim=self.sample_dim,
            feature_dim=self.feature_dim,
            feature_indexes=self.feature_indexes,
        )


class pvae_NLLLoss(nn.Module):
    """
    nn.Module Loss for NLL
    """

    def __init__(
        self,
        loss_type: str | None = None,
        sample_dim=2,
        feature_dim=1,
        feature_indexes: list[int] | None = None,
    ) -> None:
        """

        'feature_indexes' allows to indicate which features are
        taken into account for the loss computatino. For ex. [1 2 3] ->
        only takes into account B3 B4 B5 (0 is B2)

        """
        super().__init__()
        if loss_type is not None:
            sample_dim, feature_dim = get_nll_dimensions(loss_type)
        self.sample_dim = sample_dim
        self.feature_dim = feature_dim
        self.feature_indexes = feature_indexes

    def forward(self, targets, mu, var):
        return pvae_gaussian_nll_loss(
            targets,
            mu,
            var,
            sample_dim=self.sample_dim,
            feature_dim=self.feature_dim,
            feature_indexes=self.feature_indexes,
        )


class LatentLoss(nn.Module):
    def __init__(self, sample_dim=2, feature_dim=1, loss_type="nll") -> None:
        super().__init__()
        self.sample_dim = sample_dim
        self.feature_dim = feature_dim
        self.loss_type = loss_type

    def forward(self, targets, inputs):
        if self.loss_type == "nll":
            return gaussian_nll_loss(
                targets,
                inputs,
                sample_dim=self.sample_dim,
                feature_dim=self.feature_dim,
            )
        elif self.loss_type == "kl":
            return
