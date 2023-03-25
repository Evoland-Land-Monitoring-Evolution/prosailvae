import torch 
import torch.nn as nn

def gaussian_nll(x, mu, sigma, eps=1e-6, device='cpu', sum_dim=1):
    eps = torch.tensor(eps).to(device)
    return (torch.square(x - mu) / torch.max(sigma, eps)).sum(sum_dim) +  \
            torch.log(torch.max(sigma, eps)).sum(sum_dim)

def gaussian_nll_loss(tgt, recs, sample_dim=2, feature_dim=1):
    if len(recs.size()) < 3:
        raise ValueError("recs needs a batch, a feature and a sample dimension")
    elif recs.size(sample_dim)==1:
        raise ValueError("NLL needs more than 1 samples of distribution.")
    rec_err_var = torch.var(recs, sample_dim).unsqueeze(sample_dim)
    rec_mu = recs.mean(sample_dim).unsqueeze(sample_dim)
    return gaussian_nll(tgt, rec_mu, rec_err_var, sum_dim=feature_dim).mean() 

class NLLLoss(nn.Module):
    def __init__(self, sample_dim=2, feature_dim=1) -> None:
        super().__init__()
        self.sample_dim = sample_dim
        self.feature_dim = feature_dim 
    def forward(self, targets, inputs):
        return gaussian_nll_loss(targets, inputs, sample_dim=self.sample_dim, feature_dim=self.feature_dim)
    
class LatentLoss(nn.Module):
    def __init__(self, sample_dim=2, feature_dim=1, loss_type='nll') -> None:
        super().__init__()
        self.sample_dim = sample_dim
        self.feature_dim = feature_dim 
        self.loss_type = loss_type
    def forward(self, targets, inputs):
        if self.loss_type=='nll':
            return gaussian_nll_loss(targets, inputs, sample_dim=self.sample_dim, feature_dim=self.feature_dim)
        elif self.loss_type=='kl':
            return

class TotalLoss(nn.Module):
    def __init__(self, rec_sample_dim=2, rec_feature_dim=1, lat_feature_dim=1) -> None:
        super().__init__()
        self.rec_sample_dim = rec_sample_dim
        self.rec_feature_dim = rec_feature_dim 
        self.lat_feature_dim = lat_feature_dim
    def forward(self, targets, inputs):
        return gaussian_nll_loss(targets, inputs, sample_dim=self.sample_dim, feature_dim=self.feature_dim)