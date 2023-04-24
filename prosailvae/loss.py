import torch 
import torch.nn as nn
from prosailvae.dist_utils import kl_tn_uniform, truncated_gaussian_nll
from utils.utils import select_rec_loss_fn
from utils.image_utils import crop_s2_input

def gaussian_nll(x, mu, sigma, eps=1e-6, device='cpu', sum_dim=1):
    """
    Gaussian Negative Log-Likelihood
    """
    eps = torch.tensor(eps).to(device)
    return (torch.square(x - mu) / torch.max(sigma, eps)).sum(sum_dim) +  \
            torch.log(torch.max(sigma, eps)).sum(sum_dim)

def gaussian_nll_loss(tgt, recs, sample_dim=2, feature_dim=1):
    """
    Gaussian Negative Log-Likelihood loss
    """
    if len(recs.size()) < 3:
        raise ValueError("recs needs a batch, a feature and a sample dimension")
    elif recs.size(sample_dim)==1:
        raise ValueError("NLL needs more than 1 samples of distribution.")
    rec_err_var = torch.var(recs, sample_dim).unsqueeze(sample_dim)
    rec_mu = recs.mean(sample_dim).unsqueeze(sample_dim)
    return gaussian_nll(tgt, rec_mu, rec_err_var, sum_dim=feature_dim).mean() 

class NLLLoss(nn.Module):
    """
    nn.Module Loss for NLL
    """
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
    """
    Class for SimVAE's loss
    """
    def __init__(self, rec_sample_dim=2, rec_feature_dim=1, lat_feature_dim=1) -> None:
        super().__init__()
        self.rec_sample_dim = rec_sample_dim
        self.rec_feature_dim = rec_feature_dim
        self.lat_feature_dim = lat_feature_dim

    def forward(self, targets, inputs):
        """
        Computes and sums all loss components
        """
        return gaussian_nll_loss(targets, inputs, sample_dim=self.sample_dim, feature_dim=self.feature_dim)
    
    def supervised_loss():

        return
    
    def reconstruction_loss():
        return

    def latent_loss():
        return
    



class lr_finder_elbo(nn.Module):
    """
    TODO: remove this class and make it a standard class
    """
    def __init__(self, index_loss, beta_kl=1, beta_index=0, loss_type='diag_nll', ssimulator=None) -> None:
        super(lr_finder_elbo,self).__init__()
        self.beta_kl = beta_kl
        self.beta_index = beta_index
        self.index_loss = index_loss
        self.loss_type = loss_type
        self.ssimulator = ssimulator 
        self.rec_loss_fn = select_rec_loss_fn(self.loss_type)
        pass

    def lr_finder_elbo_inner(self, model_outputs, label):
        dist_params, _, _, rec = model_outputs

        if self.ssimulator.apply_norm:
            label = self.ssimulator.normalize(label)
        
        if len(label.size()) == 2:
            label = label.unsqueeze(2)
        if self.loss_type=="spatial_nll":
            hw = (label.size(3) - rec.size(3)) // 2
            if hw > 0:
                label = crop_s2_input(label, hw)
        rec_loss = self.rec_loss_fn(label, rec)
        loss_sum = rec_loss.mean()
        sigma = dist_params[:, :, 1].squeeze()
        mu = dist_params[:, :, 0].squeeze()
        if self.beta_kl > 0:
            kl_loss = self.beta_kl * kl_tn_uniform(mu, sigma).sum(1).mean()
            loss_sum += kl_loss
        if self.beta_index > 0:
            index_loss = self.beta_index * self.index_loss(label, rec, lossfn=self.rec_loss_fn)
            loss_sum += index_loss
        return loss_sum

    def forward(self, model_outputs, label):
        return self.lr_finder_elbo_inner(model_outputs, label)

def lr_finder_sup_nll(model_outputs, label):
    dist_params, _, _, _ = model_outputs
    sigma = dist_params[:, :, 1].squeeze()
    mu = dist_params[:, :, 0].squeeze()
    loss = truncated_gaussian_nll(label, mu, sigma).mean() 
    return loss