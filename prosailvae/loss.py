from dataclasses import dataclass
import torch 
import torch.nn as nn
from prosailvae.dist_utils import kl_tn_uniform, truncated_gaussian_nll
from utils.utils import select_rec_loss_fn, torch_select_unsqueeze
from utils.image_utils import crop_s2_input

@dataclass
class LossConfig:
    """
    Dataclass to hold loss config.
    """
    supervised:bool = False
    beta_kl:float = 0.0
    beta_index:float = 0.0
    beta_cyclical:float = 0.0
    snap_cyclical:bool = False
    loss_type: str = 'diag_nll'
    lat_loss_type:str = ""
    reconstruction_bands_coeffs:list[int]|None=None
    

def get_nll_dimensions(loss_type):
    simple_losses_1d = ["diag_nll", "hybrid_nll", "lai_nll"]
    if loss_type in simple_losses_1d:
        return 2, 1
    elif loss_type == "spatial_nll":
        return 1, 2 
    else:
        raise NotImplementedError
    

def select_rec_loss_fn(loss_type):
    simple_losses_1d = ["diag_nll", "hybrid_nll", "lai_nll"]
    if loss_type in simple_losses_1d:
        rec_loss_fn = NLLLoss(2,1)
    elif loss_type == "spatial_nll":
        rec_loss_fn = NLLLoss(1,2) 
    elif loss_type == "full_nll":
        rec_loss_fn = full_gaussian_nll_loss
    elif loss_type =='mse':
        rec_loss_fn = mse_loss
    else:
        raise NotImplementedError("Please choose between 'diag_nll' (diagonal covariance matrix) and 'full_nll' (full covariance matrix) for nll loss option.")
    return rec_loss_fn

def gaussian_nll(x:torch.Tensor, mu:torch.Tensor, sigma2:torch.Tensor, eps:float=1e-6, 
                 device:str='cpu', sum_dim:int=1, feature_indexes:None|list[int]=None):
    """
    Gaussian Negative Log-Likelihood
    """    
    eps = torch.tensor(eps).to(device)
    if feature_indexes is None:
        return ((torch.square(x - mu) / torch.max(sigma2, eps)) +
                torch.log(torch.max(sigma2, eps))).sum(sum_dim)
    else:
        loss = []
        for idx in feature_indexes:
            if len(sigma2.size()) != 0:
                idx_sigma2 = torch.max(sigma2.select(dim=sum_dim,index=idx), eps)
            else:
                idx_sigma2 = sigma2
            idx_loss = ((torch.square(x.select(dim=sum_dim,index=idx) 
                                      - mu.select(dim=sum_dim,index=idx)) / idx_sigma2) +
                        torch.log(idx_sigma2)).unsqueeze(sum_dim)
            loss.append(idx_loss)
        loss = torch.cat(loss, dim=sum_dim).sum(sum_dim)
        return loss


def gaussian_nll_loss(tgt, recs, sample_dim=2, feature_dim=1, feature_indexes:list[int]|None=None):

    if len(recs.size()) < 3:
        raise ValueError("recs needs a batch, a feature and a sample dimension")
    elif recs.size(sample_dim)==1:
        rec_err_var=torch.tensor(0.0001).to(tgt.device) # constant variance, enabling computation even with 1 sample
        rec_mu = recs
    else:
        rec_err_var = recs.var(sample_dim, keepdim=True)#.unsqueeze(sample_dim)
        rec_mu = recs.mean(sample_dim, keepdim=True)#.unsqueeze(sample_dim)
        # if feature_dim > sample_dim: # if feature dimension is after sample dimension, 
        #     # reducing it because sample dimension disappeared
        #     feature_dim = feature_dim - 1
    return gaussian_nll(tgt.unsqueeze(sample_dim), rec_mu, rec_err_var, 
                        sum_dim=feature_dim, feature_indexes=feature_indexes).mean()

class NLLLoss(nn.Module):
    """
    nn.Module Loss for NLL
    """
    def __init__(self, 
                 loss_type:str|None=None,
                 sample_dim=2, 
                 feature_dim=1, 
                 feature_indexes:list[int]|None=None) -> None:
        super().__init__()
        if loss_type is not None:
            sample_dim, feature_dim = get_nll_dimensions(loss_type)
        self.sample_dim = sample_dim
        self.feature_dim = feature_dim 
        self.feature_indexes = feature_indexes

    def forward(self, targets, inputs):
        return gaussian_nll_loss(targets, inputs, sample_dim=self.sample_dim, 
                                 feature_dim=self.feature_dim, feature_indexes=self.feature_indexes)

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

    def forward(self, model_outputs, model_inputs):
        """
        Computes and sums all loss components, computed from input data (with or without labels)
        and model outputs.
        """

        return # gaussian_nll_loss(targets, inputs, sample_dim=self.sample_dim, feature_dim=self.feature_dim)
    
    def supervised_loss(self):

        return
    
    def reconstruction_loss(self):

        return

    def latent_loss(self):
        
        return
    



# class lr_finder_elbo(nn.Module):
#     """
#     TODO: remove this class and make it a standard class
#     """
#     def __init__(self, index_loss, beta_kl=1, beta_index=0, loss_type='diag_nll', ssimulator=None) -> None:
#         super(lr_finder_elbo,self).__init__()
#         self.beta_kl = beta_kl
#         self.beta_index = beta_index
#         self.index_loss = index_loss
#         self.loss_type = loss_type
#         self.ssimulator = ssimulator 
#         self.rec_loss_fn = select_rec_loss_fn(self.loss_type)
#         pass

#     def lr_finder_elbo_inner(self, model_outputs, label):
#         dist_params, _, _, rec = model_outputs

#         if self.ssimulator.apply_norm:
#             label = self.ssimulator.normalize(label)
        
#         if len(label.size()) == 2:
#             label = label.unsqueeze(2)
#         if self.loss_type=="spatial_nll":
#             hw = (label.size(3) - rec.size(3)) // 2
#             if hw > 0:
#                 label = crop_s2_input(label, hw)
#         rec_loss = self.rec_loss_fn(label, rec)
#         loss_sum = rec_loss.mean()
#         sigma = dist_params[:, :, 1].squeeze()
#         mu = dist_params[:, :, 0].squeeze()
#         if self.beta_kl > 0:
#             kl_loss = self.beta_kl * kl_tn_uniform(mu, sigma).sum(1).mean()
#             loss_sum += kl_loss
#         if self.beta_index > 0:
#             index_loss = self.beta_index * self.index_loss(label, rec, lossfn=self.rec_loss_fn)
#             loss_sum += index_loss
#         return loss_sum

#     def forward(self, model_outputs, label):
#         return self.lr_finder_elbo_inner(model_outputs, label)

# def lr_finder_sup_nll(model_outputs, label):
#     dist_params, _, _, _ = model_outputs
#     sigma = dist_params[:, :, 1].squeeze()
#     mu = dist_params[:, :, 0].squeeze()
#     loss = truncated_gaussian_nll(label, mu, sigma).mean() 
#     return loss