#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 11:03:21 2022

min_sample_nb@author: yoel
"""
import torch
from math import pi
import scipy.interpolate
import torch.nn.functional as F

def rectify_pdf_from_mode(support, mode):
    sampling = (support[-1] - support[0]) / len(support)
    closest2mode_idx = (support - mode).abs().argmin()
    pdf = torch.zeros_like(support)
    pdf[closest2mode_idx] = 1/sampling
    return pdf 

def normal_pdf(x, mu, sig, eps=1e-9):
    return torch.exp((-(x-mu)**2)/(2*sig**2 + eps)) / ((sig + eps) * (2*pi)**0.5)

def normal_cdf(x, mu, sigma):
    return 0.5 * (1 + torch.erf((x-mu)/(sigma*2**0.5))) 

def get_normal_pdf(mu, sigma, support_sampling=0.01, interval_bound=1, eps=1e-9):
    support = torch.arange(mu - interval_bound, mu + interval_bound, support_sampling)
    pdf = normal_pdf(support, mu, sigma, eps=eps)
    if pdf.sum()==0:
        pdf = torch.tensor(singular_pdf(support.detach().numpy(), 
                                                      mu.squeeze()))
    return pdf, support

def kl_tn_uniform(mu, sigma, lower=torch.tensor(0.0), upper=torch.tensor(1.0)):
    alpha = (lower - mu) / sigma
    beta = (upper - mu) / sigma
    phi_alpha = torch.exp(-alpha**2/2) / torch.sqrt(torch.tensor(2*pi))
    phi_beta = torch.exp(-beta**2/2) / torch.sqrt(torch.tensor(2*pi))
    Phi_alpha = 0.5 * (1 + torch.erf(alpha/(2**0.5))) 
    Phi_beta = 0.5 * (1 + torch.erf(beta/(2**0.5))) 
    Z = Phi_beta - Phi_alpha
    kl = - torch.log(sigma.float() * Z.float()) - torch.log(torch.tensor(2*pi))/2 - 1/2 - (alpha * phi_alpha - beta* phi_beta)/(2*Z) + torch.log(upper - lower) 
    return kl

def kl_tntn(mu1, sigma1, mu2, sigma2, lower=torch.tensor(0.0), upper=torch.tensor(1.0)):
    alpha1 = (lower - mu1) / sigma1
    beta1 = (upper - mu1) / sigma1
    phi_alpha1 = torch.exp(-alpha1**2/2) / torch.sqrt(torch.tensor(2*pi))
    phi_beta1 = torch.exp(-beta1**2/2) / torch.sqrt(torch.tensor(2*pi))
    Phi_alpha1 = 0.5 * (1 + torch.erf(alpha1/(2**0.5))) 
    Phi_beta1 = 0.5 * (1 + torch.erf(beta1/(2**0.5))) 
    eta1 = Phi_beta1 - Phi_alpha1
    alpha2 = (lower - mu2) / sigma2
    beta2 = (upper - mu2) / sigma2
    Phi_alpha2 = 0.5 * (1 + torch.erf(alpha2/(2**0.5))) 
    Phi_beta2 = 0.5 * (1 + torch.erf(beta2/(2**0.5))) 
    eta2 = Phi_beta2 - Phi_alpha2
    sigma22 = sigma2.pow(2)
    sigma12 = sigma1.pow(2)
    K1 = alpha1 * phi_alpha1 - beta1 * phi_beta1
    N1 = phi_alpha1 - phi_beta1
    kl = - 1/2
    kl += torch.log(sigma2 * eta2) - torch.log(sigma1 * eta1) 
    kl += - K1 / (2 * eta1) * (1 - sigma12 / sigma22)
    kl += (mu1 - mu2).pow(2) / 2 / sigma22
    kl += sigma12 / 2 / sigma22
    kl += (mu1 - mu2) * sigma1 / eta1 / sigma22 * N1
    return kl


def numerical_kl_tn_uniform(mu, sigma, support=torch.arange(0,1, 0.001), eps=1e-9, lower=torch.tensor(0), upper=torch.tensor(1)):
    p = truncated_gaussian_pdf(support, mu, sigma, eps=eps, lower=lower, upper=upper)
    log_p = torch.log(p)
    kl = (p * log_p).sum()/len(support)
    return kl

def numerical_kl_from_pdf(pdf_p, pdf_q, dx, eps=1e-4):
    """
    Computes numerical Kullback-Leibler divegence from two densities with assumed same support
    """
    pdf_p_non_zero_idx = torch.where(pdf_p!=0)[0]
    pdf_p = pdf_p[pdf_p_non_zero_idx]
    pdf_q = pdf_q[pdf_p_non_zero_idx]
    pdf_q[pdf_q==0] = eps

    return ((pdf_p * pdf_p.log()).sum() - (pdf_p * pdf_q.log()).sum()).item() * dx

def truncated_gaussian_pdf(x, mu, sigma, eps=1e-9, lower=torch.tensor(0), upper=torch.tensor(1)):
    return normal_pdf(x, mu, sigma)  / (normal_cdf(upper*torch.ones_like(mu), mu, sigma) 
                                      - normal_cdf(lower*torch.ones_like(mu), mu, sigma) + torch.tensor(eps)) * (x>=lower) * (x<=upper)

def truncated_gaussian_cdf(x, mu, sigma, eps=1e-9, lower=torch.tensor(0), upper=torch.tensor(1)):
    return  (x>upper) + (x>=lower) * (x<=upper) * (normal_cdf(x, mu, sigma) - 
                                                   normal_cdf(lower, mu, sigma)) / (normal_cdf(upper, mu, sigma) - 
                                                                                    normal_cdf(lower, mu, sigma) + torch.tensor(eps))

def truncated_gaussian_nll(x, mu, sig, eps=1e-9, reduction='sum', lower=torch.tensor(0), upper=torch.tensor(1)):
    likelihood = truncated_gaussian_pdf(x, mu, sig, eps=eps, lower=lower, upper=upper)
    nll = -torch.log(likelihood + torch.tensor(eps))
    if reduction =="sum":
        nll = nll.sum(axis=1)
    elif reduction == "lai":
        nll = nll[:,6,:]
    if nll.isinf().any() or nll.isnan().any():
        raise ValueError()
    return nll
def get_u_bounds(mu, sigma, n_sigma=4, lower=torch.tensor(0), upper=torch.tensor(1)):
    u_ubound = truncated_gaussian_cdf(mu + n_sigma * sigma, mu, sigma, lower=lower, upper=upper)
    u_lbound = truncated_gaussian_cdf(mu - n_sigma * sigma, mu, sigma, lower=lower, upper=upper)
    return u_ubound, u_lbound
    
def sample_truncated_gaussian(mu, sigma, n_samples=1, n_sigma=4, lower=torch.tensor(0.0), upper=torch.tensor(1.0)):
    u_ubound, u_lbound = get_u_bounds(mu, sigma, n_sigma=n_sigma, lower=lower, upper=upper)
    u_dist = torch.distributions.uniform.Uniform(u_lbound, u_ubound)
    mu = mu.repeat(1, 1, n_samples) 
    sigma = sigma.repeat(1, 1, n_samples) 
    n_dist = torch.distributions.normal.Normal(torch.zeros_like(mu), 
                                                torch.ones_like(sigma))
    if n_samples == 1:
        u = u_dist.rsample()#.permute(1,2,0)
    else:
        u = u_dist.rsample(torch.tensor([n_samples]))
        if not len(u.squeeze().size()) == 1:
            u = u.squeeze(3).permute(1,2,0)
        else:
            u=u.squeeze()
    z = mu + sigma * n_dist.icdf(n_dist.cdf((lower * torch.ones_like(mu) - mu)/sigma) + 
                                u * (n_dist.cdf((upper * torch.ones_like(mu) - mu)/sigma) 
                                    - n_dist.cdf((lower * torch.ones_like(mu) - mu)/sigma)))
    return z

def get_truncated_gaussian_pdf(mu, sigma, support_sampling=0.001, eps=1e-9, lower=torch.tensor(0), upper=torch.tensor(1)):
    support = torch.arange(support_sampling, 1, support_sampling)
    return truncated_gaussian_pdf(support, mu, sigma, eps=eps, lower=lower, upper=upper), support


def truncated_gaussians_max(x, mu, sigma, eps=1e-9, lower=torch.tensor(0), upper=torch.tensor(1)):
    assert len(mu)==len(sigma)
    tot_pdf = 0
    cdf_prod = 1
    for i in range(len(mu)):
        max_cdf = truncated_gaussian_cdf(x, mu[i], sigma[i], eps=eps, lower=lower, upper=upper)
        cdf_prod = cdf_prod * max_cdf
        max_pdf = truncated_gaussian_pdf(x, mu[i], sigma[i], eps=eps, lower=lower, upper=upper)
        tot_pdf = max_pdf / (max_cdf + eps) + tot_pdf
    tot_pdf = tot_pdf * cdf_prod
    return tot_pdf


def get_latent_ordered_truncated_pdfs(mu, sigma, n_sigma_interval, support_sampling, max_matrix, latent_dim=6, eps=1e-12, lower=torch.tensor(0), upper=torch.tensor(1)):
    interval_bound = max(n_sigma_interval * sigma.max().item(), 1)
    lat_pdf_support = torch.arange(-interval_bound, interval_bound, support_sampling).to(mu.device)
    len_pdf = len(lat_pdf_support)
    batch_size=mu.size(0)
    supports = lat_pdf_support.view(1, 1, -1).repeat(batch_size, latent_dim, 1).to(mu.device)
    pdfs = torch.zeros((batch_size, latent_dim, len_pdf)).to(mu.device)
    pdfs_at_z = truncated_gaussian_pdf(supports, mu, sigma, eps=eps, lower=lower, upper=upper)
    cdfs_at_z = truncated_gaussian_cdf(supports, mu, sigma, eps=eps, lower=lower, upper=upper)
    for i in range(latent_dim):
        max_mat_col = max_matrix[:,i]
        max_mat_mask = max_mat_col.view(1,-1).repeat(pdfs.size(2), 1)
        selected_cdfs = torch.masked_select(cdfs_at_z.transpose(1,2), max_mat_mask.bool()).view(batch_size, pdfs.size(2),-1)
        prod_cdfs = torch.prod(selected_cdfs,2, keepdim=True).view(pdfs.size(2),-1,batch_size)
        prod_pdfs_invcdfs = max_mat_mask.transpose(0,1) * pdfs_at_z / (cdfs_at_z + eps)
        pdfs[:, i, :] = (prod_cdfs.view(batch_size, -1) * prod_pdfs_invcdfs.sum(axis=1))
    return pdfs, supports

def ordered_truncated_gaussian_nll(z, mu, sigma, max_matrix, eps = 1e-12, device='cpu', lower=torch.tensor(0), upper=torch.tensor(1)):
    
    n_lat = max_matrix.size(1)
    max_pdf_lat = torch.zeros(z.size(0), n_lat).to(device)
    for i in range(n_lat):
        pdfs_at_zi = truncated_gaussian_pdf(z[:,i].view(-1,1), mu, sigma, eps=eps, lower=lower, upper=upper).to(device)  
        cdfs_at_zi = truncated_gaussian_cdf(z[:,i].view(-1,1), mu, sigma, eps=eps, lower=lower, upper=upper).to(device)
        max_mat_col = max_matrix[:,i]
        max_mat_mask = max_mat_col.repeat(z.size(0),1)
        # masked select is probably triggering some errors, see : https://discuss.pytorch.org/t/logbackward-returned-nan-values-in-its-0th-output/92820/7
        selected_cdfs = torch.masked_select(cdfs_at_zi, max_mat_mask.bool()).view(z.size(0),-1)
        selected_pdfs = torch.masked_select(pdfs_at_zi, max_mat_mask.bool()).view(z.size(0),-1)
        prod_cdfs = torch.prod(selected_cdfs, 1).view(-1,1)
        prod_pdfs_invcdfs = selected_pdfs / (selected_cdfs + eps)
        max_pdf_lat[:,i] = (prod_cdfs * prod_pdfs_invcdfs.sum(axis=1).view(-1,1)).squeeze()
    #TODO: correct error in backward pass here.
    log_max_pdf_i = torch.log(max_pdf_lat + eps)
    nll = -log_max_pdf_i
    return nll.sum(axis=1) 
    
def pdfs2cdfs(pdfs):
    cdfs = pdfs.cumsum(2)
    cdfs = cdfs / cdfs[:,:,-1].view(cdfs.size(0),cdfs.size(1),1)
    return cdfs

def cdf2quantile(cdf, support, alpha=[0.5]):
    distribution = scipy.interpolate.interp1d(cdf, support, bounds_error=False, fill_value='extrapolate')
    return distribution(alpha)

def cdfs2quantiles(cdfs, supports, alpha=[0.5]):
    quantiles = torch.zeros((cdfs.size(0),len(alpha)))
    for i in range(cdfs.size(0)):
        quantiles[i,:] = torch.from_numpy(cdf2quantile(cdfs[i,:], supports[i,:], alpha=alpha))
    return quantiles

def convolve_pdfs(pdfs, supports, transfer_mat_line, n_pdf_sample_points=2001, support_max=None):
    batch_size=pdfs.size(0)
    if transfer_mat_line.count_nonzero().item()==0:
        
        print("WARNING : all null coefficients in transfer matrix line")
        return None, None
    if transfer_mat_line.count_nonzero().item()==1:
        idx_nonzero = transfer_mat_line.nonzero().item()
        convolved_pdf, convolved_pdf_support = scale_pdf(pdfs[:,idx_nonzero,:].view(batch_size,1,-1),
                                                            supports[:,idx_nonzero,:].view(batch_size,1,-1),
                                                            transfer_mat_line[idx_nonzero].item(),
                                                            0, support_max=support_max)
        convolved_pdf, convolved_pdf_support = resample_pdf(convolved_pdf,
                                                            convolved_pdf_support,
                                                            n_pdf_sample_points=n_pdf_sample_points)
        if convolved_pdf.sum()==0:
            raise NotImplementedError
        return convolved_pdf.reshape(batch_size,-1), convolved_pdf_support.reshape(batch_size,-1)
    else:
        raise NotImplementedError()

def check_pdf(pdf, sampling):
    return pdf.sum() * sampling

def singular_pdf(support, mode_value):
    
    pdf = torch.zeros_like(support)
    dx = support[:,1] - support[:,0] # assuming constant sampling
    mask_mode_support = (support - mode_value).abs().argmin(1)[0]
    pdf[torch.arange(0,pdf.size(0),1),mask_mode_support]=1/dx
    return pdf

def resample_pdf(pdf, support, n_pdf_sample_points=1001):
    while len(pdf.size())<3:
        pdf=pdf.unsqueeze(0)
    while len(support.size())<3:
        support=support.unsqueeze(0)
    resampled_pdf = F.interpolate(pdf, size=n_pdf_sample_points, mode='linear', align_corners=True)
    resampled_support = F.interpolate(support, size=n_pdf_sample_points, mode='linear', align_corners=True)
    
    zero_pdfs_idx = torch.where(resampled_pdf.sum(2)==0)[0]
    if len(zero_pdfs_idx)>0:
        support_zero_idx = support[zero_pdfs_idx,:,:]
        mode = torch.gather(support_zero_idx, 2, 
                            pdf[zero_pdfs_idx,:,:].argmax(2).view(-1,1,len(zero_pdfs_idx))).view(len(zero_pdfs_idx)*support.size(1),-1)
        s_pdf = singular_pdf(resampled_support[zero_pdfs_idx,:,:].view(len(zero_pdfs_idx)*support.size(1),-1), mode)
        resampled_pdf[zero_pdfs_idx,:,:] = s_pdf.view(len(zero_pdfs_idx),support.size(1),-1)
    return resampled_pdf, resampled_support


def scale_pdf(pdf, support, support_scale, support_min, support_max:float|None=None):
    """
    pdf dimension and support should be Batch x latent nb x latent domain
    """
    if support_max is None:
        support_max = support.max()
    assert support_scale != 0
    ext_support = support * abs(support_scale) + support_min
    if support_scale < 0: #ensuring ascending order
        pdf = pdf.flip(0)
        # ext_support = ext_support.flip(0)
    pdf = pdf / abs(support_scale)
    sampling = ext_support[0,0,1] - ext_support[0,0,0]
    if support_max is None:
        support_max =  max(abs(ext_support[-1]), abs(ext_support[0]))
    else:
        # assuming same support
        support_max = max(support_max, abs(ext_support[0,0,-1]), abs(ext_support[0,0,0]))
    upper_support = torch.arange(ext_support[0,0,-1] + sampling,
                                 support_max + sampling,
                                 sampling).to(pdf.device).repeat(pdf.size(0),
                                                                  pdf.size(1), 1)
    lower_support = torch.arange(- support_max, ext_support[0,0,0],
                                  sampling).to(pdf.device).repeat(pdf.size(0),
                                                                  pdf.size(1), 1)
    upper_pdf = torch.zeros((pdf.size(0), pdf.size(1), upper_support.size(2))).to(pdf.device)
    lower_pdf = torch.zeros((pdf.size(0), pdf.size(1), lower_support.size(2))).to(pdf.device)

    ext_pdf = torch.cat([lower_pdf, pdf, upper_pdf], axis=2)
    ext_support = torch.cat([lower_support, ext_support, upper_support], axis=2)
    return ext_pdf, ext_support
