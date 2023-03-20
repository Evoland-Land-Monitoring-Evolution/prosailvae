import torch

def NDVI(s2_r, eps=torch.tensor(1e-7), bands_dim=1):
    B4 = s2_r.select(bands_dim, 2).unsqueeze(bands_dim)
    B8 = s2_r.select(bands_dim, 6).unsqueeze(bands_dim)
    return (B8 - B4) / (B8 + B4 + eps.to(B8.device))

def mNDVI750(s2_r, eps=torch.tensor(1e-7), bands_dim=1):
    B2 = s2_r.select(bands_dim, 0).unsqueeze(bands_dim)
    B5 = s2_r.select(bands_dim, 3).unsqueeze(bands_dim)
    B6 = s2_r.select(bands_dim, 4).unsqueeze(bands_dim)
    denom = (B6 + B5 - 2 * B2)
    zero_denom_idx = denom.abs() < eps
    denom[zero_denom_idx] = eps * torch.sign(denom[zero_denom_idx])
    return (B6 - B5) / denom

def CRI2(s2_r, eps=torch.tensor(1e-7), bands_dim=1):
    B2 = s2_r.select(bands_dim, 0).unsqueeze(bands_dim)
    B5 = s2_r.select(bands_dim, 3).unsqueeze(bands_dim)
    return 1 / (B2 + eps) - 1 / (B5 + eps)

def NDII(s2_r, eps=torch.tensor(1e-7), bands_dim=1):
    B8 = s2_r.select(bands_dim, 6).unsqueeze(bands_dim)
    B11 = s2_r.select(bands_dim, 8).unsqueeze(bands_dim)
    return (B8 - B11) / (B8 + B11 + eps)

def ND_lma(s2_r, eps=torch.tensor(1e-7), bands_dim=1):
    B11 = s2_r.select(bands_dim, 8).unsqueeze(bands_dim)
    B12 = s2_r.select(bands_dim, 9).unsqueeze(bands_dim)
    return (B12 - B11) / (B12 + B11 + eps)

def LAI_savi(s2_r, eps=torch.tensor(1e-7), bands_dim=1):
    B4 = s2_r.select(bands_dim, 2).unsqueeze(bands_dim)
    B8 = s2_r.select(bands_dim, 6).unsqueeze(bands_dim)
    return - torch.log(torch.tensor(0.371) + torch.tensor(1.5) *(B8 - B4) / (B8 + B4 + torch.tensor(0.5))) / torch.tensor(2.4)

INDEX_DICT = {"NDVI":NDVI, "mNDVI750":mNDVI750, "CRI2":CRI2, "NDII":NDII, "ND_lma":ND_lma, "LAI_savi":LAI_savi}