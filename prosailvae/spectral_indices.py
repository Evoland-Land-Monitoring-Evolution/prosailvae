import torch

def NDVI(s2_r, eps=torch.tensor(1e-7), bands_dim=1):
    B4 = s2_r.select(bands_dim, 2).unsqueeze(bands_dim)
    B8 = s2_r.select(bands_dim, 6).unsqueeze(bands_dim)
    num = (B8 - B4)
    denom = (B8 + B4)
    non_zero_denom_idx = denom.abs() > eps
    ndvi = - torch.ones_like(B4)
    ndvi[non_zero_denom_idx] = num[non_zero_denom_idx] / denom[non_zero_denom_idx]
    return torch.clamp(ndvi, min=torch.tensor(-1), max=torch.tensor(1))

def mNDVI750(s2_r, eps=torch.tensor(1e-7), bands_dim=1):
    B2 = s2_r.select(bands_dim, 0).unsqueeze(bands_dim)
    B5 = s2_r.select(bands_dim, 3).unsqueeze(bands_dim)
    B6 = s2_r.select(bands_dim, 4).unsqueeze(bands_dim)
    denom = (B6 + B5 - 2 * B2)
    num = (B6 - B5)
    mndvi750 = - torch.ones_like(B6)
    non_zero_denom_idx = denom.abs() > eps
    mndvi750[non_zero_denom_idx] = num[non_zero_denom_idx] / denom[non_zero_denom_idx]
    return torch.clamp(mndvi750, min=torch.tensor(-1), max=torch.tensor(1))

def CRI2(s2_r, eps=torch.tensor(1e-7), bands_dim=1):
    B2 = s2_r.select(bands_dim, 0).unsqueeze(bands_dim)
    B5 = s2_r.select(bands_dim, 3).unsqueeze(bands_dim)
    cri2 = torch.zeros_like(B2)
    b2_and_b5_sup_0_idx = torch.logical_and(B2 > eps, B5 >= B2)
    cri2[b2_and_b5_sup_0_idx] = 1 / (B2[b2_and_b5_sup_0_idx]) - 1 / (B5[b2_and_b5_sup_0_idx])
    return torch.clamp(cri2, max=torch.tensor(20))

def NDII(s2_r, eps=torch.tensor(1e-7), bands_dim=1):
    B8 = s2_r.select(bands_dim, 6).unsqueeze(bands_dim)
    B11 = s2_r.select(bands_dim, 8).unsqueeze(bands_dim)
    num = (B8 - B11)
    denom = (B8 + B11)
    non_zero_denom_idx = denom.abs() > eps
    ndii = - torch.ones_like(B8)
    ndii[non_zero_denom_idx] = num[non_zero_denom_idx] / denom[non_zero_denom_idx]
    return torch.clamp(ndii, min=torch.tensor(-1), max=torch.tensor(1))

def ND_lma(s2_r, eps=torch.tensor(1e-7), bands_dim=1):
    B11 = s2_r.select(bands_dim, 8).unsqueeze(bands_dim)
    B12 = s2_r.select(bands_dim, 9).unsqueeze(bands_dim)
    num = (B12 - B11)
    denom = (B12 + B11)
    non_zero_denom_idx = denom.abs() > eps
    nd_lma = - torch.ones_like(B12)
    nd_lma[non_zero_denom_idx] = num[non_zero_denom_idx] / denom[non_zero_denom_idx]
    return torch.clamp(nd_lma, min=torch.tensor(-1), max=torch.tensor(1))

def LAI_savi(s2_r, eps=torch.tensor(1e-7), bands_dim=1):
    B4 = s2_r.select(bands_dim, 2).unsqueeze(bands_dim)
    B8 = s2_r.select(bands_dim, 6).unsqueeze(bands_dim)
    return - torch.log(torch.abs(torch.tensor(0.371) + torch.tensor(1.5) * (B8 - B4) / (B8 + B4 + torch.tensor(0.5))) + eps) / torch.tensor(2.4)

INDEX_DICT = {"NDVI":NDVI, "NDII":NDII, "ND_lma":ND_lma, "LAI_savi":LAI_savi}#} #"mNDVI750":mNDVI750, "CRI2":CRI2,

def get_spectral_idx(s2_r, eps=torch.tensor(1e-4), bands_dim=1, index_dict=INDEX_DICT):
    spectral_idx = []
    for idx_name, idx_fn in index_dict.items():
        idx = idx_fn(s2_r, eps=eps, bands_dim=bands_dim)
        if not s2_r.isnan().any() or s2_r.isinf().any():
            if idx.isnan().any() or idx.isinf().any():
                raise ValueError(f"{idx_name} has NaN or infinite values!")
        spectral_idx.append(idx)
    return torch.cat(spectral_idx, axis=bands_dim)
