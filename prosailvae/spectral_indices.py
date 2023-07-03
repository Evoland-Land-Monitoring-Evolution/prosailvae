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
    denom[zero_denom_idx] = eps # * torch.sign(denom[zero_denom_idx])
    idx =  (B6 - B5) / denom
    if idx.isnan().any() or idx.isinf().any():
        pass
    return idx

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
    return - torch.log(torch.abs(torch.tensor(0.371) + torch.tensor(1.5) * (B8 - B4) / (B8 + B4 + torch.tensor(0.5))) + eps) / torch.tensor(2.4)

INDEX_DICT = {"NDVI":NDVI, "CRI2":CRI2, "NDII":NDII, "ND_lma":ND_lma, "LAI_savi":LAI_savi}#}"mNDVI750":mNDVI750,

def get_spectral_idx(s2_r, eps=torch.tensor(1e-7), bands_dim=1, index_dict=INDEX_DICT):
    spectral_idx = []
    for idx_name, idx_fn in index_dict.items():
        idx = idx_fn(s2_r, eps=eps, bands_dim=bands_dim)
        if idx.isnan().any() or idx.isinf().any():
            raise ValueError(f"{idx_name} has NaN or infinite values!")
        spectral_idx.append(idx)
    return torch.cat(spectral_idx, axis=bands_dim)
