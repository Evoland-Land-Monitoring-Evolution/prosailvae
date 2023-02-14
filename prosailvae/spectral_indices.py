import torch

def NDVI(s2_r, eps=torch.tensor(1e-7)):
    if len(s2_r.size())==2:
        B4 = s2_r[:,2]
        B8 = s2_r[:,6]
    elif len(s2_r.size())==3:
        B4 = s2_r[:,2,:]
        B8 = s2_r[:,6,:]
    else:
        raise NotImplementedError
    return (B8 - B4) / (B8 + B4 + eps)

def mNDVI750(s2_r, eps=torch.tensor(1e-7)):
    if len(s2_r.size())==2:
        B2 = s2_r[:,0]
        B5 = s2_r[:,3]
        B6 = s2_r[:,4]
    elif len(s2_r.size())==3:
        B2 = s2_r[:,0,:]
        B5 = s2_r[:,3,:]
        B6 = s2_r[:,4,:]
    else:
        raise NotImplementedError
    denom = (B6 + B5 - 2 * B2)
    zero_denom_idx = denom.abs()<eps
    denom[zero_denom_idx] = eps * torch.sign(denom[zero_denom_idx])
    return (B6 - B5) / denom

def CRI2(s2_r, eps=torch.tensor(1e-7)):
    if len(s2_r.size())==2:
        B2 = s2_r[:,0]
        B5 = s2_r[:,3]
    elif len(s2_r.size())==3:
        B2 = s2_r[:,0,:]
        B5 = s2_r[:,3,:]
    else:
        raise NotImplementedError
    return 1 / (B2 + eps) - 1 / (B5 + eps)

def NDII(s2_r, eps=torch.tensor(1e-7)):
    if len(s2_r.size())==2:
        B8 = s2_r[:,6]
        B11 = s2_r[:,8]
    elif len(s2_r.size())==3:
        B8 = s2_r[:,6,:]
        B11 = s2_r[:,8,:]
    else:
        raise NotImplementedError
    return (B8 - B11) / (B8 + B11 + eps)

def ND_lma(s2_r, eps=torch.tensor(1e-7)):
    if len(s2_r.size())==2:
        B11 = s2_r[:,8]
        B12 = s2_r[:,9]
    elif len(s2_r.size())==3:
        B11 = s2_r[:,8,:]
        B12 = s2_r[:,9,:]
    else:
        raise NotImplementedError
    return (B12 - B11) / (B12 + B11 + eps)

def LAI_savi(s2_r, eps=torch.tensor(1e-7)):
    if len(s2_r.size())==2:
        B4 = s2_r[:,2]
        B8 = s2_r[:,6]
    elif len(s2_r.size())==3:
        B4 = s2_r[:,2,:]
        B8 = s2_r[:,6,:]
    else:
        raise NotImplementedError
    return -torch.log(torch.tensor(0.371) + torch.tensor(1.5) *(B8 - B4) / (B8 + B4 + torch.tensor(0.5))) / torch.tensor(2.4)