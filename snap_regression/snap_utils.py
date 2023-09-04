import torch
from torchutils.patches import patchify, unpatchify
from snap_regression.snap_nn import SnapNN

def get_weiss_biophyiscal_from_batch(batch, patch_size=32, sensor=None, ver=None, device='cpu', 
                                              lai_snap=None, ccc_snap=None, cwc_snap=None):
    if ver is None:
        if sensor is None:
            ver = "2"
        elif sensor =="2A":
            ver = "3A"
        elif sensor == "2B":
            ver = "3B"
        else:
            raise ValueError
    elif ver not in ["2", "3A", "3B"]:
        raise ValueError
    weiss_bands = torch.tensor([1, 2, 3, 4, 5, 7, 8, 9])
    weiss_angles = torch.tensor([1, 0, 2])
    s2_r, s2_a = batch
    patched_s2_r = patchify(s2_r.squeeze(), patch_size=patch_size, margin=0)
    patched_s2_a = patchify(s2_a.squeeze(), patch_size=patch_size, margin=0)
    patched_lai_image = torch.zeros((patched_s2_r.size(0), patched_s2_r.size(1), 1, patch_size, patch_size))
    patched_ccc_image = torch.zeros((patched_s2_r.size(0), patched_s2_r.size(1), 1, patch_size, patch_size))
    patched_cwc_image = torch.zeros((patched_s2_r.size(0), patched_s2_r.size(1), 1, patch_size, patch_size))
    if lai_snap is None:
        lai_snap = SnapNN(variable='lai', ver=ver, device=device)
        lai_snap.set_weiss_weights()
    if ccc_snap is None:
        ccc_snap = SnapNN(variable='ccc', ver=ver, device=device)
        ccc_snap.set_weiss_weights()
    if cwc_snap is None:
        cwc_snap = SnapNN(variable='cwc', ver=ver, device=device)
        cwc_snap.set_weiss_weights()
    for i in range(patched_s2_r.size(0)):
        for j in range(patched_s2_r.size(1)):
            x = patched_s2_r[i, j, weiss_bands, ...]
            angles = torch.cos(torch.deg2rad(patched_s2_a[i, j, weiss_angles, ...]))
            s2_data = torch.cat((x, angles),0)
            with torch.no_grad():
                lai = torch.clip(lai_snap.forward(s2_data.to(lai_snap.device), spatial_mode=True), min=0)
                ccc = torch.clip(ccc_snap.forward(s2_data.to(ccc_snap.device), spatial_mode=True), min=0) # torch.clip(cab_snap.forward(s2_data, spatial_mode=True), min=0) / torch.clip(lai, min=0.1)
                cwc = torch.clip(cwc_snap.forward(s2_data.to(cwc_snap.device), spatial_mode=True), min=0) # torch.clip(cw_snap.forward(s2_data, spatial_mode=True), min=0) / torch.clip(lai, min=0. 1)
                # lai = weiss_lai(x, angles, band_dim=0, ver=ver)
            patched_lai_image[i,j,...] = lai
            patched_ccc_image[i,j,...] = ccc
            patched_cwc_image[i,j,...] = cwc
    lai_image = unpatchify(patched_lai_image)[:,:s2_r.size(2),:s2_r.size(3)]
    ccc_image = unpatchify(patched_ccc_image)[:,:s2_r.size(2),:s2_r.size(3)]
    cwc_image = unpatchify(patched_cwc_image)[:,:s2_r.size(2),:s2_r.size(3)]
    return lai_image.cpu(), ccc_image.cpu(), cwc_image.cpu()


def get_weiss_biophyiscal_from_pixellic_batch(batch, sensor=None, ver=None, device='cpu', 
                                              lai_snap=None, ccc_snap=None, cwc_snap=None,):
    if ver is None:
        if sensor is None:
            ver = "2"
        elif sensor =="2A":
            ver = "3A"
        elif sensor == "2B":
            ver = "3B"
        else:
            raise ValueError
    elif ver not in ["2", "3A", "3B"]:
        raise ValueError
    weiss_bands = torch.tensor([1,2,3,4,5,7,8,9])
    weiss_angles = torch.tensor([1,0,2])
    s2_r, s2_a = batch
    x = s2_r[:, weiss_bands]
    angles = torch.cos(torch.deg2rad(s2_a[:, weiss_angles]))
    s2_data = torch.cat((x, angles), 1)
    with torch.no_grad():
        if lai_snap is None:
            lai_snap = SnapNN(variable='lai', ver=ver, device=device)
            lai_snap.set_weiss_weights()
        if ccc_snap is None:
            ccc_snap = SnapNN(variable='ccc', ver=ver, device=device)
            ccc_snap.set_weiss_weights()
        if cwc_snap is None:
            cwc_snap = SnapNN(variable='cwc', ver=ver, device=device)
            cwc_snap.set_weiss_weights()

        lai = torch.clip(lai_snap.forward(s2_data.to(lai_snap.device), spatial_mode=False), min=0)
        ccc = torch.clip(ccc_snap.forward(s2_data.to(ccc_snap.device), spatial_mode=False), min=0) 
        cwc = torch.clip(cwc_snap.forward(s2_data.to(cwc_snap.device), spatial_mode=False), min=0) 
    return lai, ccc, cwc