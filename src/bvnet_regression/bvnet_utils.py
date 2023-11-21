import os

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torchutils.patches import patchify, unpatchify

import prosailvae
from bvnet_regression.bvnet import BVNET
from dataset.bvnet_dataset import load_bvnet_dataset


def initialize_bvnet(
    variable,
    train_loader,
    valid_loader,
    loc_bv,
    scale_bv,
    res_dir,
    n_models=10,
    n_epochs=20,
    lr=1e-3,
):
    best_valid_loss = np.inf
    for i in range(n_models):
        model = BVNET(
            ver="3A",
            variable=variable,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr_scheduler = ReduceLROnPlateau(
            optimizer=optimizer, patience=n_epochs, threshold=0.001
        )
        _, all_valid_losses, all_lr = model.train_model(
            train_loader,
            valid_loader,
            optimizer,
            epochs=n_epochs,
            lr_scheduler=lr_scheduler,
            disable_tqdm=True,
            lr_recompute=n_epochs,
            loc_bv=loc_bv,
            scale_bv=scale_bv,
            res_dir=None,
        )
        if min(all_valid_losses) < best_valid_loss:
            model.save_weights(res_dir)
            best_valid_loss = min(all_valid_losses)
    model.load_weights(res_dir)
    return model


def get_bvnet_dataloader(
    variable="lai",
    valid_ratio=0.05,
    batch_size=2048,
    s2_r=None,
    prosail_vars=None,
    max_samples=50000,
    psoil0=0.3,
):
    if prosail_vars is None or s2_r is None:
        s2_r, prosail_vars = load_bvnet_dataset(
            os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/lai/",
            mode="bvnet",
            psoil0=psoil0,
        )
    s2_r = s2_r[:max_samples, :]
    prosail_vars = prosail_vars[:max_samples, :]
    s2_a = prosail_vars[:, -3:]

    bv = {
        "lai": prosail_vars[:, 6],
        "cab": prosail_vars[:, 1],
        "cw": prosail_vars[:, 4],
        "ccc": prosail_vars[:, 6] * prosail_vars[:, 1],
        "cwc": prosail_vars[:, 6] * prosail_vars[:, 4],
    }
    bv = bv[variable]
    loc_bv = bv.mean(0)
    scale_bv = bv.std(0)
    data_weiss = torch.from_numpy(
        np.concatenate((s2_r, np.cos(np.deg2rad(s2_a)), bv.reshape(-1, 1)), 1)
    )
    seed = 4567895683301
    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)
    idx = torch.randperm(len(bv), generator=g_cpu)

    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)

    n_valid = int(valid_ratio * data_weiss.size(0))
    idx = torch.randperm(data_weiss.size(0), generator=g_cpu)
    data_valid = data_weiss[idx[:n_valid], :].float()
    data_train = data_weiss[idx[n_valid:], :].float()
    train_dataset = TensorDataset(data_train[:, :-1], data_train[:, -1].unsqueeze(1))
    valid_dataset = TensorDataset(data_valid[:, :-1], data_valid[:, -1].unsqueeze(1))

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, num_workers=0, shuffle=True
    )
    if n_valid > 0:
        valid_loader = DataLoader(
            dataset=valid_dataset, batch_size=batch_size, num_workers=0, shuffle=True
        )
    else:
        valid_loader = None
    return train_loader, valid_loader, loc_bv, scale_bv


def get_bvnet_biophyiscal_from_batch(
    batch,
    patch_size=32,
    sensor=None,
    ver=None,
    device="cpu",
    lai_bvnet=None,
    ccc_bvnet=None,
    cwc_bvnet=None,
):
    if ver is None:
        if sensor is None:
            ver = "2"
        elif sensor == "2A":
            ver = "3A"
        elif sensor == "2B":
            ver = "3B"
        else:
            raise ValueError
    elif ver not in ["2", "3A", "3B"]:
        raise ValueError
    bvnet_bands = torch.tensor([1, 2, 3, 4, 5, 7, 8, 9])
    bvnet_angles = torch.tensor([1, 0, 2])
    s2_r, s2_a = batch
    patched_s2_r = patchify(s2_r.squeeze(), patch_size=patch_size, margin=0)
    patched_s2_a = patchify(s2_a.squeeze(), patch_size=patch_size, margin=0)
    patched_lai_image = torch.zeros(
        (patched_s2_r.size(0), patched_s2_r.size(1), 1, patch_size, patch_size)
    )
    patched_ccc_image = torch.zeros(
        (patched_s2_r.size(0), patched_s2_r.size(1), 1, patch_size, patch_size)
    )
    patched_cwc_image = torch.zeros(
        (patched_s2_r.size(0), patched_s2_r.size(1), 1, patch_size, patch_size)
    )
    if lai_bvnet is None:
        lai_bvnet = BVNET(variable="lai", ver=ver, device=device)
        lai_bvnet.set_snap_weights()
    if ccc_bvnet is None:
        ccc_bvnet = BVNET(variable="ccc", ver=ver, device=device)
        ccc_bvnet.set_snap_weights()
    if cwc_bvnet is None:
        cwc_bvnet = BVNET(variable="cwc", ver=ver, device=device)
        cwc_bvnet.set_snap_weights()
    for i in range(patched_s2_r.size(0)):
        for j in range(patched_s2_r.size(1)):
            x = patched_s2_r[i, j, bvnet_bands, ...]
            angles = torch.cos(torch.deg2rad(patched_s2_a[i, j, bvnet_angles, ...]))
            s2_data = torch.cat((x, angles), 0)
            with torch.no_grad():
                lai = torch.clip(
                    lai_bvnet.forward(s2_data.to(lai_bvnet.device), spatial_mode=True),
                    min=0,
                )
                ccc = torch.clip(
                    ccc_bvnet.forward(s2_data.to(ccc_bvnet.device), spatial_mode=True),
                    min=0,
                )  # torch.clip(cab_bvnet.forward(s2_data, spatial_mode=True), min=0) / torch.clip(lai, min=0.1)
                cwc = torch.clip(
                    cwc_bvnet.forward(s2_data.to(cwc_bvnet.device), spatial_mode=True),
                    min=0,
                )  # torch.clip(cw_bvnet.forward(s2_data, spatial_mode=True), min=0) / torch.clip(lai, min=0. 1)
                # lai = bvnet_lai(x, angles, band_dim=0, ver=ver)
            patched_lai_image[i, j, ...] = lai
            patched_ccc_image[i, j, ...] = ccc
            patched_cwc_image[i, j, ...] = cwc
    lai_image = unpatchify(patched_lai_image)[:, : s2_r.size(2), : s2_r.size(3)]
    ccc_image = unpatchify(patched_ccc_image)[:, : s2_r.size(2), : s2_r.size(3)]
    cwc_image = unpatchify(patched_cwc_image)[:, : s2_r.size(2), : s2_r.size(3)]
    return lai_image.cpu(), ccc_image.cpu(), cwc_image.cpu()


def get_bvnet_biophyiscal_from_pixellic_batch(
    batch,
    sensor=None,
    ver=None,
    device="cpu",
    lai_bvnet=None,
    ccc_bvnet=None,
    cwc_bvnet=None,
):
    if ver is None:
        if sensor is None:
            ver = "2"
        elif sensor == "2A":
            ver = "3A"
        elif sensor == "2B":
            ver = "3B"
        else:
            raise ValueError
    elif ver not in ["2", "3A", "3B"]:
        raise ValueError
    bvnet_bands = torch.tensor([1, 2, 3, 4, 5, 7, 8, 9])
    bvnet_angles = torch.tensor([1, 0, 2])
    s2_r, s2_a = batch
    x = s2_r[:, bvnet_bands]
    angles = torch.cos(torch.deg2rad(s2_a[:, bvnet_angles]))
    s2_data = torch.cat((x, angles), 1)
    with torch.no_grad():
        if lai_bvnet is None:
            lai_bvnet = BVNET(variable="lai", ver=ver, device=device)
            lai_bvnet.set_snap_weights()
        if ccc_bvnet is None:
            ccc_bvnet = BVNET(variable="ccc", ver=ver, device=device)
            ccc_bvnet.set_snap_weights()
        if cwc_bvnet is None:
            cwc_bvnet = BVNET(variable="cwc", ver=ver, device=device)
            cwc_bvnet.set_snap_weights()

        lai = torch.clip(
            lai_bvnet.forward(s2_data.to(lai_bvnet.device), spatial_mode=False), min=0
        )
        ccc = torch.clip(
            ccc_bvnet.forward(s2_data.to(ccc_bvnet.device), spatial_mode=False), min=0
        )
        cwc = torch.clip(
            cwc_bvnet.forward(s2_data.to(cwc_bvnet.device), spatial_mode=False), min=0
        )
    return lai, ccc, cwc
