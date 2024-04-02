import numpy as np
import rasterio as rio
import torch
from affine import Affine
from torchutils.patches import patchify, unpatchify

from ..datamodules.mmdc_interface import MMDC_DATA_COMPONENTS, mmdc2pvae_batch


def unbatchify(tensor: torch.Tensor, batch_size=1) -> torch.Tensor:
    """
    Reshapes tensor into a patch-like tensor.
    Assumes that the patch spatial dimensions were put into the batch dimension.
    """
    n_tensor_dim = len(tensor.size())
    if n_tensor_dim == 3:
        patch_size = torch.sqrt(torch.tensor(tensor.size(0) / batch_size)).int().item()
        n_feat = tensor.size(1)
        n_samples = tensor.size(2)
        return tensor.reshape(
            batch_size, patch_size, patch_size, n_feat, n_samples
        ).permute(0, 4, 3, 1, 2)

    raise NotImplementedError


def crop_s2_input(s2_input: torch.Tensor, hw_crop: int = 0) -> torch.Tensor:
    """
    Crops pixels on the border of an image tensor.
    Assumes at least a 3D Tensor, whose spatial dimensions are the last two
    """
    if hw_crop == 0:
        return s2_input
    return s2_input[..., hw_crop:-hw_crop, hw_crop:-hw_crop]


def batchify_batch_latent(tensor: torch.Tensor):
    """
    Puts a patch tensor into a batched form.
    """
    # Input dim (B x 2L x H x W)
    tensor = tensor.permute(0, 2, 3, 1)
    return tensor.reshape(-1, tensor.size(3))  # Output dim (BxHxW) x 2L


def get_invalid_symetrical_padding(enc_kernel_sizes):
    """
    Computes how much border pixels are lost due to convolutions
    """
    hw = 0
    for kernel_size in enumerate(enc_kernel_sizes):
        hw += kernel_size // 2
    return hw


def rgb_render(
    data: np.ndarray,
    clip: int = 2,
    bands: list[int] | None = None,
    norm: bool = True,
    dmin: np.ndarray = None,
    dmax: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for visualization with matplot lib

    :param data: nd_array of shape [bands, w, h]
    :param clip: clip percentile (between 0 and 100). Ignored if norm is False
    :bands: List of bands to extract (len is 1 or 3 for RGB)
    :norm: If true, clip a percentile at each end

    :returns: a tuple of data ready for matplotlib, dmin, dmax
    """
    if bands is None:
        bands = [2, 1, 0]
    assert len(bands) == 1 or len(bands) == 3
    assert clip >= 0 and clip <= 100

    # Extract bands from data
    data_ready = np.take(data, bands, axis=0)

    # If normalization is on
    if norm:
        # Rescale and clip data according to percentile
        if dmin is None:
            dmin = np.percentile(data_ready, clip, axis=(1, 2))
        if dmax is None:
            dmax = np.percentile(data_ready, 100 - clip, axis=(1, 2))
        data_ready = np.clip(
            (np.einsum("ijk->jki", data_ready) - dmin) / (dmax - dmin), 0, 1
        )

    else:
        data_ready = np.einsum("ijk->jki", data_ready)

    # Strip of one dimension if number of bands is 1
    if data_ready.shape[-1] == 1:
        data_ready = data_ready[:, :, 0]

    return data_ready, dmin, dmax


def get_encoded_image_from_batch(
    batch,
    PROSAIL_VAE,
    patch_size=32,
    bands=None,
    mode="lat_mode",
    padding=False,
    no_rec=False,
):
    if bands is None:
        bands = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    local_batch = batch
    if len(batch[0]) == MMDC_DATA_COMPONENTS:
        local_batch = mmdc2pvae_batch(batch)
    s2_r, s2_a = local_batch
    hw = PROSAIL_VAE.encoder.nb_enc_cropped_hw
    patched_s2_r = patchify(s2_r.squeeze(), patch_size=patch_size, margin=hw).to(
        PROSAIL_VAE.device
    )
    patched_s2_a = patchify(s2_a.squeeze(), patch_size=patch_size, margin=hw).to(
        PROSAIL_VAE.device
    )
    patched_sim_image = torch.zeros(
        (
            patched_s2_r.size(0),
            patched_s2_r.size(1),
            PROSAIL_VAE.encoder.output_size,
            patch_size,
            patch_size,
        )
    ).to(PROSAIL_VAE.device)
    patched_rec_image = torch.zeros(
        (patched_s2_r.size(0), patched_s2_r.size(1), len(bands), patch_size, patch_size)
    ).to(PROSAIL_VAE.device)
    patched_std_image = torch.zeros(
        (
            patched_s2_r.size(0),
            patched_s2_r.size(1),
            PROSAIL_VAE.encoder.output_size,
            patch_size,
            patch_size,
        )
    ).to(PROSAIL_VAE.device)
    for i in range(patched_s2_r.size(0)):
        for j in range(patched_s2_r.size(1)):
            x = patched_s2_r[i, j, ...].unsqueeze(0)
            angles = patched_s2_a[i, j, ...].unsqueeze(0)
            with torch.no_grad():
                if no_rec:
                    dist_params, z, sim = PROSAIL_VAE.point_estimate_sim(
                        x, angles, mode=mode
                    )
                else:
                    dist_params, z, sim, rec = PROSAIL_VAE.point_estimate_rec(
                        x, angles, mode=mode, apply_norm=False
                    )
                    patched_rec_image[i, j, :, :, :] = rec

            patched_sim_image[i, j, :, :, :] = sim.squeeze(0)
            std_sim = PROSAIL_VAE.sim_space.get_distribution_from_lat_params(
                dist_params
            ).variance.sqrt()
            patched_std_image[i, j, :, :, :] = std_sim  # dist_params.squeeze(0)[1,...]
    sim_image = unpatchify(patched_sim_image)[:, : s2_r.size(2), : s2_r.size(3)]
    rec_image = unpatchify(patched_rec_image)[:, : s2_r.size(2), : s2_r.size(3)]
    std_image = unpatchify(patched_std_image)[:, : s2_r.size(2), : s2_r.size(3)]

    if not padding:
        # doesn't remove invalid pixels on image borders that were lost to convolutions
        sim_image = crop_s2_input(sim_image, hw)
        rec_image = crop_s2_input(rec_image, hw)
        s2_a = crop_s2_input(s2_a, hw)
        s2_r = crop_s2_input(s2_r, hw)
        std_image = crop_s2_input(std_image, hw)
    return rec_image, sim_image, s2_r, s2_a, std_image


def check_is_patch(tensor: torch.Tensor):
    """
    Checks if a tensor (reflectances or angles) is a patch or a pixellic batch
    """
    if len(tensor.size()) == 4:  # (B x S) x F x H x W
        return True
    if len(tensor.size()) == 5:  # B x S x F x H x W
        return True
    elif len(tensor.size()) == 2:  # B x F
        return False
    else:
        raise ValueError


def tensor_to_raster(
    tensor: torch.Tensor,
    file_path: str,
    crs: str,
    resolution: float,
    dtype=np.float32,
    bounds: rio.coords.BoundingBox | None = None,
    xcoords: np.ndarray | None = None,
    ycoords: np.ndarray | None = None,
    nodata: int = -10000,
    hw: int = 0,
    half_res_coords: bool = True,
):
    assert len(tensor.size()) == 3
    if bounds is None:
        assert not (xcoords is None or ycoords is None)
        bounds = [xcoords[0], ycoords[-1], xcoords[-1], ycoords[0]]
    assert len(xcoords) == tensor.size(2) - 2 * hw
    assert len(ycoords) == tensor.size(1) - 2 * hw
    half_res = 0
    if half_res_coords:
        half_res = 0.5
    geotransform = (
        bounds[0] + (hw - half_res) * resolution,
        resolution,
        0.0,
        bounds[3] - (hw - half_res) * resolution,
        0.0,
        -resolution,
    )
    transform = Affine.from_gdal(*geotransform)
    count = tensor.size(0)
    profile = {
        "driver": "GTiff",
        "count": count,
        "height": tensor.size(1),
        "width": tensor.size(2),
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "lzw",
        "predictor": 2,
    }

    with rio.open(file_path, "w", **profile) as ds:
        ds.write(tensor.detach().cpu().numpy())
