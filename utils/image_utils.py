import torch

def unbatchify(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reshapes tensor into a patch-like tensor.
    Assumes that the patch spatial dimensions were put into the batch dimension.
    """
    n_tensor_dim = len(tensor.size())
    if n_tensor_dim == 3:
        patch_size = torch.sqrt(torch.tensor(tensor.size(0))).int().item()
        n_feat = tensor.size(1)
        n_samples = tensor.size(2)
        return tensor.reshape(patch_size, patch_size, n_feat, n_samples).permute(3,2,0,1)

    raise NotImplementedError

def crop_s2_input(s2_input:torch.Tensor, hw_crop:int=0) -> torch.Tensor:
    """
    Crops pixels on the border of an image tensor.
    Assumes at least a 3D Tensor, whose spatial dimensions are the last two
    """
    return s2_input[..., hw_crop:-hw_crop, hw_crop:-hw_crop]


def batchify_batch_latent(tensor:torch.Tensor):
    """
    Puts a patch tensor into a batched form.
    """
    # Input dim (B x 2L x H x W)
    tensor = tensor.permute(0, 2, 3, 1)
    return tensor.reshape(-1, tensor.size(3)) # Output dim (BxHxW) x 2L

def get_invalid_symetrical_padding(enc_kernel_sizes):
    """
    Computes how much border pixels are lost due to convolutions
    """
    hw = 0
    for kernel_size in enumerate(enc_kernel_sizes):
        hw += kernel_size//2
    return hw