import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import trange

from ..dataset.bvnet_dataset import load_bvnet_dataset

SNAP_WEIGHTS_PATH = Path(__file__).parent / "weights"


def normalize(
    unnormalized: torch.Tensor, min_sample: torch.Tensor, max_sample: torch.Tensor
):
    """
    Normalize with sample min and max of distribution
    """
    return 2 * (unnormalized - min_sample) / (max_sample - min_sample) - 1


def denormalize(
    normalized: torch.Tensor, min_sample: torch.Tensor, max_sample: torch.Tensor
):
    """
    de-normalize with sample min and max of distribution
    """
    return 0.5 * (normalized + 1) * (max_sample - min_sample) + min_sample


@dataclass
class NormBVNETNNV2:
    """
    Min and max snap nn input and output for normalization
    """

    input_min: torch.Tensor = torch.tensor(
        [
            0.0000,
            0.0000,
            0.0000,
            0.00663797254225,
            0.0139727270189,
            0.0266901380821,
            0.0163880741923,
            0.0000,
            0.918595400582,
            0.342022871159,
            -1.0000,
        ]
    )

    input_max: torch.Tensor = torch.tensor(
        [
            0.253061520472,
            0.290393577911,
            0.305398915249,
            0.608900395798,
            0.753827384323,
            0.782011770669,
            0.493761397883,
            0.49302598446,
            1.0000,
            0.936206429175,
            1.0000,
        ]
    )


class NormBVNETNNV3B:
    """
    Min and max snap nn input and output for normalization
    """

    input_min: torch.Tensor = torch.tensor(
        [
            0.0000,
            0.0000,
            0.0000,
            0.0119814116908,
            0.0169060342706,
            0.0176448354545,
            0.0147283842139,
            0.0000,
            0.979624800125,
            0.342108564072,
            -1.0000,
        ]
    )

    input_max: torch.Tensor = torch.tensor(
        [
            0.247742161604,
            0.305951681647,
            0.327098829671,
            0.599329840352,
            0.741682769861,
            0.780987637826,
            0.507673379171,
            0.502205128583,
            1.0000,
            0.927484749175,
            1.0000,
        ]
    )


class NormBVNETNNV3A:
    """
    Min and max snap nn input and output for normalization
    """

    input_min: torch.Tensor = torch.tensor(
        [
            0.0000,
            0.0000,
            0.0000,
            0.008717364330310326,
            0.019693160430621366,
            0.026217828282102625,
            0.018931934894415213,
            0.0000,
            0.979624800125421,
            0.342108564072183,
            -1.0000,
        ]
    )

    input_max: torch.Tensor = torch.tensor(
        [
            0.23901527463861838,
            0.29172736471507876,
            0.32652671459255694,
            0.5938903910368211,
            0.7466909927207045,
            0.7582393779705984,
            0.4929337190581187,
            0.4877499217101771,
            1.0000,
            0.9274847491748729,
            1.0000,
        ]
    )


@dataclass
class DenormSNAPLAIV2:
    lai_min: torch.Tensor = torch.tensor(0.000319182538301)
    lai_max: torch.Tensor = torch.tensor(14.4675094548)


@dataclass
class DenormSNAPLAIV3B:
    lai_min: torch.Tensor = torch.tensor(0.000233773908827)
    lai_max: torch.Tensor = torch.tensor(13.834592547)


@dataclass
class DenormSNAPLAIV3A:
    lai_min: torch.Tensor = torch.tensor(0.00023377390882650673)
    lai_max: torch.Tensor = torch.tensor(13.834592547008839)


@dataclass
class DenormSNAPCCCV2:
    ccc_min: torch.Tensor = torch.tensor(0.00742669295987)
    ccc_max: torch.Tensor = torch.tensor(873.90822211)


@dataclass
class DenormSNAPCCCV3B:
    ccc_min: torch.Tensor = torch.tensor(0.0184770096032)
    ccc_max: torch.Tensor = torch.tensor(888.156665152)


@dataclass
class DenormSNAPCCCV3A:
    ccc_min: torch.Tensor = torch.tensor(0.01847700960324858)
    ccc_max: torch.Tensor = torch.tensor(888.1566651521919)


@dataclass
class DenormSNAPCWCV2:
    cwc_min: torch.Tensor = torch.tensor(3.85066859366e-06)
    cwc_max: torch.Tensor = torch.tensor(0.522417054645)


@dataclass
class DenormSNAPCWCV3B:
    cwc_min: torch.Tensor = torch.tensor(2.84352788861e-06)
    cwc_max: torch.Tensor = torch.tensor(0.419181347199)


@dataclass
class DenormSNAPCWCV3A:
    cwc_min: torch.Tensor = torch.tensor(4.227082600108468e-06)
    cwc_max: torch.Tensor = torch.tensor(0.5229998511245837)


def get_SNAP_norm_factors(ver: str = "2", variable="lai"):
    """
    Get normalization factor for BVNET NN
    """
    if ver == "2":
        bvnet_norm = NormBVNETNNV2()
        if variable == "lai":
            variable_min = DenormSNAPLAIV2().lai_min
            variable_max = DenormSNAPLAIV2().lai_max
        elif variable == "ccc":
            variable_min = DenormSNAPCCCV2().ccc_min
            variable_max = DenormSNAPCCCV2().ccc_max
        elif variable == "cab":
            variable_min = DenormSNAPCCCV2().ccc_min / (
                DenormSNAPLAIV2().lai_max - DenormSNAPLAIV2().lai_min
            )
            variable_max = DenormSNAPCCCV2().ccc_max / (
                DenormSNAPLAIV2().lai_max - DenormSNAPLAIV2().lai_min
            )
        elif variable == "cwc":
            variable_min = DenormSNAPCWCV2().cwc_min
            variable_max = DenormSNAPCWCV2().cwc_max
        elif variable == "cw":
            variable_min = DenormSNAPCWCV2().cwc_min / (
                DenormSNAPLAIV2().lai_max - DenormSNAPLAIV2().lai_min
            )
            variable_max = DenormSNAPCWCV2().cwc_max / (
                DenormSNAPLAIV2().lai_max - DenormSNAPLAIV2().lai_min
            )
        else:
            raise NotImplementedError
    elif ver == "3B":
        bvnet_norm = NormBVNETNNV3B()
        if variable == "lai":
            variable_min = DenormSNAPLAIV3B().lai_min
            variable_max = DenormSNAPLAIV3B().lai_max
        elif variable == "ccc":
            variable_min = DenormSNAPCCCV3B().ccc_min
            variable_max = DenormSNAPCCCV3B().ccc_max
        elif variable == "cab":
            variable_min = DenormSNAPCCCV3B().ccc_min / (
                DenormSNAPLAIV3B().lai_max - DenormSNAPLAIV3B().lai_min
            )
            variable_max = DenormSNAPCCCV3B().ccc_max / (
                DenormSNAPLAIV3B().lai_max - DenormSNAPLAIV3B().lai_min
            )
        elif variable == "cwc":
            variable_min = DenormSNAPCWCV3B().cwc_min
            variable_max = DenormSNAPCWCV3B().cwc_max
        elif variable == "cw":
            variable_min = DenormSNAPCWCV3B().cwc_min / (
                DenormSNAPLAIV3B().lai_max - DenormSNAPLAIV3B().lai_min
            )
            variable_max = DenormSNAPCWCV3B().cwc_max / (
                DenormSNAPLAIV3B().lai_max - DenormSNAPLAIV3B().lai_min
            )
        else:
            raise NotImplementedError
    elif ver == "3A":
        bvnet_norm = NormBVNETNNV3A()
        if variable == "lai":
            variable_min = DenormSNAPLAIV3A().lai_min
            variable_max = DenormSNAPLAIV3A().lai_max
        elif variable == "ccc":
            variable_min = DenormSNAPCCCV3A().ccc_min
            variable_max = DenormSNAPCCCV3A().ccc_max
        elif variable == "cab":
            variable_min = DenormSNAPCCCV3A().ccc_min / (
                DenormSNAPLAIV3A().lai_max - DenormSNAPLAIV3A().lai_min
            )
            variable_max = DenormSNAPCCCV3A().ccc_max / (
                DenormSNAPLAIV3A().lai_max - DenormSNAPLAIV3A().lai_min
            )
        elif variable == "cwc":
            variable_min = DenormSNAPCWCV3A().cwc_min
            variable_max = DenormSNAPCWCV3A().cwc_max
        elif variable == "cw":
            variable_min = DenormSNAPCWCV3A().cwc_min / (
                DenormSNAPLAIV3A().lai_max - DenormSNAPLAIV3A().lai_min
            )
            variable_max = DenormSNAPCWCV3A().cwc_max / (
                DenormSNAPLAIV3A().lai_max - DenormSNAPLAIV3A().lai_min
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return bvnet_norm.input_min, bvnet_norm.input_max, variable_min, variable_max


class BVNET(nn.Module):
    """Neural Network with BVNET architecture to predict LAI from S2
    reflectances and angles

    """

    def __init__(
        self, device: str = "cpu", ver: str = "3A", variable="lai", third_layer=False
    ):
        super().__init__()
        assert ver in ["2", "3A", "3B"]
        input_min, input_max, variable_min, variable_max = get_SNAP_norm_factors(
            ver=ver, variable=variable
        )
        input_size = len(input_max)  # 8 bands + 3 angles
        hidden_layer_size = 5
        if not third_layer:
            layers = OrderedDict(
                [
                    (
                        "layer_1",
                        nn.Linear(
                            in_features=input_size, out_features=hidden_layer_size
                        ).float(),
                    ),
                    ("tanh", nn.Tanh()),
                    (
                        "layer_2",
                        nn.Linear(
                            in_features=hidden_layer_size, out_features=1
                        ).float(),
                    ),
                ]
            )
        else:
            layers = OrderedDict(
                [
                    (
                        "layer_0",
                        nn.Linear(
                            in_features=input_size, out_features=input_size
                        ).float(),
                    ),
                    ("tanh", nn.Tanh()),
                    (
                        "layer_1",
                        nn.Linear(
                            in_features=input_size, out_features=hidden_layer_size
                        ).float(),
                    ),
                    ("tanh", nn.Tanh()),
                    (
                        "layer_2",
                        nn.Linear(
                            in_features=hidden_layer_size, out_features=1
                        ).float(),
                    ),
                ]
            )
        self.input_min = input_min.float().to(device)
        self.input_max = input_max.float().to(device)
        self.variable_min = variable_min.float().to(device)
        self.variable_max = variable_max.float().to(device)
        self.net: nn.Sequential = nn.Sequential(layers).to(device)
        self.device = device
        self.ver = ver
        self.variable = variable

    def save_weights(self, dir, prefix=""):
        weights_1 = self.net.layer_1.weight.detach().cpu().numpy()
        weights_2 = self.net.layer_2.weight.detach().cpu().numpy()
        bias_1 = self.net.layer_1.bias.detach().cpu().numpy()
        bias_2 = self.net.layer_2.bias.detach().cpu().numpy()
        pd.DataFrame(weights_1).to_csv(
            os.path.join(dir, f"{prefix}{self.ver}_{self.variable}_weights_1.csv"),
            header=False,
            index=False,
            mode="w",
        )
        pd.DataFrame(weights_2).to_csv(
            os.path.join(dir, f"{prefix}{self.ver}_{self.variable}_weights_2.csv"),
            header=False,
            index=False,
            mode="w",
        )
        pd.DataFrame(bias_1).to_csv(
            os.path.join(dir, f"{prefix}{self.ver}_{self.variable}_bias_1.csv"),
            header=False,
            index=False,
            mode="w",
        )
        pd.DataFrame(bias_2).to_csv(
            os.path.join(dir, f"{prefix}{self.ver}_{self.variable}_bias_2.csv"),
            header=False,
            index=False,
            mode="w",
        )

    def load_weights(self, dir, prefix=""):
        weights_1 = torch.from_numpy(
            pd.read_csv(
                os.path.join(dir, f"{prefix}{self.ver}_{self.variable}_weights_1.csv"),
                header=None,
            ).values
        ).float()
        weights_2 = torch.from_numpy(
            pd.read_csv(
                os.path.join(dir, f"{prefix}{self.ver}_{self.variable}_weights_2.csv"),
                header=None,
            ).values
        ).float()
        bias_1 = torch.from_numpy(
            pd.read_csv(
                os.path.join(dir, f"{prefix}{self.ver}_{self.variable}_bias_1.csv"),
                header=None,
            ).values
        ).float()
        bias_2 = torch.from_numpy(
            pd.read_csv(
                os.path.join(dir, f"{prefix}{self.ver}_{self.variable}_bias_2.csv"),
                header=None,
            ).values
        ).float()
        self.net.layer_1.bias = nn.Parameter(bias_1.to(self.device).reshape(-1))
        self.net.layer_1.weight = nn.Parameter(weights_1.to(self.device))
        self.net.layer_2.bias = nn.Parameter(bias_2.to(self.device).reshape(-1))
        self.net.layer_2.weight = nn.Parameter(weights_2.to(self.device))

    def set_snap_weights(self):
        """
        Set Neural Network weights and biases to BVNET's original values
        """
        self.load_weights(SNAP_WEIGHTS_PATH, prefix="weiss_")

    def forward(self, s2_data: torch.Tensor, spatial_mode=False):
        """
        Forward method of BVNET NN to predict a biophysical variable
        """
        if spatial_mode:
            if len(s2_data.size()) == 3:
                (_, size_h, size_w) = s2_data.size()
                s2_data = s2_data.permute(1, 2, 0).reshape(size_h * size_w, -1)
            else:
                raise NotImplementedError
        s2_data_norm = normalize(s2_data, self.input_min, self.input_max)
        variable_norm = self.net.forward(s2_data_norm)
        variable = denormalize(variable_norm, self.variable_min, self.variable_max)
        if spatial_mode:
            variable = variable.reshape(size_h, size_w, 1).permute(2, 0, 1)
        return variable

    def train_model(
        self,
        train_loader,
        valid_loader,
        optimizer,
        epochs: int = 100,
        lr_scheduler=None,
        disable_tqdm: bool = False,
        cycle_training=True,
        lr_recompute=10,
        res_dir=".",
        loc_bv=0,
        scale_bv=1,
    ):
        """
        Fit and validate the model to data for a number of epochs
        """
        all_train_losses = []
        all_valid_losses = []
        all_lr = []
        lr_init = 1e-3
        best_valid_loss = np.inf
        for _ in trange(epochs, disable=disable_tqdm):
            train_loss = self.fit(
                train_loader, optimizer, loc_bv=loc_bv, scale_bv=scale_bv
            )
            all_train_losses.append(train_loss.item())
            valid_loss = self.validate(valid_loader, loc_bv=loc_bv, scale_bv=scale_bv)
            all_valid_losses.append(valid_loss.item())
            if valid_loss.item() < best_valid_loss:
                best_valid_loss = valid_loss.item()
                if res_dir is not None:
                    self.save_weights(res_dir)
            all_lr.append(optimizer.param_groups[0]["lr"])
            if lr_scheduler is not None:
                lr_scheduler.step(valid_loss)
            if all_lr[-1] <= 5e-8:
                if not cycle_training:
                    break  # stop training if lr too low
                for g in optimizer.param_groups:
                    g["lr"] = lr_init
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer,
                    patience=lr_recompute,
                    threshold=0.01,
                    threshold_mode="abs",
                )
        if res_dir is not None:
            self.load_weights(res_dir)
        return all_train_losses, all_valid_losses, all_lr

    def fit(self, loader, optimizer, loc_bv=0, scale_bv=1):
        """
        Apply mini-batch optimization from a train dataloader
        """
        self.train()
        loss_mean = torch.tensor(0.0).to(self.device)
        for _, batch in enumerate(loader):
            loss = self.get_batch_loss(batch, loc_bv=loc_bv, scale_bv=scale_bv)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                loss_mean += loss / batch[0].size(0)
        return loss_mean

    def validate(self, loader, loc_bv=0, scale_bv=1):
        """
        Compute loss on loader with mini-batches
        """
        self.eval()
        with torch.no_grad():
            loss_mean = torch.tensor(0.0).to(self.device)
            for _, batch in enumerate(loader):
                loss = self.get_batch_loss(batch, loc_bv=loc_bv, scale_bv=scale_bv)
                loss_mean += loss / batch[0].size(0)
        return loss_mean

    def get_batch_loss(self, batch, loc_bv=0, scale_bv=1):
        """
        Computes loss on batch
        """
        s2_data, variable = batch
        variable_pred = self.forward(s2_data.to(self.device))
        return (
            (
                (variable_pred - loc_bv) / scale_bv
                - ((variable.to(self.device) - loc_bv) / scale_bv)
            )
            .pow(2)
            .mean()
        )


def test_snap_nn(ver="2"):
    """
    Test if BVNET neural network's outputs are identical to that of the translated java
    code of BVNET
    """
    from weiss_lai_sentinel_hub import (
        get_layer_1_neuron_biases,
        get_layer_1_neuron_weights,
        get_layer_2_bias,
        get_layer_2_weights,
        get_norm_factors,
        layer2,
        neuron,
    )

    import prosailvae

    s2_r, prosail_vars = load_bvnet_dataset(
        os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/lai/",
        mode="snap",
    )
    s2_a = prosail_vars[:, -3:]
    lai = prosail_vars[:, 6].reshape(-1, 1)
    bvnet = BVNET(ver=ver, variable="lai")
    bvnet.set_snap_weights()
    sample = torch.cat(
        (torch.from_numpy(s2_r), torch.cos(torch.from_numpy(s2_a))), 1
    ).float()
    ver = ver
    norm_factors = get_norm_factors(ver=ver)
    w1, w2, w3, w4, w5 = get_layer_1_neuron_weights(ver=ver)
    b1, b2, b3, b4, b5 = get_layer_1_neuron_biases(ver=ver)
    wl2 = get_layer_2_weights(ver=ver)
    bl2 = get_layer_2_bias(ver=ver)

    b03_norm = normalize(
        sample[:, 0].unsqueeze(1),
        norm_factors["min_sample_B03"],
        norm_factors["max_sample_B03"],
    )
    b04_norm = normalize(
        sample[:, 1].unsqueeze(1),
        norm_factors["min_sample_B04"],
        norm_factors["max_sample_B04"],
    )
    b05_norm = normalize(
        sample[:, 2].unsqueeze(1),
        norm_factors["min_sample_B05"],
        norm_factors["max_sample_B05"],
    )
    b06_norm = normalize(
        sample[:, 3].unsqueeze(1),
        norm_factors["min_sample_B06"],
        norm_factors["max_sample_B06"],
    )
    b07_norm = normalize(
        sample[:, 4].unsqueeze(1),
        norm_factors["min_sample_B07"],
        norm_factors["max_sample_B07"],
    )
    b8a_norm = normalize(
        sample[:, 5].unsqueeze(1),
        norm_factors["min_sample_B8A"],
        norm_factors["max_sample_B8A"],
    )
    b11_norm = normalize(
        sample[:, 6].unsqueeze(1),
        norm_factors["min_sample_B11"],
        norm_factors["max_sample_B11"],
    )
    b12_norm = normalize(
        sample[:, 7].unsqueeze(1),
        norm_factors["min_sample_B12"],
        norm_factors["max_sample_B12"],
    )
    viewZen_norm = normalize(
        sample[:, 8].unsqueeze(1),
        norm_factors["min_sample_viewZen"],
        norm_factors["max_sample_viewZen"],
    )
    sunZen_norm = normalize(
        sample[:, 9].unsqueeze(1),
        norm_factors["min_sample_sunZen"],
        norm_factors["max_sample_sunZen"],
    )
    relAzim_norm = sample[:, 10].unsqueeze(1)
    band_dim = 1
    with torch.no_grad():
        x_norm = normalize(sample, bvnet.input_min, bvnet.input_max)
        snap_input = torch.cat(
            (
                b03_norm,
                b04_norm,
                b05_norm,
                b06_norm,
                b07_norm,
                b8a_norm,
                b11_norm,
                b12_norm,
                viewZen_norm,
                sunZen_norm,
                relAzim_norm,
            ),
            axis=band_dim,
        )
        assert torch.isclose(snap_input, x_norm, atol=1e-5, rtol=1e-5).all()
        nb_dim = len(b03_norm.size())
        neuron1 = neuron(snap_input, w1, b1, nb_dim, sum_dim=band_dim)
        neuron2 = neuron(snap_input, w2, b2, nb_dim, sum_dim=band_dim)
        neuron3 = neuron(snap_input, w3, b3, nb_dim, sum_dim=band_dim)
        neuron4 = neuron(snap_input, w4, b4, nb_dim, sum_dim=band_dim)
        neuron5 = neuron(snap_input, w5, b5, nb_dim, sum_dim=band_dim)
        linear_1_snap = nn.Linear(11, 5)
        linear_1_snap.weight = bvnet.net.layer_1.weight
        linear_1_snap.bias = bvnet.net.layer_1.bias
        assert torch.isclose(
            linear_1_snap(x_norm),
            bvnet.net.layer_1.bias + x_norm @ bvnet.net.layer_1.weight.transpose(1, 0),
            atol=1e-4,
        ).all()

        n_snap_nn = torch.tanh(
            bvnet.net.layer_1.bias + x_norm @ bvnet.net.layer_1.weight.transpose(1, 0)
        )
        assert torch.isclose(
            n_snap_nn,
            torch.cat((neuron1, neuron2, neuron3, neuron4, neuron5), axis=1),
            atol=1e-4,
        ).all()

        linear_2_snap = bvnet.net.layer_2
        # linear_2_snap.weight = snap_nn.net[2].weight
        # linear_2_snap.bias = snap_nn.net[2].bias
        assert torch.isclose(
            linear_2_snap(n_snap_nn),
            bvnet.net.layer_2.bias
            + n_snap_nn @ bvnet.net.layer_2.weight.transpose(1, 0),
            atol=1e-4,
        ).all()
        layer_2_output = layer2(
            neuron1, neuron2, neuron3, neuron4, neuron5, wl2, bl2, sum_dim=band_dim
        )
        l_snap_nn = (
            bvnet.net.layer_2.bias
            + n_snap_nn @ bvnet.net.layer_2.weight.transpose(1, 0)
        )
        lai_prenorm_snap = bvnet.net.forward(x_norm)
        assert torch.isclose(
            l_snap_nn.squeeze(), layer_2_output.squeeze(), atol=1e-4
        ).all()
        assert torch.isclose(
            lai_prenorm_snap.squeeze(), layer_2_output.squeeze(), atol=1e-4
        ).all()
        lai = denormalize(
            layer_2_output,
            norm_factors["min_sample_lai"],
            norm_factors["max_sample_lai"],
        )
        snap_lai = denormalize(l_snap_nn, bvnet.variable_min, bvnet.variable_max)
        assert torch.isclose(snap_lai.squeeze(), lai.squeeze(), atol=1e-4).all()
        assert torch.isclose(
            bvnet.forward(sample).squeeze(), lai.squeeze(), atol=1e-4
        ).all()


def main():
    dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/bvnet_regression/weights"
    if not os.path.isdir(dir):
        os.makedirs(dir)
    for ver in ["2", "3A", "3B"]:
        for variable in ["lai", "cab", "cw"]:
            print(ver, variable)
            model = BVNET(ver=ver, variable=variable)
            model.set_snap_weights()
    pass


if __name__ == "__main__":
    main()
