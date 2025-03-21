import os

import numpy as np

from prosailvae.utils import save_dict

config = {
    "batch_size": 2048,
    "epochs": 50,
    "lr": 0.001,
    "hidden_layers_size": [256],
    "test_size": 0.01,
    "valid_ratio": 0.01,
    "workers": 0,
    "encoder_last_activation": None,
    "n_samples": 1,
    "supervised": True,
    "beta_kl": 0,
    "beta_index": 0,
    "dataset_file_prefix": "weiss_",
    "simulated_dataset": True,
    "lr_recompute": 51,
    "loss_type": "diag_nll",
    "apply_norm_rec": False,
    "exp_lr_decay": 0.98,
    "supervised_kl": False,
    "encoder_type": "rnn",
    "rnn_depth": 2,
    "rnn_number": 5,
}
save_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/config"
config_dir = "/pvae_configs/"
if not os.path.isdir(save_dir + config_dir):
    os.makedirs(save_dir + config_dir)
hidden_layers_width = np.array([2**k for k in range(9, 12)])
rnn_depth = np.array([2, 3, 4, 5, 6])
rnn_number = np.array([3, 4, 5, 6, 9, 12, 24])
print(len(hidden_layers_width) * len(rnn_depth) * len(rnn_number))
# hidden_layers_width = np.array([2**k for k in range(5,12)])
# rnn_depth = np.arange(1,8)
# rnn_number = np.arange(1,12)
# hidden_layers_width = np.array([2**k for k in range(5,7)])
# rnn_depth = np.arange(1,2)
# rnn_number = np.arange(1,2)
# print(len(hidden_layers_width) * len(rnn_depth) * len(rnn_number))
list_name_rel_path = []
configs_array = np.zeros((len(hidden_layers_width), len(rnn_depth), len(rnn_number), 4))
configs_idx = np.zeros((len(hidden_layers_width), len(rnn_depth), len(rnn_number)))
config_number = 0
for i in range(len(hidden_layers_width)):
    for j in range(len(rnn_depth)):
        for k in range(len(rnn_number)):
            config["hidden_layers_size"] = [hidden_layers_width[i]]
            config["rnn_depth"] = rnn_depth[j]
            config["rnn_number"] = rnn_number[k]
            configs_array[i, j, k, 1] = hidden_layers_width[i]
            configs_array[i, j, k, 2] = rnn_depth[j]
            configs_array[i, j, k, 3] = rnn_number[k]
            config_name = (
                config_dir
                + f"config_L_{hidden_layers_width[i]}_D_{rnn_depth[j]}_N_{rnn_number[k]}.json"
            )
            list_name_rel_path.append(config_name)
            save_dict(config, save_dir + config_name)
            configs_array[i, j, k, 0] = config_number
            config_number += 1
            if not os.path.isfile(save_dir + "/list_configs_rnn.txt"):
                with open(save_dir + "/list_configs_rnn.txt", "w") as outfile:
                    outfile.write(f"{config_name}\n")
            else:
                with open(save_dir + "/list_configs_rnn.txt", "a") as outfile:
                    outfile.write(f"{config_name}\n")
np.save(file=save_dir + "/configs_var_params.npy", arr=configs_array)
