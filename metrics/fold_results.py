import os
import argparse
import socket
import pandas as pd
import prosailvae
import torch
from prosailvae.ProsailSimus import PROSAILVARS

def get_prosailvae_results_gather_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')

    parser.add_argument("-r", dest="root_results_dir",
                        help="path to fold root results directory",
                        type=str, default="")
    return parser

def get_results_dirs_names():
    if socket.gethostname()=='CELL200973':
        args=[ "-r", ""]
        parser = get_prosailvae_results_gather_parser().parse_args(args)    
    else:
        parser = get_prosailvae_results_gather_parser().parse_args()
    if len(parser.root_results_dir)==0:
        root_res_dir = os.path.join(os.path.join(os.path.dirname(prosailvae.__file__),
                                                     os.pardir),"results/37099873_jobarray/")
    else:
        root_res_dir =  os.path.join(parser.root_results_dir, os.pardir)
    with open(root_res_dir + "/results_directory_names.txt") as f:
        res_dirs =  [line.rstrip() for line in f]
    return root_res_dir, res_dirs

def plot_losses(val_losses, gathered_res_dir, model_names=None):
    if model_names is None or len(model_names)!=len(val_losses):
        model_names = [str(i+1) for i in range(len(val_losses))]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(dpi=150)
    ax.scatter([i+1 for i in range(len(val_losses))], val_losses, s=20, marker='o')
    ax.set_xticks([i+1 for i in range(len(val_losses))]) 
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Test Loss")
    ax.set_xlabel("Model")
    fig.savefig(gathered_res_dir + "/test_loss.svg")
    pass


def main():
    root_res_dir, res_dirs = get_results_dirs_names()
    gathered_res_dir = root_res_dir + "/fold_results/"
    if not os.path.isdir(gathered_res_dir):
        os.makedirs(gathered_res_dir)
    all_test_losses = torch.zeros((len(res_dirs),1))
    alpha_pi = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 
                0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    all_aer_percentiles = torch.zeros((len(res_dirs), 5, len(PROSAILVARS)))
    all_are_percentiles = torch.zeros((len(res_dirs), 5, len(PROSAILVARS)))
    all_mae = torch.zeros((len(res_dirs), len(PROSAILVARS)))
    all_maer = torch.zeros((len(res_dirs), len(PROSAILVARS)))
    all_mpiw = torch.zeros((len(res_dirs), len(alpha_pi), len(PROSAILVARS)))
    all_mpiwr = torch.zeros((len(res_dirs), len(alpha_pi), len(PROSAILVARS)))
    all_picp = torch.zeros((len(res_dirs), len(alpha_pi), len(PROSAILVARS)))
    all_lat_nll = torch.zeros((len(res_dirs), len(PROSAILVARS)))
    if os.path.isfile(root_res_dir + "/model_names.txt"):
        with open(root_res_dir + "/model_names.txt") as f:
            model_names =  [line.rstrip() for line in f]
    else:
        model_names = [str(i+1) for i in range(len(res_dirs))]
    for i, dir_name in enumerate(res_dirs):
        test_loss = pd.read_csv(dir_name + "/loss/test_loss.csv")["loss_sum"].values[0]
        all_test_losses[i,0] = test_loss
        all_picp[i,:,:] = torch.from_numpy(pd.read_csv(dir_name + "/metrics/picp.csv", index_col=[0]).drop(columns=["alpha"]).values)
        all_mpiw[i,:,:] = torch.from_numpy(pd.read_csv(dir_name + "/metrics/mpiw.csv", index_col=[0]).drop(columns=["alpha"]).values)
        all_mpiwr[i,:,:] = torch.from_numpy(pd.read_csv(dir_name + "/metrics/mpiwr.csv", index_col=[0]).values)
        all_mae[i,:] = torch.from_numpy(pd.read_csv(dir_name + "/metrics/mae.csv", index_col=[0]).values).squeeze()
        all_maer[i,:] = torch.from_numpy(pd.read_csv(dir_name + "/metrics/maer.csv", index_col=[0]).values).squeeze()
        all_lat_nll[i,:] = torch.load(dir_name + '/params_nll.pt', map_location=torch.device('cpu')).squeeze()
        all_aer_percentiles[i,:,:] = torch.load(dir_name + '/metrics/aer_percentiles.pt', map_location=torch.device('cpu'))
        all_are_percentiles[i,:,:] = torch.load(dir_name + '/metrics/are_percentiles.pt', map_location=torch.device('cpu'))     
        pass
    pd.DataFrame(data=all_test_losses.reshape(1,-1).cpu().numpy(), columns=model_names).to_csv(gathered_res_dir+'/all_losses.csv')
    torch.save(all_picp, gathered_res_dir+'/picp.pt')
    torch.save(all_mpiw, gathered_res_dir+'/mpiw.pt')
    torch.save(all_mpiwr, gathered_res_dir+'/mpiwr.pt')
    torch.save(all_mae, gathered_res_dir+'/mae.pt')
    torch.save(all_maer, gathered_res_dir+'/maer.pt')
    torch.save(all_maer, gathered_res_dir+'/maer.pt')
    torch.save(all_lat_nll, gathered_res_dir+'/lat_nll.pt')
    torch.save(all_aer_percentiles, gathered_res_dir+'/aer_percentiles.pt')
    torch.save(all_are_percentiles, gathered_res_dir+'/are_percentiles.pt')
    pass

if __name__ == "__main__":
    main()