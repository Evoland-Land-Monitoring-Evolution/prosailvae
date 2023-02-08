import os
import argparse
import socket
import pandas as pd
import prosailvae
import torch
from prosailvae.ProsailSimus import PROSAILVARS
import shutil
import numpy as np

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if (os.path.isdir(os.path.join(a_dir, name)) and name.isdigit())]

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
        root_res_dir =  parser.root_results_dir
    gathered_res_dir = root_res_dir + "/agregated_results/"
    if os.path.isdir(gathered_res_dir):
        shutil.rmtree(gathered_res_dir)
    res_dirs = get_immediate_subdirectories(root_res_dir)
    res_dirs.sort(key=int)

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
    n_folds = 5
    root_res_dir, res_dirs = get_results_dirs_names()
    gathered_res_dir = root_res_dir + "/agregated_results/"
    os.makedirs(gathered_res_dir)
    all_test_losses = torch.zeros((len(res_dirs), n_folds))
    alpha_pi = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 
                0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    all_aer_percentiles = torch.zeros((len(res_dirs), n_folds, 5, len(PROSAILVARS)))
    all_are_percentiles = torch.zeros((len(res_dirs), n_folds, 5, len(PROSAILVARS)))
    all_mae = torch.zeros((len(res_dirs), n_folds, len(PROSAILVARS)))
    all_maer = torch.zeros((len(res_dirs), n_folds, len(PROSAILVARS)))
    all_mpiw = torch.zeros((len(res_dirs), n_folds, len(alpha_pi), len(PROSAILVARS)))
    all_mpiwr = torch.zeros((len(res_dirs), n_folds, len(alpha_pi), len(PROSAILVARS)))
    all_picp = torch.zeros((len(res_dirs), n_folds, len(alpha_pi), len(PROSAILVARS)))
    all_lat_nll = torch.zeros((len(res_dirs), n_folds, len(PROSAILVARS)))
    if os.path.isfile(root_res_dir + "/model_names.txt"):
        with open(root_res_dir + "/model_names.txt") as f:
            model_names =  [line.rstrip() for line in f]
    else:
        model_names = [str(i+1) for i in range(len(res_dirs))]
    VALUE_ERROR = 100000
    for i, dir_name in enumerate(res_dirs):
        print(dir_name)
        if not os.path.isdir(root_res_dir + dir_name + "/fold_results/"):
            print(f"WARNING : no folds were available in experiment {dir_name}")
            test_loss = VALUE_ERROR * np.ones((n_folds))
            picp = VALUE_ERROR * torch.ones(n_folds, len(alpha_pi), len(PROSAILVARS))
            mpiw = VALUE_ERROR * torch.ones(n_folds, len(alpha_pi), len(PROSAILVARS))
            mpiwr = VALUE_ERROR * torch.ones(n_folds, len(alpha_pi), len(PROSAILVARS))
            mae = VALUE_ERROR * torch.ones(n_folds, len(PROSAILVARS))
            maer = VALUE_ERROR * torch.ones(n_folds, len(PROSAILVARS))
            lat_nll = VALUE_ERROR * torch.ones(n_folds, len(PROSAILVARS))
            aer_percentiles = VALUE_ERROR * torch.ones(n_folds, 5, len(PROSAILVARS))
            are_percentiles = VALUE_ERROR * torch.ones(n_folds, 5, len(PROSAILVARS))
        else:  
            test_loss = pd.read_csv(root_res_dir + dir_name + "/fold_results/all_losses.csv", index_col=[0]).values.reshape(-1)
            picp = torch.load(root_res_dir + dir_name + "/fold_results/picp.pt")
            mpiw = torch.load(root_res_dir + dir_name + "/fold_results/mpiw.pt")
            mpiwr = torch.load(root_res_dir + dir_name + "/fold_results/mpiwr.pt")
            mae = torch.load(root_res_dir + dir_name + "/fold_results/mae.pt")
            maer = torch.load(root_res_dir + dir_name + "/fold_results/maer.pt")
            lat_nll = torch.load(root_res_dir + dir_name + '/fold_results/lat_nll.pt')
            aer_percentiles = torch.load(root_res_dir + dir_name + '/fold_results/aer_percentiles.pt')
            are_percentiles = torch.load(root_res_dir + dir_name + '/fold_results/are_percentiles.pt') 

            if len(test_loss) < n_folds:
                print(f"WARNING : not all folds were available in experiment {dir_name}")
                diff_size = n_folds-len(test_loss)
                test_loss = np.concatenate([test_loss, VALUE_ERROR * np.ones((diff_size))])
                picp = torch.cat((picp, VALUE_ERROR * torch.ones(diff_size, len(alpha_pi), len(PROSAILVARS))), dim=0)
                mpiw = torch.cat((mpiw, VALUE_ERROR * torch.ones(diff_size, len(alpha_pi), len(PROSAILVARS))), dim=0)
                mpiwr = torch.cat((mpiwr, VALUE_ERROR * torch.ones(diff_size, len(alpha_pi), len(PROSAILVARS))), dim=0)
                mae = torch.cat((mae, VALUE_ERROR * torch.ones(diff_size, len(PROSAILVARS))), dim=0)
                maer = torch.cat((maer, VALUE_ERROR * torch.ones(diff_size, len(PROSAILVARS))), dim=0)
                lat_nll = torch.cat((lat_nll, VALUE_ERROR * torch.ones(diff_size, len(PROSAILVARS))), dim=0)
                aer_percentiles = torch.cat((aer_percentiles, VALUE_ERROR * torch.ones(diff_size, 5, len(PROSAILVARS))), dim=0)
                are_percentiles = torch.cat((are_percentiles, VALUE_ERROR * torch.ones(diff_size, 5, len(PROSAILVARS))), dim=0)

            all_test_losses[i,:] = torch.from_numpy(test_loss).reshape(1,-1)
            all_picp[i,:,:,:] = picp
            all_mpiw[i,:,:,:] = mpiw
            all_mpiwr[i,:,:,:] = mpiwr
            all_mae[i,:,:] = mae
            all_maer[i,:,:] = maer
            all_lat_nll[i,:,:] = lat_nll
            all_aer_percentiles[i,:,:,:] = aer_percentiles
            all_are_percentiles[i,:,:,:] = are_percentiles
        pass
    
    torch.save(all_picp, gathered_res_dir+'/picp.pt')
    torch.save(all_mpiw, gathered_res_dir+'/mpiw.pt')
    torch.save(all_mpiwr, gathered_res_dir+'/mpiwr.pt')
    torch.save(all_mae, gathered_res_dir+'/mae.pt')
    torch.save(all_maer, gathered_res_dir+'/maer.pt')
    torch.save(all_maer, gathered_res_dir+'/maer.pt')
    torch.save(all_lat_nll, gathered_res_dir+'/lat_nll.pt')
    torch.save(all_aer_percentiles, gathered_res_dir+'/aer_percentiles.pt')
    torch.save(all_are_percentiles, gathered_res_dir+'/are_percentiles.pt')
    pd.DataFrame(data=all_test_losses.cpu().squeeze().numpy()).to_csv(gathered_res_dir+'/all_losses.csv')
    pass

if __name__ == "__main__":
    main()