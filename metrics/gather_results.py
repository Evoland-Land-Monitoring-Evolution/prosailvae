import os
import argparse
import socket
import pandas as pd
import prosailvae
import torch
from prosailvae.ProsailSimus import PROSAILVARS, ProsailVarsDist, BANDS
if __name__ == "__main__":
    from prosail_plots import plot_metric_boxplot
else:
    from metrics.prosail_plots import plot_metric_boxplot

def get_prosailvae_results_gather_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')

    parser.add_argument("-r", dest="root_results_dir",
                        help="path to root results direcotry",
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
    with open(root_res_dir + "/results_directory_names.txt") as f:
        res_dirs =  [line.rstrip() for line in f]
    return root_res_dir, res_dirs

def main():
    root_res_dir, res_dirs = get_results_dirs_names()
    gathered_res_dir = root_res_dir + "/aggregated_results/"
    if not os.path.isdir(gathered_res_dir):
        os.makedirs(gathered_res_dir)
    val_losses = []
    ae_percentiles = torch.zeros((len(res_dirs, len(PROSAILVARS) ,5)))
    are_percentiles = torch.zeros((len(res_dirs, len(PROSAILVARS) ,5)))
    for i, dir_name in enumerate(res_dirs):
        val_loss = pd.read_csv(root_res_dir+ "/" + dir_name + "/loss/valid_loss.csv")["loss_sum"].min()
        val_losses.append(val_loss)
        ae_percentiles[i,:,:] = torch.load(dir_name + '/metrics/ae_percentiles.pt')
        plot_metric_boxplot(ae_percentiles, gathered_res_dir, "agregated_ae")
        are_percentiles[i,:,:] = torch.load(dir_name + '/metrics/are_percentiles.pt')
        plot_metric_boxplot(ae_percentiles, gathered_res_dir, "agregated_are")
        pass
    pass

if __name__ == "__main__":
    main()