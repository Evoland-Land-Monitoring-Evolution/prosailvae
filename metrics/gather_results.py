import os
import argparse
import socket
import pandas as pd
import prosailvae

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
    val_losses = []
    for i, dir_name in enumerate(res_dirs):
        val_loss = pd.read_csv(root_res_dir+ "/" +dir_name + "/loss/valid_loss.csv")["loss_sum"].min()
        val_losses.append(val_loss)
        pass
    pass

if __name__ == "__main__":
    main()