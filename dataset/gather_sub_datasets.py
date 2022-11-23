#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:45:29 2022

@author: yoel
"""
import torch
import os
import argparse

def get_dataset_files(data_dir, filename=""):
    list_files = []
    for file in os.listdir(data_dir):
        if file.endswith(filename+".pt"):
            list_files.append(file)
    return list_files

def gather_dataset(data_dir):

    list_files_refl = get_dataset_files(data_dir, filename="prosail_s2_sim_refl")
    prosail_vars = torch.tensor([])
    prosail_refl = torch.tensor([])
    list_dataset_nb = get_sub_dataset_numbers(list_files_refl, 
                                              filename="prosail_s2_sim_refl")
    for i in list_dataset_nb:
        vars_i = torch.load(data_dir + f"/{i}_prosail_sim_vars.pt" )
        prosail_vars = torch.cat((prosail_vars, vars_i), axis=0)
        refl_i = torch.load(data_dir + f"/{i}_prosail_s2_sim_refl.pt")
        prosail_refl = torch.cat((prosail_refl, refl_i), axis=0)
    
    return prosail_refl, prosail_vars

def get_sub_dataset_numbers(list_files, filename="prosail_s2_sim_refl"):
    list_dataset_nb = []
    for file in list_files:
        list_dataset_nb.append(int(file.replace('_'+filename+'.pt','')))
    return list_dataset_nb

def get_data_gathering_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')

    parser.add_argument("-d", dest="data_dir",
                        help="path to data direcotry",
                        type=str, default="")

    return parser

def get_refl_normalization(prosail_refl):
    return prosail_refl.mean(0), prosail_refl.std(0)

if __name__ == "__main__":
    parser = get_data_gathering_parser().parse_args()
    prosail_refl, prosail_vars = gather_dataset(parser.data_dir)
    norm_mean, norm_std = get_refl_normalization(prosail_refl)
    torch.save(prosail_vars, parser.data_dir + "/full_" + "prosail_sim_vars.pt") 
    torch.save(prosail_refl, parser.data_dir + "/full_" + "prosail_s2_sim_refl.pt") 
    torch.save(norm_mean, parser.data_dir + "/full_" + "norm_mean.pt") 
    torch.save(norm_std, parser.data_dir + "/full_" + "norm_std.pt") 