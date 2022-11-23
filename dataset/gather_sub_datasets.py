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
            list_files.append(os.path.join(data_dir, file))
    return list_files

def gather_dataset(data_dir):

    list_files_refl = get_dataset_files(data_dir, filename="prosail_s2_sim_refl")
    list_files_vars = get_dataset_files(data_dir, filename="prosail_sim_vars")
    prosail_vars = torch.tensor([[]])
    prosail_refl = torch.tensor([[]])
    
    for i in range(len(list_files_refl)):
        vars_i = torch.load(data_dir + f"/{i}_" + list_files_refl[i])
        prosail_vars = torch.concat((prosail_vars, vars_i), axis=0)
        refl_i = torch.load(data_dir + f"/{i}_" + list_files_vars[i])
        prosail_refl = torch.concat((prosail_refl, refl_i), axis=0)
    
    return prosail_refl, prosail_vars

def get_data_gathering_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')

    parser.add_argument("-d", dest="data_dir",
                        help="path to data direcotry",
                        type=str, default="")

    return parser

if __name__ == "__main__":
    parser = get_data_gathering_parser().parse_args()
    prosail_refl, prosail_vars = gather_dataset(parser.data_dir)
    torch.save(prosail_vars, parser.data_dir + "/full_" + "prosail_sim_vars.pt") 
    torch.save(prosail_refl, parser.data_dir + "/full_" + "prosail_s2_sim_refl.pt") 