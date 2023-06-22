import pandas as pd 
import numpy as np
import os
from tabulate import tabulate

def get_df_results_from_array(array, model_names, metrics):
    pass

def main():
    data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/comparison_global"
    filename = ".npy"
    results = np.read(os.path.join(data_dir, filename))
    model_names = []
    metrics = []
    get_df_results_from_array(results, model_names, metrics)
    print(tabulate(df_mae, tablefmt="latex_raw", headers=metrics, floatfmt=".2f", showindex=model_names))
    pass

if __name__ == "__main__":
    main()