import pandas as pd 
import numpy as np
import os
from tabulate import tabulate

def get_df_results_from_array(array, model_names, metrics):

    return pd.DataFrame(data=array.squeeze(), columns=metrics)

def main():
    for metric in ["rmse", "picp"]:
        print(metric)
        data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/comparison_global/dev"
        filename = f"sim_tg_mean_simple_interpolate_lai_Land_cover_{metric}.npy"
        results = np.load(os.path.join(data_dir, filename))
        
        model_names = ["pvae", "pvae spatial", "SNAP"]
        if metric=="picp":
            model_names = model_names[:-1]
        metrics = ["Spain", "England", "Belgium", "All"]
        df_results = get_df_results_from_array(results, model_names, metrics)
        print(tabulate(df_results, tablefmt="latex_raw", headers=metrics, floatfmt=".2f", showindex=model_names))

    pass

if __name__ == "__main__":
    main()