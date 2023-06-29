import matplotlib.pyplot as plt
import torch
import numpy as np
from sensorsio.utils import rgb_render
from dataset.frm4veg_validation import load_frm4veg_data
import seaborn as sns
import pandas as pd
import tikzplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    'mathtext.fontset' : 'custom',
    'mathtext.rm': 'Bitstream Vera Sans',
    'mathtext.it': 'Bitstream Vera Sans:italic',
    'mathtext.bf': 'Bitstream Vera Sans:bold'
})

def plot_frm4veg_validation_patch(gdf, pred_at_patch: np.ndarray,
                                  #pred_at_site: np.ndarray,
                                  variable:str="lai", 
                                  legend_columns=2):
    
    df_sns_plot = pd.DataFrame({variable: gdf[variable].values.reshape(-1),
                                #f"Predicted {variable}": pred_at_site,
                                "Land Cover": gdf["land cover"],
                                "x": gdf["x_idx"],
                                "y": gdf["y_idx"],
                                })
    df_sns_plot["Land Cover"] = df_sns_plot["Land Cover"].apply(lambda x: "Rapeseed" if x == "Rappesed" else x)
    fig, ax = plt.subplots(tight_layout=True, dpi=150, figsize=(5,5))
    s = pred_at_patch.shape
    
    if s[0]==1 and len(s)==3:
        im = ax.imshow(pred_at_patch.squeeze())
        plt.colorbar(im)
    elif (s[0] >= 3 and len(s)==3) or (s[1] >= 3 and len(s)==4):
        tensor_visu, _, _ = rgb_render(pred_at_patch.squeeze())
        im = ax.imshow(tensor_visu)
    land_cover = pd.unique(df_sns_plot["Land Cover"])
    for i, lc in enumerate(land_cover) :
        data = df_sns_plot[df_sns_plot["Land Cover"]==lc]
        ax.scatter(data['x'], data['y'], label=lc, marker="o", edgecolor="w")
    # g = sns.scatterplot(data=df_sns_plot, x='x', y="y", hue="Land Cover", ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend(loc="upper center", ncol=legend_columns, bbox_to_anchor=(0.5, -0.03))
    # sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, -0.03), ncol=legend_columns,
    #                     frameon=True)
    # fig.tight_layout()
    return fig, ax

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

frm4veg_data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/frm4veg_validation"
frm4veg_barrax_filename = "2B_20180516_FRM_Veg_Barrax_20180605"
frm4veg_wytham_filename = "2A_20180629_FRM_Veg_Wytham_20180703"
gdf_barrax_lai, s2_r_barrax, _, _, _ = load_frm4veg_data(frm4veg_data_dir, frm4veg_barrax_filename, variable="lai")
fig, ax = plot_frm4veg_validation_patch(gdf_barrax_lai, s2_r_barrax, variable="lai", legend_columns=3)
tikzplotlib_fix_ncols(fig)
tikzplotlib.save("test.tex")
fig.savefig("barrax_site.svg")
gdf_wytham_lai, s2_r_wytham, _, _, _ = load_frm4veg_data(frm4veg_data_dir, frm4veg_wytham_filename, variable="lai")
fig, ax = plot_frm4veg_validation_patch(gdf_wytham_lai, s2_r_wytham, variable="lai", legend_columns=2)
fig.savefig("wytham_site.svg")