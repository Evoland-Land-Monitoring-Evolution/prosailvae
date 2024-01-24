import os

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import torch


def plot_rsr(rsr, res_dir="."):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        sharey=True,
        facecolor="w",
        figsize=(10, 4),
        dpi=150,
        gridspec_kw={"width_ratios": [3, 1, 1.25]},
    )
    # bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A","B09","B10", "B11", "B12"]
    bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    lamb = np.arange(400, 2501)
    xlim_1 = (400, 1000)
    xlim_2 = (1500, 1700)
    xlim_3 = (2000, 2400)
    colors = plt.cm.tab10(np.linspace(0, 1, len(bands)))
    for i in range(len(bands)):
        rsr_i = rsr.squeeze()[i, :].numpy()
        non_zero_rsr = rsr_i[rsr_i > 0.001]
        non_zero_rsrl = lamb[rsr_i > 0.001]
        if non_zero_rsrl[-1] < xlim_1[1]:
            ax1.plot(non_zero_rsrl, non_zero_rsr, label=bands[i], color=colors[i])
        elif non_zero_rsrl[-1] < xlim_2[1]:
            ax2.plot(non_zero_rsrl, non_zero_rsr, label=bands[i], color=colors[i])
        elif non_zero_rsrl[-1] < xlim_3[1]:
            ax3.plot(non_zero_rsrl, non_zero_rsr, label=bands[i], color=colors[i])
        else:
            raise ValueError

    ax1.set_xlim(xlim_1[0], xlim_1[1])
    # ax2.set_xlim(1300,1700)
    ax2.set_xlim(xlim_2[0], xlim_2[1])
    ax3.set_xlim(xlim_3[0], xlim_3[1])

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax1.yaxis.tick_left()
    # ax1.tick_params(labelright='off')
    # ax2.tick_params(labelright='off')
    ax2.yaxis.set_visible(False)
    # ax2.set_yticklabels([])

    ax3.yaxis.tick_right()

    ax1.set_ylabel("Relative Spectral Response")
    ax2.set_xlabel("Wavelength (nm)")
    box = ax3.get_position()
    ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax3.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(which="both", axis="both", direction="in")
    ax2.tick_params(which="both", axis="both", direction="in")
    ax3.tick_params(which="both", axis="both", direction="in")

    ax1.tick_params(which="both", width=1)
    ax1.tick_params(which="major", length=5)
    ax1.tick_params(which="minor", length=3)
    ax2.tick_params(which="major", length=5)
    ax2.tick_params(which="minor", length=3)
    ax3.tick_params(which="major", length=5)
    ax3.tick_params(which="minor", length=3)

    ax1t = ax1.twiny()
    ax1t.spines["right"].set_visible(False)
    ax1t.tick_params(which="both", direction="in")
    ax1t.tick_params(which="major", length=5)
    ax1t.tick_params(which="minor", length=3)
    ax1t.xaxis.set_minor_locator(AutoMinorLocator())
    ax1t.set_xticklabels([])

    ax2t = ax2.twiny()
    ax2t.spines["left"].set_visible(False)
    ax2t.spines["right"].set_visible(False)
    ax2t.tick_params(which="both", direction="in")
    ax2t.tick_params(which="major", length=5)
    ax2t.tick_params(which="minor", length=3)
    ax2t.xaxis.set_minor_locator(AutoMinorLocator())
    ax2t.set_xticklabels([])
    ax3.tick_params(
        axis="x", which="both", bottom=True, top=True, labelbottom=True, labeltop=False
    )
    if res_dir is not None:
        fig.savefig(res_dir + "/rsr.svg")
    return fig, (ax1, ax2, ax3)


def plot_solar_rsr(solar, res_dir):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    fig, ax = plt.subplots(1, 1, facecolor="w", figsize=(10, 4), dpi=150)
    ax.plot(torch.arange(400, 2501), solar.squeeze() / solar.max(), "k")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized spectral response")
    ax.set_xlim(400, 2350)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in")
    ax.tick_params(which="major", length=5)
    ax.tick_params(which="minor", length=3)
    if res_dir is not None:
        fig.savefig(res_dir + "/solar_rsr.svg")
    return fig, ax


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def main(rsr_file="", res_dir=""):
    prospect_range = (400, 2500)
    bands = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
    device = "cpu"
    prospect_range = prospect_range
    rsr = torch.from_numpy(np.loadtxt(rsr_file, unpack=True)).to(device)
    nb_bands = rsr.shape[0] - 2
    rsr_range = (int(rsr[0, 0].item() * 1000), int(rsr[0, -1].item() * 1000))
    nb_lambdas = prospect_range[1] - prospect_range[0] + 1
    rsr_prospect = torch.zeros([rsr.shape[0], nb_lambdas]).to(device)
    rsr_prospect[0, :] = torch.linspace(
        prospect_range[0], prospect_range[1], nb_lambdas
    ).to(device)
    rsr_prospect[1:, : -(prospect_range[1] - rsr_range[1])] = rsr[
        1:, (prospect_range[0] - rsr_range[0]) :
    ]

    solar = rsr_prospect[1, :].unsqueeze(0)
    rsr = rsr_prospect[2:, :].unsqueeze(0)
    rsr = rsr[:, bands, :]
    fig, axs = plot_rsr(rsr, res_dir=None)
    tikzplotlib_fix_ncols(fig)
    tikzplotlib.save(os.path.join(res_dir, "rsr.tex"))
    fig, ax = plot_solar_rsr(solar, res_dir=None)
    tikzplotlib_fix_ncols(fig)
    tikzplotlib.save(os.path.join(res_dir, "solar.tex"))
    pass


if __name__ == "__main__":
    rsr_file = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/sentinel2.rsr"
    res_dir = "/home/yoel/Documents/Productions/2nd article/src/figures"
    main(rsr_file, res_dir)
