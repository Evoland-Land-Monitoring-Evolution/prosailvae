import numpy as np
import numpyro
import torch

from dataset.generate_dataset import partial_sample_prosail_vars
from prosailvae.prosail_var_dists import get_prosail_var_dist


def simulate_prosail_samples_close_to_ref(
    s2_r_ref,
    noise=0,
    psimulator=None,
    ssimulator=None,
    lai=None,
    tts=None,
    tto=None,
    psi=None,
    eps_mae=1e-3,
    max_iter=100,
    samples_per_iter=1024,
    prosail_var_dist_type="legacy",
):
    best_prosail_vars = np.ones((1, 14))
    best_prosail_s2_sim = np.ones((1, 10))
    best_mae = np.inf
    iter = 0
    bins = 200
    aggregate_s2_hist = np.zeros((bins, 10))
    prosail_var_dist = get_prosail_var_dist(prosail_var_dist_type)
    with numpyro.handlers.seed(rng_seed=5):
        while best_mae > eps_mae and iter < max_iter:
            if iter % 10 == 0:
                print(f"{iter} - {best_mae}")
            prosail_vars = partial_sample_prosail_vars(
                prosail_var_dist,
                lai=lai,
                tts=tts,
                tto=tto,
                psi=psi,
                n_samples=samples_per_iter,
            )
            prosail_s2_sim = ssimulator(
                psimulator(torch.from_numpy(prosail_vars).view(-1, 14).float().detach())
            ).numpy()

            aggregate_s2_hist += np.apply_along_axis(
                lambda a: np.histogram(a, bins=bins, range=[0.0, 1.0])[0],
                0,
                prosail_s2_sim,
            )
            if noise > 0:
                raise NotImplementedError
            mare = np.abs((s2_r_ref - prosail_s2_sim) / (s2_r_ref + 1e-8)).mean(1)
            best_mae_iter = mare.min()

            if best_mae_iter < best_mae:
                best_mae = best_mae_iter
                best_prosail_vars = prosail_vars[mare.argmin(), :]
                best_prosail_s2_sim = prosail_s2_sim[mare.argmin(), :]
            iter += 1
    if iter == max_iter:
        print(
            f"WARNING : No sample with mae better than {eps_mae} was generated in {max_iter} iterations with {samples_per_iter} samples each ({max_iter * samples_per_iter} samples) "
        )
    else:
        print(
            f"A sample with mae better than {eps_mae} was generated in {max_iter} iterations with {samples_per_iter} samples each ({max_iter * samples_per_iter} samples) "
        )

    return (
        best_prosail_vars,
        best_prosail_s2_sim,
        max_iter * samples_per_iter,
        aggregate_s2_hist,
        best_mae,
    )


def simulate_lai_with_rec_error_hist(
    s2_r_ref,
    noise=0,
    psimulator=None,
    ssimulator=None,
    lai=None,
    tts=None,
    tto=None,
    psi=None,
    max_iter=100,
    samples_per_iter=1024,
    log_err=True,
    uniform_mode=True,
    lai_corr=False,
    lai_conv_override=None,
    bvnet_bands=False,
    prosail_var_dist_type="legacy",
):
    prosail_var_dist = get_prosail_var_dist(prosail_var_dist_type)
    best_mae = np.inf
    iter = 0
    bins = 200
    aggregate_lai_hist = np.zeros((bins, 1))
    heatmap = 0
    min_lai = prosail_var_dist.lai.low
    max_lai = prosail_var_dist.lai.high
    min_err = 0
    max_err = 2
    n_bin_err = 100
    n_bin_lai = 200
    xedges = np.linspace(min_lai, max_lai, n_bin_lai)
    if log_err:
        max_err = 5
        min_err = 5e-3
        yedges = np.logspace(np.log10(min_err), np.log10(max_err), n_bin_err)
    else:
        yedges = np.linspace(min_err, max_err, n_bin_err)
    with numpyro.handlers.seed(rng_seed=5):
        for iter in range(max_iter):
            if iter % 10 == 0:
                print(f"{iter} - {best_mae}")
            prosail_vars = partial_sample_prosail_vars(
                prosail_var_dist,
                lai=lai,
                tts=tts,
                tto=tto,
                psi=psi,
                n_samples=samples_per_iter,
                uniform_mode=uniform_mode,
                lai_corr=lai_corr,
                lai_conv_override=lai_conv_override,
            )
            lai_sim = prosail_vars[:, 6]
            prosail_s2_sim = ssimulator(
                psimulator(torch.from_numpy(prosail_vars).view(-1, 14).float().detach())
            ).numpy()
            aggregate_lai_hist += np.histogram(
                lai_sim, bins=n_bin_lai, range=[min_lai, max_lai]
            )[0].reshape(-1, 1)
            if noise > 0:
                raise NotImplementedError
            if bvnet_bands:
                mare = np.abs(
                    (
                        s2_r_ref[:, [1, 2, 3, 4, 5, 7, 8, 9]]
                        - prosail_s2_sim[:, [1, 2, 3, 4, 5, 7, 8, 9]]
                    )
                    / (s2_r_ref[:, [1, 2, 3, 4, 5, 7, 8, 9]] + 1e-8)
                ).mean(1)
            else:
                mare = np.abs((s2_r_ref - prosail_s2_sim) / (s2_r_ref + 1e-8)).mean(1)
            xs = lai_sim
            ys = mare
            hist, xedges, yedges = np.histogram2d(xs, ys, bins=[xedges, yedges])
            heatmap += hist
            best_mae_iter = mare.min()
            if best_mae_iter < best_mae:
                best_mae = best_mae_iter
                best_prosail_vars = prosail_vars[mare.argmin(), :]
                best_prosail_s2_sim = prosail_s2_sim[mare.argmin(), :]
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return (
        best_prosail_vars,
        best_prosail_s2_sim,
        heatmap,
        extent,
        aggregate_lai_hist,
        best_mae,
    )


def simulate_lai_with_rec_error_hist_with_enveloppe(
    s2_r_ref,
    noise=0,
    psimulator=None,
    ssimulator=None,
    lai=None,
    tts=None,
    tto=None,
    psi=None,
    max_iter=100,
    samples_per_iter=1024,
    log_err=True,
    uniform_mode=True,
    lai_corr=False,
    lai_conv_override=None,
    bvnet_bands=False,
    sigma=2,
    prosail_var_dist_type="legacy",
):
    AD = 0.01
    MD = 2
    prosail_var_dist = get_prosail_var_dist(prosail_var_dist_type)
    best_mae = np.inf
    iter = 0
    bins = 200
    aggregate_lai_hist = np.zeros((bins, 1))
    heatmap = 0
    min_lai = prosail_var_dist.lai.low
    max_lai = prosail_var_dist.lai.high
    min_err = 0
    max_err = 2
    n_bin_err = 100
    n_bin_lai = 200
    xedges = np.linspace(min_lai, max_lai, n_bin_lai)
    all_cases_in_enveloppe_err = []
    all_cases_in_enveloppe_LAI = []
    if log_err:
        max_err = 5
        min_err = 5e-3
        yedges = np.logspace(np.log10(min_err), np.log10(max_err), n_bin_err)
    else:
        yedges = np.linspace(min_err, max_err, n_bin_err)
    with numpyro.handlers.seed(rng_seed=5):
        for iter in range(max_iter):
            if iter % 10 == 0:
                print(f"{iter} - {best_mae}")
            prosail_vars = partial_sample_prosail_vars(
                prosail_var_dist,
                lai=lai,
                tts=tts,
                tto=tto,
                psi=psi,
                n_samples=samples_per_iter,
                uniform_mode=uniform_mode,
                lai_corr=lai_corr,
                lai_conv_override=lai_conv_override,
            )
            lai_sim = prosail_vars[:, 6]
            prosail_s2_sim = ssimulator(
                psimulator(torch.from_numpy(prosail_vars).view(-1, 14).float().detach())
            ).numpy()

            aggregate_lai_hist += np.histogram(
                lai_sim, bins=n_bin_lai, range=[min_lai, max_lai]
            )[0].reshape(-1, 1)
            if noise > 0:
                raise NotImplementedError
            if bvnet_bands:
                mare = np.abs(
                    (
                        s2_r_ref[:, [1, 2, 3, 4, 5, 7, 8, 9]]
                        - prosail_s2_sim[:, [1, 2, 3, 4, 5, 7, 8, 9]]
                    )
                    / (s2_r_ref[:, [1, 2, 3, 4, 5, 7, 8, 9]] + 1e-8)
                ).mean(1)
                enveloppe_low = (
                    s2_r_ref[:, [1, 2, 3, 4, 5, 7, 8, 9]] * (1 - sigma * MD / 100)
                    - sigma * AD
                )
                enveloppe_high = (
                    s2_r_ref[:, [1, 2, 3, 4, 5, 7, 8, 9]] * (1 + sigma * MD / 100)
                    + sigma * AD
                )
                cases_in_sigma_enveloppe = np.logical_and(
                    (prosail_s2_sim[:, [1, 2, 3, 4, 5, 7, 8, 9]] < enveloppe_high).all(
                        1
                    ),
                    (prosail_s2_sim[:, [1, 2, 3, 4, 5, 7, 8, 9]] > enveloppe_low).all(
                        1
                    ),
                )
            else:
                mare = np.abs((s2_r_ref - prosail_s2_sim) / (s2_r_ref + 1e-8)).mean(1)
                enveloppe_low = s2_r_ref * (1 - sigma * MD / 100) - sigma * AD
                enveloppe_high = s2_r_ref * (1 + sigma * MD / 100) + sigma * AD
                cases_in_sigma_enveloppe = np.logical_and(
                    (prosail_s2_sim < enveloppe_high).all(1),
                    (prosail_s2_sim > enveloppe_low).all(1),
                )

            if cases_in_sigma_enveloppe.any():
                all_cases_in_enveloppe_err.append(
                    mare[cases_in_sigma_enveloppe].reshape(-1, 1)
                )
                all_cases_in_enveloppe_LAI.append(
                    prosail_vars[cases_in_sigma_enveloppe, 6].reshape(-1, 1)
                )
            xs = lai_sim
            ys = mare
            hist, xedges, yedges = np.histogram2d(xs, ys, bins=[xedges, yedges])
            heatmap += hist
            best_mae_iter = mare.min()
            if best_mae_iter < best_mae:
                best_mae = best_mae_iter
                best_prosail_vars = prosail_vars[mare.argmin(), :]
                best_prosail_s2_sim = prosail_s2_sim[mare.argmin(), :]
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    if len(all_cases_in_enveloppe_err) > 0:
        all_cases_in_enveloppe_err = np.concatenate(all_cases_in_enveloppe_err, axis=0)
        all_cases_in_enveloppe_LAI = np.concatenate(all_cases_in_enveloppe_LAI, axis=0)
    return (
        best_prosail_vars,
        best_prosail_s2_sim,
        heatmap,
        extent,
        aggregate_lai_hist,
        best_mae,
        all_cases_in_enveloppe_err,
        all_cases_in_enveloppe_LAI,
    )
