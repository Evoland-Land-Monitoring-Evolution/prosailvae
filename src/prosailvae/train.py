"""
Created on Mon Nov 14 14:20:44 2022

@author: yoel
"""
import argparse
import logging
import logging.config
import os
import shutil
import socket
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from dataset.loaders import get_simloader, get_train_valid_test_loader_from_patches
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
from utils.utils import (
    get_RAM_usage,
    get_total_RAM,
    load_standardize_coeffs,
    plot_grad_flow,
    save_dict,
)

from metrics.results import (
    get_res_dir_path,
    plot_losses,
    save_results_on_s2_data,
    save_validation_results,
)
from prosailvae.prosail_vae import (
    ProsailVAEConfig,
    get_prosail_vae_config,
    load_params,
    load_prosail_vae_with_hyperprior,
)
from prosailvae.ProsailSimus import get_bands_idx

torch.autograd.set_detect_anomaly(True)

CUDA_LAUNCH_BLOCKING = 1
LOGGER_NAME = "PROSAIL-VAE logger"
PC_SOCKET_NAME = "CELL200973"  # toggle options for dev and debug on PC


@dataclass
class DatasetConfig:
    dataset_file_prefix: str = "small_test_"


@dataclass
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 1
    lr: float = 0.0001
    test_size: float = 0.01
    valid_ratio: float = 0.01
    k_fold: int = 0
    n_fold: int | None = None
    n_samples: int = 2


def get_training_config(params):
    return TrainingConfig(
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        lr=params["lr"],
        test_size=params["test_size"],
        valid_ratio=params["valid_ratio"],
        k_fold=params["k_fold"],
        n_fold=params["n_fold"],
        n_samples=params["n_samples"],
    )


@dataclass
class ModelConfig:
    supervised: bool = False
    beta_kl: float = 1
    beta_index: float = 1


def get_prosailvae_train_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description="Parser for data generation")

    parser.add_argument(
        "-n", dest="n_fold", help="number k of fold", type=int, default=0
    )

    parser.add_argument(
        "-c",
        dest="config_file",
        help="name of config json file on config directory.",
        type=str,
        default="config.json",
    )

    parser.add_argument(
        "-cd",
        dest="config_dir",
        help="path to config directory",
        type=str,
        default="",
        required=True,
    )

    parser.add_argument(
        "-x",
        dest="n_xp",
        help="Number of experience (to use in case of kfold)",
        type=int,
        default=1,
    )

    parser.add_argument(
        "-o",
        dest="overwrite_xp",
        help="Allow overwrite of experiment (fold)",
        type=bool,
        default=True,
    )

    parser.add_argument(
        "-r",
        dest="root_results_dir",
        help="path to root results directory",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-rsr", dest="rsr_dir", help="directory of rsr_file", type=str, required=True
    )

    parser.add_argument(
        "-a",
        dest="xp_array",
        help="array training (false for single xp) ",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "-p",
        dest="plot_results",
        help="toggle results plotting",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "-w",
        dest="weiss_mode",
        # TODO replace weiss mode with explicit band selection
        help="removes B2 and B8 bands for validation with weiss data",
        type=bool,
        default=False,
    )
    return parser


def switch_loss(epoch, n_epoch, PROSAIL_VAE, swith_ratio=0.75):
    loss_type = PROSAIL_VAE.decoder.loss_type
    if loss_type == "hybrid_nll":
        if epoch > swith_ratio * n_epoch:
            PROSAIL_VAE.decoder.loss_type = "full_nll"


def initialize_by_training(
    n_models: int,
    n_epochs: int,
    n_samples: int,
    train_loader,
    valid_loader,
    lr: float,
    logger,
    pv_config: ProsailVAEConfig,
    pv_config_hyper: ProsailVAEConfig | None = None,
    break_at_rec_loss=None,
    max_sec=3600,
):
    """
    Initialize prosial_vae by running a few models for several epochs at high lr,
      and selecting the best.
    """
    min_valid_loss = torch.inf
    broke_at_rec = False
    logger.info(f"Intializing by training {n_models} models for {n_epochs} epochs:")
    best_model_idx = 0
    t0 = time.time()
    for i in range(n_models):
        logger.info(f"=================== Model {i} / {n_models} ================")
        prosail_vae = load_prosail_vae_with_hyperprior(
            pv_config=pv_config,
            pv_config_hyper=pv_config_hyper,
            logger_name=LOGGER_NAME,
        )
        optimizer = optim.Adam(prosail_vae.parameters(), lr=lr, weight_decay=1e-2)
        _, all_valid_loss_df, _, _ = training_loop(
            prosail_vae,
            optimizer,
            n_epochs,
            train_loader,
            valid_loader,
            res_dir=None,
            n_samples=n_samples,
            lr_recompute=None,
            exp_lr_decay=-1,
            plot_gradient=False,  # parser.plot_results,
            lr_recompute_mode=False,
            cycle_training=False,
            accum_iter=1,
            lrs_threshold=0.01,
            max_sec=None,
        )

        model_min_loss = all_valid_loss_df["loss_sum"].values.min()
        if min_valid_loss > model_min_loss:
            min_valid_loss = model_min_loss
            best_model_idx = i
            prosail_vae.save_ae(
                n_epochs, optimizer, model_min_loss, pv_config.vae_save_file_path
            )
        if break_at_rec_loss is not None:
            if all_valid_loss_df["rec_loss"].values.min() <= break_at_rec_loss:
                logger.info(
                    f"Model {i} has gone under threshold loss "
                    f"{all_valid_loss_df['rec_loss'].values.min()}"
                    f" < {break_at_rec_loss}."
                )
                broke_at_rec = True
                break
        if time.time() - t0 > max_sec:
            break
    logger.info(f"Best model is model {best_model_idx}.")
    logger.info("=====================================================================")
    return broke_at_rec


def training_loop(
    prosail_vae,
    optimizer,
    n_epoch,
    train_loader,
    valid_loader,
    res_dir=None,
    n_samples=20,
    lr_recompute=None,
    exp_lr_decay=0,
    plot_gradient=False,
    lr_recompute_mode=True,
    cycle_training=False,
    accum_iter=1,
    lrs_threshold=0.01,
    lr_init=5e-4,
    validation_at_every_epoch=None,
    validation_dir=None,
    frm4veg_data_dir=None,
    frm4veg_2021_data_dir=None,
    belsar_data_dir=None,
    lai_cyclical_loader=None,
    max_sec=None,
):
    t_init = time.time()
    cyclical_lai_precomputed = True
    if lai_cyclical_loader is None:
        lai_cyclical_loader = valid_loader
        cyclical_lai_precomputed = False
    logger = logging.getLogger(LOGGER_NAME)
    tbeg = time.time()
    if prosail_vae.decoder.loss_type == "mse":
        n_samples = 1
        logger.info("MSE Loss enabled, setting number of monte-carlo samples to 1")
    all_train_loss_df = pd.DataFrame()
    all_valid_loss_df = pd.DataFrame()
    info_df = pd.DataFrame()
    best_val_loss = torch.inf
    total_ram = get_total_RAM()
    if exp_lr_decay > 0:
        if lr_recompute_mode:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=exp_lr_decay
            )
        else:
            if lr_recompute is not None:
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer,
                    patience=lr_recompute,
                    threshold=lrs_threshold,
                    threshold_mode="abs",
                )

    max_train_samples_per_epoch = 50
    max_valid_samples_per_epoch = 200
    if socket.gethostname() == PC_SOCKET_NAME:
        max_train_samples_per_epoch = 5
        max_valid_samples_per_epoch = 2
    all_cyclical_rmse = []
    logger.info(f"\nWill train for {n_epoch} epochs")
    with logging_redirect_tqdm():
        for epoch in trange(n_epoch, desc="PROSAIL-VAE training", leave=True):
            logger.info(f"\n Starting epoch {epoch} / {n_epoch}")
            if validation_at_every_epoch is not None:
                if epoch % validation_at_every_epoch == 0:
                    validation_dir_at_epoch = os.path.join(
                        validation_dir, f"epoch_{epoch}"
                    )
                    if not os.path.exists(validation_dir_at_epoch):
                        os.makedirs(validation_dir_at_epoch)
                    _, cyclical_rmse = prosail_vae.get_cyclical_metrics_from_loader(
                        lai_cyclical_loader, lai_precomputed=cyclical_lai_precomputed
                    )
                    all_cyclical_rmse.append(cyclical_rmse.cpu().item())
                    save_validation_results(
                        prosail_vae,
                        validation_dir_at_epoch,
                        frm4veg_data_dir=frm4veg_data_dir,
                        frm4veg_2021_data_dir=frm4veg_2021_data_dir,
                        belsar_data_dir=belsar_data_dir,
                        model_name=f"pvae_{epoch}",
                        method="simple_interpolate",
                        mode="sim_tg_mean",
                        remove_files=True,
                    )

            t0 = time.time()
            if optimizer.param_groups[0]["lr"] < 5e-8:
                if not cycle_training:
                    break  # stop training if lr too low
                for g in optimizer.param_groups:
                    g["lr"] = lr_init
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer,
                    patience=lr_recompute,
                    threshold=0.01,
                    threshold_mode="abs",
                )
            if lr_recompute_mode:
                raise NotImplementedError

            info_df = pd.concat(
                [
                    info_df,
                    pd.DataFrame(
                        {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]},
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )

            try:
                train_loss_dict = prosail_vae.fit(
                    train_loader,
                    optimizer,
                    n_samples=n_samples,
                    max_samples=max_train_samples_per_epoch,
                    accum_iter=accum_iter,
                )
                if plot_gradient and res_dir is not None:
                    if not os.path.isdir(os.path.join(res_dir, "gradient_flows")):
                        os.makedirs(os.path.join(res_dir, "gradient_flows"))
                    plot_grad_flow(
                        prosail_vae,
                        savefile=os.path.join(
                            os.path.join(res_dir, "gradient_flows"),
                            f"gradient_flows/grad_flow_{epoch}.svg",
                        ),
                    )
            except Exception as exc:
                logger.error(f"Error during Training at epoch {epoch} !")
                logger.error("Original error :")
                logger.error(str(exc))
                print(f"Error during Training at epoch {epoch} !")
                print("Original error :")
                print(str(exc))
                traceback.print_exc()
                break
            try:
                valid_loss_dict = prosail_vae.validate(
                    valid_loader,
                    n_samples=n_samples,
                    max_samples=max_valid_samples_per_epoch,
                )
                if exp_lr_decay > 0:
                    if lr_recompute_mode:
                        lr_scheduler.step()
                    else:
                        lr_scheduler.step(valid_loss_dict["loss_sum"])

            except Exception as exc:
                logger.error(f"Error during Validation at epoch {epoch} !")
                logger.error("Original error :")
                logger.error(str(exc))
                print(f"Error during Validation at epoch {epoch} !")
                print("Original error :")
                print(str(exc))
                traceback.print_exc()
            t1 = time.time()
            ram_usage = get_RAM_usage()
            train_loss_info = "- ".join(
                [
                    f"{key}: {'{:.2E}'.format(train_loss_dict[key])} "
                    for key in train_loss_dict.keys()
                ]
            )
            valid_loss_info = "- ".join(
                [
                    f"{key}: {'{:.2E}'.format(valid_loss_dict[key])} "
                    for key in valid_loss_dict.keys()
                ]
            )
            logger.info(
                f"\n{epoch} -- RAM: {ram_usage} / {total_ram} -- lr: "
                f"{'{:.2E}'.format(optimizer.param_groups[0]['lr'])} --"
                f" {'{:.1f}'.format(t1-t0)} s -- {train_loss_info} -- {valid_loss_info}"
            )
            train_loss_dict["epoch"] = epoch
            valid_loss_dict["epoch"] = epoch
            all_train_loss_df = pd.concat(
                [all_train_loss_df, pd.DataFrame(train_loss_dict, index=[0])],
                ignore_index=True,
            )
            all_valid_loss_df = pd.concat(
                [all_valid_loss_df, pd.DataFrame(valid_loss_dict, index=[0])],
                ignore_index=True,
            )
            if valid_loss_dict["loss_sum"] < best_val_loss:
                best_val_loss = valid_loss_dict["loss_sum"]
                if res_dir is not None:
                    prosail_vae.save_ae(
                        epoch,
                        optimizer,
                        best_val_loss,
                        os.path.join(res_dir, "prosailvae_weights.tar"),
                    )
            t_end = time.time()
            if max_sec is not None:
                if t_end - t_init > max_sec:
                    logger.info(
                        f"Time limit of {max_sec} seconds over, ending training early."
                    )
                    break
            # if os.path.isfile(os.path.join(res_dir, "stop.txt")):
            #     break
    if n_epoch < 1:  # In case we just want to plot results
        all_train_loss_df = pd.DataFrame(
            data={"loss_sum": 10000, "epoch": 0}, index=[0]
        )
        all_valid_loss_df = pd.DataFrame(
            data={"loss_sum": 10000, "epoch": 0}, index=[0]
        )
        info_df = pd.DataFrame(data={"lr": 10000, "epoch": 0}, index=[0])
        prosail_vae.save_ae(
            0, optimizer, 0, os.path.join(res_dir, "prosailvae_weights.tar")
        )
    tend = time.time()
    logger.info(f"Total training time: {tend-tbeg:.1f} seconds")
    return all_train_loss_df, all_valid_loss_df, info_df, all_cyclical_rmse


def setup_training():
    """
    Read parser and config files to launch training
    """
    if socket.gethostname() == PC_SOCKET_NAME:
        args = [
            "-n",
            "0",
            "-c",
            "config.json",
            "-x",
            "1",
            "-o",
            "True",
            "-r",
            "",
            "-rsr",
            "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/",
            "-a",
            "False",
            "-p",
            "False",
            "-cd",
            "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/config/",
        ]
        parser = get_prosailvae_train_parser().parse_args(args)
    else:
        parser = get_prosailvae_train_parser().parse_args()

    xp_array = parser.xp_array
    job_array_dir = None
    if xp_array:
        job_array_dir = os.path.join(parser.root_results_dir, os.pardir)
    config_dir = parser.config_dir
    params = load_params(config_dir, config_file=parser.config_file, parser=parser)
    model_name = parser.config_file[:-5]

    sim_data_dir = params["sim_data_dir"]
    s2_data_dir = params["s2_data_dir"]
    assert parser.n_fold < parser.n_xp

    root_results_dir = parser.root_results_dir
    res_dir = get_res_dir_path(
        root_results_dir, params, parser.n_xp, parser.overwrite_xp
    )
    save_dict(params, res_dir + "/config.json")
    params["vae_save_file_path"] = res_dir + "/prosailvae_weights.tar"

    logging.basicConfig(
        filename=res_dir + "/training_log.log", level=logging.INFO, force=True
    )
    logger_name = "PROSAIL-VAE logger"
    # create logger
    logger = logging.getLogger(logger_name)
    logger.info("Starting training of PROSAIL-VAE.")
    logger.info(
        "========================================================================"
    )
    logger.info("Parameters are : ")
    for _, key in enumerate(params):
        logger.info(f"{key} : {params[key]}")
    logger.info(
        "========================================================================"
    )
    if params["supervised_kl"]:
        logger.info("Supervised KL loss (hyperprior) enabled.")

        logger.info(
            f"copying {params['supervised_config_file']} into "
            f"{res_dir+'/sup_kl_model_config.json'}"
        )
        shutil.copyfile(
            params["supervised_config_file"], res_dir + "/sup_kl_model_config.json"
        )
        logger.info(
            f"copying {params['supervised_weight_file']} into "
            f"{res_dir+'/sup_kl_model_weights.tar'}"
        )
        shutil.copyfile(
            params["supervised_weight_file"], res_dir + "/sup_kl_model_weights.tar"
        )
        params_sup_kl_model = load_params(
            res_dir, "/sup_kl_model_config.json", parser=None
        )
        params_sup_kl_model["vae_load_file_path"] = (
            res_dir + "/sup_kl_model_weights.tar"
        )
        params_sup_kl_model["load_model"] = True

        sup_kl_io_coeffs = load_standardize_coeffs(
            os.path.dirname(params["supervised_config_file"])
        )
        torch.save(sup_kl_io_coeffs.bands.loc, res_dir + "/norm_mean.pt")
        torch.save(sup_kl_io_coeffs.bands.scale, res_dir + "/norm_std.pt")
        torch.save(sup_kl_io_coeffs.idx.loc, res_dir + "/idx_loc.pt")
        torch.save(sup_kl_io_coeffs.idx.scale, res_dir + "/idx_scale.pt")
        torch.save(sup_kl_io_coeffs.angles.loc, res_dir + "/angles_loc.pt")
        torch.save(sup_kl_io_coeffs.angles.scale, res_dir + "/angles_scale.pt")

    else:
        params_sup_kl_model = None
        sup_kl_io_coeffs = None
    frm4veg_data_dir = params["frm4veg_data_dir"]
    frm4veg_2021_data_dir = params["frm4veg_2021_data_dir"]
    belsar_dir = params["belsar_dir"]
    return (
        params,
        parser,
        res_dir,
        sim_data_dir,
        s2_data_dir,
        params_sup_kl_model,
        job_array_dir,
        sup_kl_io_coeffs,
        frm4veg_data_dir,
        frm4veg_2021_data_dir,
        belsar_dir,
        model_name,
    )


def get_training_data(params, logger):
    s2_data_dir = params["s2_data_dir"]
    sim_data_dir = params["sim_data_dir"]
    bands, prosail_bands = get_bands_idx(params["weiss_bands"])
    if params["simulated_dataset"]:
        logger.info(
            f"Loading training and validation loader in"
            f" {sim_data_dir}/{params['dataset_file_prefix']}..."
        )
        if params["batch_size"] < 2:
            raise ValueError("With simulated data, batch_size cannot be inferior to 2.")
        train_loader, valid_loader = get_simloader(
            valid_ratio=params["valid_ratio"],
            file_prefix=params["dataset_file_prefix"],
            sample_ids=None,
            batch_size=params["batch_size"],
            data_dir=sim_data_dir,
        )
        data_dir = sim_data_dir
    else:
        logger.info(f"Loading training and validation loader in" f" {s2_data_dir} ...")
        train_loader, valid_loader, _ = get_train_valid_test_loader_from_patches(
            s2_data_dir,
            batch_size=params["batch_size"],
            num_workers=0,
            max_valid_samples=1000,
        )
        data_dir = s2_data_dir

    if params["apply_norm_rec"]:
        io_coeffs = load_standardize_coeffs(
            data_dir,
            params["dataset_file_prefix"],
            n_idx=0 if params["weiss_bands"] else 4,
        )
    else:
        io_coeffs = load_standardize_coeffs(
            None, params["dataset_file_prefix"], n_idx=0 if params["weiss_bands"] else 4
        )

    logger.info(
        f"Training ({len(train_loader.dataset)} samples) "
        f"and validation ({len(valid_loader.dataset)} samples) loaders, loaded."
    )

    return train_loader, valid_loader, io_coeffs, bands, prosail_bands


def train_prosailvae(
    params,
    parser,
    res_dir,
    params_sup_kl_model,
    sup_kl_io_coeffs,
    validation_dir=None,
    frm4veg_data_dir=None,
    frm4veg_2021_data_dir=None,
    belsar_data_dir=None,
    lai_cyclical_loader=None,
):
    """
    Intializes and trains a prosail instance
    """
    logger = logging.getLogger(LOGGER_NAME)
    (train_loader, valid_loader, io_coeffs, bands, prosail_bands) = get_training_data(
        params, logger
    )
    if params["load_model"]:
        vae_load_file_path = os.path.join(
            params["vae_load_dir_path"], "prosailvae_weights.tar"
        )
        io_coeffs = load_standardize_coeffs(
            params["vae_load_dir_path"],
            # params["dataset_file_prefix"],
            n_idx=0 if params["weiss_bands"] else 4,
        )
    else:
        vae_load_file_path = None

    torch.save(io_coeffs.bands.loc, res_dir + "/norm_mean.pt")
    torch.save(io_coeffs.bands.scale, res_dir + "/norm_std.pt")
    torch.save(io_coeffs.idx.loc, res_dir + "/idx_loc.pt")
    torch.save(io_coeffs.idx.scale, res_dir + "/idx_scale.pt")
    torch.save(io_coeffs.angles.loc, res_dir + "/angles_loc.pt")
    torch.save(io_coeffs.angles.scale, res_dir + "/angles_scale.pt")
    logger.info(f"io_coeffs.bands.loc : {io_coeffs.bands.loc}")
    logger.info(f"io_coeffs.bands.scale : {io_coeffs.bands.scale}")
    logger.info(f"io_coeffs.idx.loc : {io_coeffs.idx.loc}")
    logger.info(f"io_coeffs.idx.scale : {io_coeffs.idx.scale}")

    params["vae_load_file_path"] = vae_load_file_path
    training_config = get_training_config(params)
    pv_config = get_prosail_vae_config(
        params,
        bands=bands,
        prosail_bands=prosail_bands,
        inference_mode=False,
        rsr_dir=parser.rsr_dir,
        io_coeffs=io_coeffs,
    )
    pv_config_hyper = None
    if params_sup_kl_model is not None:
        bands_hyper, prosail_bands_hyper = get_bands_idx(
            params_sup_kl_model["weiss_bands"]
        )
        pv_config_hyper = get_prosail_vae_config(
            params_sup_kl_model,
            bands=bands_hyper,
            prosail_bands=prosail_bands_hyper,
            inference_mode=True,
            rsr_dir=parser.rsr_dir,
            io_coeffs=sup_kl_io_coeffs,
        )
    if params["init_model"]:
        n_models = params["n_init_models"]
        lr = params["init_lr"]
        n_epochs = params["n_init_epochs"]
        broke_at_rec = initialize_by_training(
            n_models=n_models,
            n_epochs=n_epochs,
            train_loader=train_loader,
            valid_loader=valid_loader,
            lr=lr,
            logger=logger,
            n_samples=training_config.n_samples,
            pv_config=pv_config,
            pv_config_hyper=pv_config_hyper,
            break_at_rec_loss=params["break_init_at_rec_loss"],
        )
        if params["break_init_at_rec_loss"] is not None and not broke_at_rec:
            broke_at_rec = initialize_by_training(
                n_models=n_models,
                n_epochs=n_epochs,
                train_loader=train_loader,
                valid_loader=valid_loader,
                lr=1e-3,
                logger=logger,
                n_samples=training_config.n_samples,
                pv_config=pv_config,
                pv_config_hyper=pv_config_hyper,
                break_at_rec_loss=params["break_init_at_rec_loss"],
            )
            if not broke_at_rec:
                logger.info("No good initialization was found !")
        # Changing config to load the best model initialized
        params["load_model"] = True
        params["vae_load_file_path"] = params["vae_save_file_path"]
        pv_config = get_prosail_vae_config(
            params,
            bands=bands,
            prosail_bands=prosail_bands,
            inference_mode=False,
            rsr_dir=parser.rsr_dir,
            io_coeffs=io_coeffs,
        )

    prosail_vae = load_prosail_vae_with_hyperprior(
        pv_config=pv_config, pv_config_hyper=pv_config_hyper, logger_name=LOGGER_NAME
    )
    lr = params["lr"]
    lr_recompute_mode = params["lr_recompute_mode"]
    optimizer = optim.Adam(prosail_vae.parameters(), lr=lr, weight_decay=1e-2)
    logger.info("PROSAIL-VAE and optimizer initialized.")

    # Training
    logger.info(f"Starting Training loop for {params['epochs']} epochs.")

    (all_train_loss_df, all_valid_loss_df, info_df, all_cyclical_rmse) = training_loop(
        prosail_vae,
        optimizer,
        params["epochs"],
        train_loader,
        valid_loader,
        res_dir=res_dir,
        n_samples=params["n_samples"],
        lr_recompute=params["lr_recompute"],
        exp_lr_decay=params["exp_lr_decay"],
        plot_gradient=False,  # parser.plot_results,
        lr_recompute_mode=lr_recompute_mode,
        cycle_training=params["cycle_training"],
        accum_iter=params["accum_iter"],
        lrs_threshold=params["lrs_threshold"],
        lr_init=params["lr"],
        validation_at_every_epoch=params["validation_at_every_epoch"],
        validation_dir=validation_dir,
        frm4veg_data_dir=frm4veg_data_dir,
        frm4veg_2021_data_dir=frm4veg_2021_data_dir,
        belsar_data_dir=belsar_data_dir,
        lai_cyclical_loader=lai_cyclical_loader,
        max_sec=10.5 * 3600,
    )
    logger.info("Training Completed !")
    # Loading best model
    params["load_model"] = True
    params["vae_load_file_path"] = params["vae_save_file_path"]
    pv_config = get_prosail_vae_config(
        params,
        bands=bands,
        prosail_bands=prosail_bands,
        inference_mode=False,
        rsr_dir=parser.rsr_dir,
        io_coeffs=io_coeffs,
    )

    prosail_vae = load_prosail_vae_with_hyperprior(
        pv_config=pv_config, pv_config_hyper=pv_config_hyper, logger_name=LOGGER_NAME
    )
    if len(all_cyclical_rmse):
        pd.DataFrame(all_cyclical_rmse).to_csv(
            os.path.join(res_dir, "cyclical_rmse.csv")
        )
    return prosail_vae, all_train_loss_df, all_valid_loss_df, info_df


def configureEmissionTracker(parser):
    logger = logging.getLogger(LOGGER_NAME)
    try:
        from codecarbon import OfflineEmissionsTracker

        tracker = OfflineEmissionsTracker(
            country_iso_code="FRA", output_dir=parser.root_results_dir
        )
        tracker.start()
        useEmissionTracker = True
    except Exception:
        logger.error(
            "Couldn't start codecarbon ! Emissions not tracked for this execution."
        )
        useEmissionTracker = False
        tracker = None
    return tracker, useEmissionTracker


def save_array_xp_path(job_array_dir, res_dir):
    if job_array_dir is not None:
        if not os.path.isfile(job_array_dir + "/results_directory_names.txt"):
            with open(job_array_dir + "/results_directory_names.txt", "w") as outfile:
                outfile.write(f"{res_dir}\n")
        else:
            with open(job_array_dir + "/results_directory_names.txt", "a") as outfile:
                outfile.write(f"{res_dir}\n")


def main():
    (
        params,
        parser,
        res_dir,
        sim_data_dir,
        s2_data_dir,
        params_sup_kl_model,
        job_array_dir,
        sup_kl_io_coeffs,
        frm4veg_data_dir,
        frm4veg_2021_data_dir,
        belsar_data_dir,
        model_name,
    ) = setup_training()
    try:
        lai_cyclical_loader = None

        validation_dir = os.path.join(res_dir, "validation")
        if not Path(validation_dir).exists():
            os.makedirs(validation_dir)
        (prosail_vae, all_train_loss_df, all_valid_loss_df, info_df) = train_prosailvae(
            params,
            parser,
            res_dir,
            params_sup_kl_model,
            sup_kl_io_coeffs=sup_kl_io_coeffs,
            validation_dir=validation_dir,
            frm4veg_data_dir=frm4veg_data_dir,
            frm4veg_2021_data_dir=frm4veg_2021_data_dir,
            belsar_data_dir=belsar_data_dir,
            lai_cyclical_loader=lai_cyclical_loader,
        )
        plot_losses(
            res_dir,
            all_train_loss_df,
            all_valid_loss_df,
            info_df,
            LOGGER_NAME=LOGGER_NAME,
            plot_results=parser.plot_results,
        )
        min_loss = (
            all_valid_loss_df["rec_loss"].min()
            if "rec_loss" in all_valid_loss_df.columns
            else all_valid_loss_df["loss_sum"].min()
        )
        min_loss_df = pd.DataFrame({"Loss": [min_loss]})
        cyclical_rmse_df = pd.DataFrame(data={"cyclical_rmse": [1.0]})
        _, valid_loader, test_loader = get_train_valid_test_loader_from_patches(
            params["s2_data_dir"],
            bands=torch.arange(10),
            batch_size=1,
            num_workers=0,
            max_valid_samples=None,
        )
        cyclical_rmse = prosail_vae.get_cyclical_rmse_from_loader(
            valid_loader, lai_precomputed=False
        )
        cyclical_rmse_df = pd.DataFrame(data={"cyclical_rmse": [cyclical_rmse.item()]})
        lai_cyclical_loader = valid_loader
        info_test_data = np.load(os.path.join(params["s2_data_dir"], "test_info.npy"))

        save_results_on_s2_data(
            prosail_vae,
            test_loader,
            res_dir,
            LOGGER_NAME=LOGGER_NAME,
            plot_results=parser.plot_results,
            info_test_data=info_test_data,
            max_test_patch=50 if not socket.gethostname() == PC_SOCKET_NAME else 2,
            lai_cyclical_loader=lai_cyclical_loader,
        )
        global_results_df = pd.concat(
            (pd.DataFrame({"model": [model_name]}), cyclical_rmse_df, min_loss_df),
            axis=1,
        )
        res_df_filename = os.path.join(
            res_dir,
            "model_results.csv",
        )
        if not os.path.isfile(res_df_filename):
            global_results_df.to_csv(
                res_df_filename, header=global_results_df.columns, index=False
            )
        else:  # else it exists so append without writing the header
            global_results_df.to_csv(
                res_df_filename, mode="a", index=False, header=False
            )

        save_array_xp_path(job_array_dir, res_dir)
        if params["k_fold"] > 1:
            save_array_xp_path(os.path.join(res_dir, os.path.pardir), res_dir)
    except Exception as exc:
        traceback.print_exc()
        print(exc)
    pass


if __name__ == "__main__":
    main()
