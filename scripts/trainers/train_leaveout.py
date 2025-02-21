import glob
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import pyro
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt

import pickle
import os
import numpy as np
import argparse

import warnings

warnings.filterwarnings("ignore")

# %%
import sys

# Adjust this path if necessary
sys.path.append("/work/submit/josemm/WORKS/Theory/BANANNE/")
from bannane.src.config import Config
from bannane.src.data_loader import MultiIsotopeDataLoader
from bannane.src.preprocess import (
    split_data_multi_isotope,
    preprocess_data_multi_isotope,
)
from bannane.src.model import HierarchicalBNN
from bannane.src.trainer import Trainer, bannane_inference
from bannane.src.plotting import plot_training_losses_history, plot_mape_history


def train_leave_out(n_out_list: List[int], config: Config, logs_dir: str = ""):
    """
    Train a model leaving out a set of nuclei (given by n_out_list).
    After training, make predictions on:
      (a) the initial validation set (from available data), and
      (b) the complete extrapolation set (all fidelities for the left-out nuclei).
    The predictions are written as CSV files.
    """
    # Set data and model save paths to incorporate the left-out nuclei
    n_out_str = "_".join(map(str, n_out_list))
    config.data_directory = (
        "/work/submit/josemm/WORKS/Theory/BANANNE/bannane/DATA/all_o"
    )
    config.model_save_path = os.path.join(logs_dir, f"model_leaveout_{n_out_str}.pth")
    config.temperature_save_path = os.path.join(
        logs_dir, f"temperature_leaveout_{n_out_str}.pth"
    )
    config.scaler_X_path = os.path.join(logs_dir, f"scaler_X_leaveout_{n_out_str}.pth")
    config.scaler_y_path = os.path.join(logs_dir, f"scaler_y_leaveout_{n_out_str}.pth")

    # Set seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    pyro.set_rng_seed(config.seed)

    # 3) Load full data
    loader = MultiIsotopeDataLoader(config)
    full_data = loader.load_all_data()

    # Split the data into available (for training/validation) and extrapolation (left-out nuclei)
    available_data = full_data[~full_data["N"].isin(n_out_list)]
    extrapolation_data = full_data[full_data["N"].isin(n_out_list)]

    print(f"Available data (training + validation): {len(available_data)}")
    print(f"Extrapolation (left-out nuclei) data: {len(extrapolation_data)}")

    # 4) Train/Val/Test split on available data only
    train_df, val_df, test_df = split_data_multi_isotope(available_data, config)
    print("Length of train, val, test:", len(train_df), len(val_df), len(test_df))

    # 5) Preprocess the training data
    train_datasets, scaler_X, scaler_y = preprocess_data_multi_isotope(train_df, config)

    # Preprocess the available-data validation set
    # (Note: here we keep the validation set separate from the left-out nuclei)
    val_datasets, _, _ = preprocess_data_multi_isotope(val_df, config)

    # 6) Fidelity levels: use those present in the full data so that all levels are covered.
    fidelity_levels = sorted(full_data[config.fidelity_col].unique())
    print("Fidelity levels:", fidelity_levels)

    # 7) Initialize model and trainer
    bnn = HierarchicalBNN(config, fidelity_levels).to(config.device)
    trainer = Trainer(bnn, config)
    histories = trainer.train(train_datasets, fidelity_levels, val_datasets)

    # 8) Load best model and set to evaluation mode
    best_model = HierarchicalBNN.load(
        config.model_save_path, config, fidelity_levels
    ).to(config.device)
    best_model.eval()
    trainer.model = best_model

    # 9) Plot the training history (loss and MAPE)
    plot_training_losses_history(
        histories, os.path.join(logs_dir, f"training_history_leaveout_{n_out_str}.png")
    )
    plot_mape_history(
        histories, os.path.join(logs_dir, f"training_mapes_leaveout_{n_out_str}.png")
    )

    ###############################
    # Predictions on the validation set (from available_data)
    ###############################
    print("Performing predictions on the available-data validation set...")
    # We will loop over unique nucleus IDs in the validation set so that the N value is correctly passed
    val_predictions = pd.DataFrame()
    for nucleus in val_df["N"].unique():
        df_nuc = val_df[val_df["N"] == nucleus]
        for fidelity in df_nuc[config.fidelity_col].unique():
            subset = df_nuc[df_nuc[config.fidelity_col] == fidelity].copy()
            x_val = subset[config.input_cols].values
            # For predictions on available nuclei we pass the current nucleus number
            y_pred, y_unc = bannane_inference(
                x_unscaled=x_val,
                fidelity=fidelity,
                Z=8,
                N=nucleus,
                scaler_X=scaler_X,
                scaler_y=scaler_y,
                trainer=trainer,
                config=config,
                return_unc=True,
            )
            # Add predictions to the subset
            subset["eb_pred"] = y_pred[:, 0]
            subset["r_pred"] = y_pred[:, 1]
            subset["eb_unc"] = y_unc[:, 0]
            subset["r_unc"] = y_unc[:, 1]
            val_predictions = pd.concat([val_predictions, subset], ignore_index=True)

    val_csv = os.path.join(logs_dir, f"validation_leaveout_{n_out_str}.csv")
    val_predictions.to_csv(val_csv, index=False)
    print(f"Validation predictions saved to {val_csv}")

    ###############################
    # Predictions on the extrapolation set (left-out nuclei)
    ###############################
    print("Performing predictions on the extrapolation (left-out nuclei) data...")
    extr_predictions = pd.DataFrame()
    # Loop over each left-out nucleus (each N in n_out_list)
    for nucleus in sorted(extrapolation_data["N"].unique()):
        df_nuc = extrapolation_data[extrapolation_data["N"] == nucleus]
        for fidelity in df_nuc[config.fidelity_col].unique():
            subset = df_nuc[df_nuc[config.fidelity_col] == fidelity].copy()
            x_extr = subset[config.input_cols].values
            y_pred, y_unc = bannane_inference(
                x_unscaled=x_extr,
                fidelity=fidelity,
                Z=8,
                N=nucleus,  # here, N is the left-out nucleus number
                scaler_X=scaler_X,
                scaler_y=scaler_y,
                trainer=trainer,
                config=config,
                return_unc=True,
            )
            subset["eb_pred"] = y_pred[:, 0]
            subset["r_pred"] = y_pred[:, 1]
            subset["eb_unc"] = y_unc[:, 0]
            subset["r_unc"] = y_unc[:, 1]
            extr_predictions = pd.concat([extr_predictions, subset], ignore_index=True)

    extr_csv = os.path.join(logs_dir, f"extrapolation_leaveout_{n_out_str}.csv")
    extr_predictions.to_csv(extr_csv, index=False)
    print(f"Extrapolation predictions saved to {extr_csv}")

    print(f"Finished leave-out procedure for nuclei: {n_out_list}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_out_list",
        type=str,
        default=None,
        help="Comma-separated list of nuclei to leave out",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="/work/submit/josemm/WORKS/Theory/BANANNE/LABS/LEAVEOUT",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_embedding_dim", type=int, default=20)
    parser.add_argument("--embed_z", action="store_true")
    parser.add_argument("--shared_latent_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--fidelity_embedding_dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_iterations", type=int, default=30_000)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--perform_temperature_scaling", action="store_true")

    args = parser.parse_args()

    # 1) Create config
    config = Config()

    # 2) Update config from command line arguments
    config.seed = args.seed
    config.n_embedding_dim = args.n_embedding_dim
    config.embed_z = args.embed_z
    config.shared_latent_dim = args.shared_latent_dim
    config.hidden_dim = args.hidden_dim
    config.num_heads = args.num_heads
    config.num_iterations = args.num_iterations
    config.dropout = args.dropout
    config.fidelity_embedding_dim = args.fidelity_embedding_dim
    config.learning_rate = args.learning_rate
    config.test_size = args.test_size
    config.val_size = args.val_size
    config.perform_temperature_scaling = args.perform_temperature_scaling

    config.patience = 200
    config.lr_patience = 25
    config.lr_decay = 2
    config.loss_weights = {4: 1.0, 6: 1.5, 8: 2.0, 10: 2.5}
    config.perform_temperature_scaling = True
    if args.n_out_list is not None:
        try:
            n_out_list = [int(x) for x in args.n_out_list.split(",")]
        except Exception as e:
            raise ValueError(
                "Please provide a valid comma-separated list of integers for --n_out_list"
            ) from e

    # 4) Call the main training function
    train_leave_out(n_out_list, config, logs_dir=args.logs_dir)
