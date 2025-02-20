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
import torch
import os
import numpy as np
import argparse

import warnings

warnings.filterwarnings("ignore")

# %%
import sys

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


# %%
from dataclasses import dataclass
from typing import List


config = Config()

config.loss_weights = {4: 1.0, 6: 1.5, 8: 2.0, 10: 2.5}
config.perform_temperature_scaling = True
config.fidelity_embedding_dim = 6

def train_emax(n_out: int, config: Config = config, logs_dir: str = ""):
    config.data_directory = "/work/submit/josemm/WORKS/Theory/BANANNE/bannane/DATA/all_o"
    config.model_save_path = logs_dir + f"/model_{n_out}.pth"
    config.temperature_save_path = logs_dir + f"/temperature_{n_out}.pth"
    config.scaler_X_path = logs_dir + f"/scaler_X_{n_out}.pth"
    config.scaler_y_path = logs_dir + f"/scaler_y_{n_out}.pth"
    # 2) Seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    pyro.set_rng_seed(config.seed)
    # 3) Load data
    loader = MultiIsotopeDataLoader(config)
    full_data = loader.load_all_data()

    #
    available_data = full_data[(full_data["N"] != n_out)]
    data_for_n = full_data[(full_data["N"] == n_out)]

    low_fid_data = data_for_n[data_for_n["emax"] == 4]
    available_data = pd.concat([available_data, low_fid_data]).reset_index(drop=True)

    extrapolation_data = data_for_n[data_for_n["emax"] != 4]

    print(f"Available data: {len(available_data)}")
    print(f"Extrapolation data: {len(extrapolation_data)}")

    # 4) Train/Val/Test split
    train_df, val_df, test_df = split_data_multi_isotope(available_data, config)
    print("Length of train, val, test:", len(train_df), len(val_df), len(test_df))

    # 5) Preprocess
    train_datasets, scaler_X, scaler_y = preprocess_data_multi_isotope(train_df, config)

    # Use the LOO as validation so concatenate
    val_df = pd.concat([val_df, extrapolation_data.sample(frac=0.3)]).reset_index(drop=True)
    val_datasets, _, _ = preprocess_data_multi_isotope(val_df, config)

    # 6) Fidelity levels
    fidelity_levels = sorted(full_data[config.fidelity_col].unique())
    print("Fidelity levels:", fidelity_levels)

    # 7) Initialize model
    bnn = HierarchicalBNN(config, fidelity_levels).to(config.device)

    # 8) Train
    trainer = Trainer(bnn, config)
    histories = trainer.train(train_datasets, fidelity_levels, val_datasets)

    # 9) Load best model
    best_model = HierarchicalBNN.load(
        config.model_save_path, config, fidelity_levels
    ).to(config.device)
    best_model.eval()
    trainer.model = best_model


    # %%
    # Plot the training history with rolling window average
    plot_training_losses_history(histories, logs_dir + f"/training_history_loo_{n_out}.png")
    plot_mape_history(histories, logs_dir + f"/training_mapes_loo_{n_out}.png")

    validation_data = extrapolation_data

    grouped = validation_data.groupby(["Z", "N"])

    results = []

    n_bootstrap = 1000

    for (Z, N), group in grouped:
        A = Z + N  # Mass number
        isotope = f"$^{{{A}}}$O"  # Isotope notation for labeling
        for emax in group["emax"].unique():
            group = group[group["emax"] == emax]
            x = group[config.input_cols].values
            y_true = group[
                ["Energy ket", "Rch"]
            ].values  # Assuming these are the target columns

            if len(x) == 0:
                continue  # Skip if no data for this isotope

            y_pred, y_unc = bannane_inference(
                x_unscaled=x,
                fidelity=emax,
                Z=Z,
                N=N,
                scaler_X=scaler_X,
                scaler_y=scaler_y,
                trainer=trainer,
                config=config,
                return_unc=True,
            )

            # Calculate residuals
            residuals = y_pred - y_true

            # Compute RMSE for Energy ket (E_b) and Charge Radii (R_ch)
            rmse_Eb = np.sqrt(np.mean(residuals[:, 0] ** 2))
            rmse_Rch = np.sqrt(np.mean(residuals[:, 1] ** 2))

            # Bootstrap to estimate uncertainty
            bootstrap_rmse_Eb = []
            bootstrap_rmse_Rch = []
            for _ in range(n_bootstrap):
                indices = np.random.randint(0, len(residuals), len(residuals))
                sample_residuals = residuals[indices]
                bootstrap_rmse_Eb.append(np.sqrt(np.mean(sample_residuals[:, 0] ** 2)))
                bootstrap_rmse_Rch.append(np.sqrt(np.mean(sample_residuals[:, 1] ** 2)))

            # Calculate standard deviation of bootstrap RMSEs as uncertainty
            std_rmse_Eb = np.std(bootstrap_rmse_Eb)
            std_rmse_Rch = np.std(bootstrap_rmse_Rch)

            # Append the results to the list
            results.append(
                {
                    "Isotope": isotope,
                    "N": N,
                    "RMSE_Eb (MeV)": rmse_Eb,
                    "Uncertainty_RMSE_Eb (MeV)": std_rmse_Eb,
                    "RMSE_Rch (fm)": rmse_Rch,
                    "Uncertainty_RMSE_Rch (fm)": std_rmse_Rch,
                }
            )

    # Step 4: Create a DataFrame from the results
    rmse_table = pd.DataFrame(results)

    # Optional: Sort the table by Neutron number N for better readability
    rmse_table = rmse_table.sort_values(by="N").reset_index(drop=True)

    # Display the table
    rmse_table
    # Save the table to a CSV file
    rmse_table.to_csv(f"availabe_rmse_table_eemax_{n_out}.csv", index=False)

    ###### For the extrapolation data ######

    N_extrapolation = n_out
    out_df_pred = pd.DataFrame()
    for extr_fidelity in extrapolation_data.emax.unique():
        extr_data = extrapolation_data[
            extrapolation_data["emax"] == extr_fidelity
        ].copy()

        print(
            "Extrapolation data emax:",
            extr_fidelity,
            "Number of data points:",
            len(extr_data),
        )

        # Ensure there is data for the fidelity
        if validation_data.empty:
            print(f"No data for fidelity {extr_fidelity}")
            continue

        x = extr_data[config.input_cols].values
        ys = extr_data[config.target_cols].values

        y_pred, y_unc = bannane_inference(
            x_unscaled=x,
            fidelity=extr_fidelity,
            Z=8,
            N=N_extrapolation,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            trainer=trainer,
            config=config,
            return_unc=True,
        )

        extr_data["eb_pred"] = y_pred[:, 0]
        extr_data["r_pred"] = y_pred[:, 1]
        extr_data["eb_unc"] = y_unc[:, 0]
        extr_data["r_unc"] = y_unc[:, 1]

        out_df_pred = pd.concat([out_df_pred, extr_data])

    out_df_pred.to_csv(f"extrapolation_eemax_{n_out}.csv", index=False)

    print(f"Finished LOO for AE n={n_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_out", type=int, default=100)
    parser.add_argument("--logs_dir", type=str,
                        default="/work/submit/josemm/WORKS/Theory/BANANNE/LABS/TRAIN_ALL")
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
    parser.add_argument("--test_size", type=float, default=0.0001)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--perform_temperature_scaling", action="store_true")

    args = parser.parse_args()

    # 1) Create config
    config = Config()

    # 2) Update from command line
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

    # 3) Call main function
    train_emax(args.n_out, config, logs_dir=args.logs_dir)
