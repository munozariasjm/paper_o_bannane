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

sys.path.append("../../../BANANNE/")
from bannane.src.config import Config
from bannane.src.data_loader import MultiIsotopeDataLoader
from bannane.src.preprocess import (
    split_data_multi_isotope,
    preprocess_data_multi_isotope,
)
from bannane.src.model import HierarchicalBNN
from bannane.src.trainer import Trainer, bannane_inference
from bannane.src.plotting import plot_training_losses_history, plot_mape_history
from dataclasses import dataclass
from typing import List


def train_all(id, config, logs_dir: str = ""):
    config.data_directory = "../../../BANANNE/bannane/DATA/all_o"
    config.model_save_path = logs_dir + f"/model_{id}.pth"
    config.temperature_save_path = logs_dir + f"/temperature_{id}.pth"
    config.scaler_X_path = logs_dir + f"/scaler_X_{id}.pth"
    config.scaler_y_path = logs_dir + f"/scaler_y_{id}.pth"
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    pyro.set_rng_seed(config.seed)

    loader = MultiIsotopeDataLoader(config)
    full_data = loader.load_all_data()

    train_df, val_df, test_df = split_data_multi_isotope(full_data, config)
    train_datasets, scaler_X, scaler_y = preprocess_data_multi_isotope(train_df, config)

    val_datasets, _, _ = preprocess_data_multi_isotope(val_df, config)

    fidelity_levels = sorted(full_data[config.fidelity_col].unique())
    print("Fidelity levels:", fidelity_levels)

    bnn = HierarchicalBNN(config, fidelity_levels).to(config.device)

    trainer = Trainer(bnn, config)
    histories = trainer.train(train_datasets, fidelity_levels, val_datasets)

    best_model = HierarchicalBNN.load(
        config.model_save_path, config, fidelity_levels
    ).to(config.device)
    best_model.eval()
    trainer.model = best_model

    plot_training_losses_history(
        histories, logs_dir + f"/training_history_all_{id}.png"
    )
    plot_mape_history(histories, logs_dir + f"/training_mapes_all_{id}.png")

    validation_data = val_df

    grouped = validation_data.groupby(["Z", "N"])

    results = []

    n_bootstrap = 1000

    for (Z, N), group in grouped:
        A = Z + N  # Mass number
        isotope = f"$^{{{A}}}$O"  # Isotope notation for labeling

        x = group[config.input_cols].values
        y_true = group[
            ["Energy ket", "Rch"]
        ].values  # Assuming these are the target columns

        if len(x) == 0:
            continue  # Skip if no data for this isotope

        y_pred, y_unc = bannane_inference(
            x_unscaled=x,
            fidelity=8,
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

        mape_Eb = np.mean(np.abs(residuals[:, 0] / y_true[:, 0])) * 100
        mape_Rch = np.mean(np.abs(residuals[:, 1] / y_true[:, 1])) * 100

        # Bootstrap to estimate uncertainty
        bootstrap_rmse_Eb = []
        bootstrap_rmse_Rch = []
        bootstrap_mape_Eb = []
        bootstrap_mape_Rch = []
        for _ in range(n_bootstrap):
            indices = np.random.randint(0, len(residuals), len(residuals))
            sample_residuals = residuals[indices]
            bootstrap_rmse_Eb.append(np.sqrt(np.mean(sample_residuals[:, 0] ** 2)))
            bootstrap_rmse_Rch.append(np.sqrt(np.mean(sample_residuals[:, 1] ** 2)))
            bootstrap_mape_Eb.append(
                np.mean(np.abs(sample_residuals[:, 0] / y_true[indices, 0])) * 100
            )
            bootstrap_mape_Rch.append(
                np.mean(np.abs(sample_residuals[:, 1] / y_true[indices, 1])) * 100
            )

        # Calculate standard deviation of bootstrap RMSEs as uncertainty
        std_rmse_Eb = np.std(bootstrap_rmse_Eb)
        std_rmse_Rch = np.std(bootstrap_rmse_Rch)
        std_mape_Eb = np.std(bootstrap_mape_Eb)
        std_mape_Rch = np.std(bootstrap_mape_Rch)

        # Append the results to the list
        results.append(
            {
                "Isotope": isotope,
                "N": N,
                "RMSE_Eb (MeV)": rmse_Eb,
                "Uncertainty_RMSE_Eb (MeV)": std_rmse_Eb,
                "RMSE_Rch (fm)": rmse_Rch,
                "Uncertainty_RMSE_Rch (fm)": std_rmse_Rch,
                "MAPE_Eb (%)": mape_Eb,
                "Uncertainty_MAPE_Eb (%)": std_mape_Eb,
                "MAPE_Rch (%)": mape_Rch,
                "Uncertainty_MAPE_Rch (%)": std_mape_Rch,
            }
        )

    # Step 4: Create a DataFrame from the results
    rmse_table = pd.DataFrame(results)

    # Optional: Sort the table by Neutron number N for better readability
    rmse_table = rmse_table.sort_values(by="N").reset_index(drop=True)

    # Display the table
    print(rmse_table)
    # Save the table to a CSV file
    rmse_table.to_csv(f"availabe_rmse_table_all_{id}.csv", index=False)

    ###### For the extrapolation data ######

    out_df_pred = pd.DataFrame()
    for extr_fidelity in validation_data.emax.unique():
        for n in validation_data.N.unique():
            extr_data = validation_data[
                (validation_data["emax"] == extr_fidelity) & (validation_data["N"] == n)
            ].copy()

            print(
                "Extrapolation data emax:",
                extr_fidelity,
                "Number of data points:",
                len(extr_data),
            )

            if len(extr_data) == 0:
                continue

            x = extr_data[config.input_cols].values
            ys = extr_data[config.target_cols].values

            y_pred, y_unc = bannane_inference(
                x_unscaled=x,
                fidelity=extr_fidelity,
                Z=8,
                N=n,
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

    out_df_pred.to_csv(f"valid_data_all_{id}.csv", index=False)

    print(f"Finished LOO for AE n={id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=100)
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="/work/submit/josemm/WORKS/Theory/BANANNE/LABS/TRAIN_ALL",
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

    # 2) Update from command line
    idd = args.id
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
    train_all(idd, config, logs_dir=args.logs_dir)
