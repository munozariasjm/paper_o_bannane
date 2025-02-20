# %%
import os
import sys
import numpy as np
import pandas as pd
import torch
import pyro
import warnings
from typing import List, Tuple
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


sys.path.append("/work/submit/josemm/WORKS/Theory/BANANNE/")
warnings.filterwarnings("ignore")

from bannane.src.config import Config
from bannane.src.data_loader import MultiIsotopeDataLoader
from bannane.src.preprocess import split_data_multi_isotope, preprocess_data_multi_isotope
from bannane.src.model import HierarchicalBNN
from bannane.src.trainer import Trainer, bannane_inference
from SALib.sample import saltelli
from SALib.analyze import sobol
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def train(n, config, logs_dir = '/work/submit/josemm/WORKS/Theory/BANANNE/_results/sobols/objects'):
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    config = Config()

    config.data_directory = '/work/submit/josemm/WORKS/Theory/BANANNE/bannane/DATA/all_o'
    config.file_pattern = "*_radii.csv"

    config.model_save_path = logs_dir + f'/model_sobol{n}.pth'
    config.temperature_save_path = logs_dir + f'/temperature_sobol{n}.pth'
    config.scaler_X_path = logs_dir + f'/scaler_X_sobol{n}.pth'
    config.scaler_y_path = logs_dir + f'/scaler_y_sobol{n}.pth'

    config.patience = 150
    config.lr_patience = 25
    config.lr_decay = 2
    config.shared_latent_dim = 128
    config.hidden_dim = 64

    config.num_heads = 2
    config.fidelity_embedding_dim = 8

    config.num_iterations = 30_000
    config.dropout = 0.1
    config.learning_rate = 5e-5

    config.loss_weights = {4: 1.0, 6: 1.5, 8: 2.0, 10: 2.5}
    config.perform_temperature_scaling = False


    # %%
    # 1) Initialize configuration and seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    pyro.set_rng_seed(config.seed)

    # 2) Load the full dataset
    loader = MultiIsotopeDataLoader(config)
    full_data = loader.load_all_data()

    # 3) Split into training, validation, and test sets
    train_df, val_df, test_df = split_data_multi_isotope(full_data, config)

    # 4) Preprocess data for each split
    train_datasets, scaler_X, scaler_y = preprocess_data_multi_isotope(train_df, config)
    val_datasets, _, _ = preprocess_data_multi_isotope(val_df, config)
    test_datasets, _, _ = preprocess_data_multi_isotope(test_df, config)

    # 5) Determine fidelity levels
    fidelity_levels = sorted(full_data[config.fidelity_col].unique())
    print(f"[Main] Fidelity levels: {fidelity_levels}")

    # 6) Initialize model and trainer
    bnn_model = HierarchicalBNN(config, fidelity_levels).to(config.device)
    trainer = Trainer(bnn_model, config)

    # 7) Train the model
    histories = trainer.train(train_datasets, fidelity_levels, val_datasets)

    # 8) Load the best model from disk
    best_model = HierarchicalBNN.load(config.model_save_path, config, fidelity_levels, device=config.device).to(config.device)

    best_model.eval()
    trainer.model = best_model
    return trainer, scaler_X, scaler_y, config


parameter_limits = [
    [-0.3588133858416398, -0.3221123390359522],
    [-0.3550367309847966, -0.3230356738391033],
    [-0.3531790737162967, -0.3198873235396882],
    [-0.2705565764558439, -0.2096640980382431],
    [2.197708705687826, 2.7940609681638446],
    [0.3561319578458198, 1.459562171959771],
    [-0.1301180174812798, 0.1989705404930199],
    [-1.3255957701239036, -0.866215494729709],
    [0.4940537296404043, 1.048656993966592],
    [0.2152844384736036, 0.6330251628592818],
    [-1.0001830037246255, -0.6905557349041349],
    [-0.7584507373606426, -0.6980546574315976],
    [-0.7109383542863426, -0.1718457990012224],
    [-1.0227287042294668, -0.4009843085185863],
    [0.8321833569145188, 1.213730054070106],
    [-1.874049115315852, 1.875273497658268],
    [-0.4485129790034055, 0.7133368422147623],
]
import matplotlib.pyplot as plt


# Function to perform Sobol analysis for a given isotope (N)
def perform_sobol_analysis( N, num_samples, config, trainer, scaler_X, scaler_y):
    """
    Perform Sobol sensitivity analysis for a given neutron number N.

    Args:
        N (int): Neutron number for the isotope.
        num_samples (int): Base sample size. Total samples will be (2 * num_samples * (num_vars + 2)).

    Returns:
        dict: Sobol sensitivity indices for E_b and R_ch.
    """

    # Define the Sobol problem
    problem = {
        'num_vars': len(config.input_cols),
        'names': config.input_cols,
        'bounds': parameter_limits
    }
    # Generate samples using Saltelli's sampler
    print(f"Generating Sobol samples for N={N}...")
    param_values = saltelli.sample(problem, num_samples)

    # Due to computational constraints, you might need to reduce num_samples
    # For example, use num_samples=2**13 for faster execution

    # Evaluate the model on the samples
    print(f"Evaluating the model for N={N}...")
    # Split the parameters into input features
    x_samples = param_values[:, :]  # All columns correspond to config.input_cols

    # Perform inference
    y_pred, y_unc = bannane_inference(
        x_samples,
        fidelity=8,
        Z=8,
        N=N,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        trainer=trainer,
        config=config,
        return_unc=True
    )

    # Extract E_b and R_ch predictions
    E_b = y_pred[:, 0]
    R_ch = y_pred[:, 1]

    # Perform Sobol analysis for E_b
    print(f"Performing Sobol analysis for E_b (N={N})...")
    Si_Eb = sobol.analyze(problem, E_b, print_to_console=False)

    # Perform Sobol analysis for R_ch
    print(f"Performing Sobol analysis for R_ch (N={N})...")
    Si_Rch = sobol.analyze(problem, R_ch, print_to_console=False)

    return {'E_b': Si_Eb, 'R_ch': Si_Rch}

# %%
def compute_sobols(N, sobol_output_path, two_exp, config, trainer, scaler_X, scaler_y):
    os.makedirs(sobol_output_path, exist_ok=True)
    sobol_output_file = os.path.join(sobol_output_path, f"sobol_results_N{N}.npy")
    sobol_results = perform_sobol_analysis(N, num_samples=2**two_exp, config=config, trainer=trainer, scaler_X=scaler_X, scaler_y=scaler_y)
    np.save(sobol_output_file, sobol_results)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=8, help="Neutron number for the isotope.")

    args = parser.parse_args()
    n = args.N
    two_exp = 16
    config = Config()
    print("Started training")
    trainer, scaler_X, scaler_y, config = train(n, config)
    sobol_output_path = "/work/submit/josemm/WORKS/Theory/BANANNE/_results/sobols"
    print("Started computing Sobols")
    compute_sobols(n, sobol_output_path, two_exp, config, trainer, scaler_X, scaler_y)
    print("Finished computing Sobols, saved to", sobol_output_path)
