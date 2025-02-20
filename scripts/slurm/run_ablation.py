#!/usr/bin/env python3
import os
import subprocess
import itertools
import numpy as np

# Define the grid search parameters
shared_latent_dims = [int(x) for x in np.arange(24, 129, 8)]
# Path to the python script that runs your training (the code you posted)
training_script = "/work/submit/josemm/WORKS/Theory/BANANNE/bannane/scripts/trainers/train_all.py"
# Base saving path (update as needed)
saving_path = "/work/submit/josemm/WORKS/Theory/BANANNE/LABS/ablation"

# Directory to store the sbatch job files and logs (make sure they exist)
jobs_dir = "./jobs"
logs_dir = "./logs"
os.makedirs(jobs_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

trial_id = 0
# Loop over all combinations using itertools.product
for shared_latent in shared_latent_dims:
    trial_id += 1

    # Create a unique job name and file name
    job_name = f"trial_{trial_id}"
    job_file = os.path.join(jobs_dir, f"{job_name}.sh")

    # Write out the sbatch script; adjust SLURM header options as needed
    script_contents = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/{job_name}.out
#SBATCH --error={logs_dir}/{job_name}.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

source /work/submit/josemm/SOFTWARE/anaconda3/bin/activate
conda activate envBANANNE
echo "Activated conda environment"

# Run the training script with the current set of hyperparameters
python {training_script} \\
    --trial_id {trial_id} \\
    --saving_path {saving_path} \\
    --shared_latent_dim {shared_latent} \\
"""

    # Write the script to file
    with open(job_file, "w") as f:
        f.write(script_contents)

    # Submit the job using sbatch
    print(
        f"Submitting {job_name} with parameters:  shared_latent_dim={shared_latent}"
    )
    subprocess.run(["sbatch", job_file])
