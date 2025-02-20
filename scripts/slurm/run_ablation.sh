#!/bin/bash
#SBATCH --job-name=ablation_study
#SBATCH --output=ablation_%A_%a.out
#SBATCH --error=ablation_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-27

# Activate Conda environment
source /work/submit/josemm/SOFTWARE/anaconda3/bin/activate
conda activate envBANANNE
echo "Activated conda environment"

# Define variables
shared_latent_dim=$((20 + SLURM_ARRAY_TASK_ID * 4))  # Remove space after '='
echo "SLURM_ARRAY_TASK_ID is: $SLURM_ARRAY_TASK_ID"
echo "shared_latent_dim is: $shared_latent_dim"

# Define saving path (modify as needed)
saving_path= "/work/submit/josemm/WORKS/Theory/BANANNE/LABS/ablation_study/shared_latent_dim"

echo "Saving path is: $saving_path"

# Run Python script
python /work/submit/josemm/WORKS/Theory/BANANNE/bannane/scripts/trainers/train_all.py \
  --trial_id ${SLURM_ARRAY_TASK_ID} \
  --saving_path ${saving_path} \
  --shared_latent_dim ${shared_latent_dim}
