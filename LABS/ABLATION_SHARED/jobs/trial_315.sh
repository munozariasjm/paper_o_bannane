#!/bin/bash
#SBATCH --job-name=trial_315
#SBATCH --output=./logs/trial_315.out
#SBATCH --error=./logs/trial_315.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

source /work/submit/josemm/SOFTWARE/anaconda3/bin/activate
conda activate envBANANNE
echo "Activated conda environment"

# Run the training script with the current set of hyperparameters
python /work/submit/josemm/WORKS/Theory/BANANNE/bannane/scripts/trainers/train_all.py \
    --id 315 \
    --logs_dir /work/submit/josemm/WORKS/Theory/BANANNE/LABS/ABLATION_SHARED \
    --seed 315 \
    --shared_latent_dim 24 \
    --hidden_dim 128 \
    --n_embedding_dim 24 \
    --fidelity_embedding_dim 8 \
