#!/bin/bash
#SBATCH --job-name=trial_1
#SBATCH --output=./logs/trial_1.out
#SBATCH --error=./logs/trial_1.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

source /work/submit/josemm/SOFTWARE/anaconda3/bin/activate
conda activate envBANANNE
echo "Activated conda environment"

# Run the training script with the current set of hyperparameters
python /work/submit/josemm/WORKS/Theory/BANANNE/bannane/scripts/trainers/train_all.py \
    --id 1 \
    --logs_dir /work/submit/josemm/WORKS/Theory/BANANNE/LABS/ABLATION_SHARED \
    --seed 1 \
    --shared_latent_dim 8 \
    --hidden_dim 24 \
    --n_embedding_dim 16 \
    --fidelity_embedding_dim 2 \
