#!/bin/bash
#SBATCH --job-name=trial_475
#SBATCH --output=./logs/trial_475.out
#SBATCH --error=./logs/trial_475.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

source /work/submit/josemm/SOFTWARE/anaconda3/bin/activate
conda activate envBANANNE
echo "Activated conda environment"

# Run the training script with the current set of hyperparameters
python /work/submit/josemm/WORKS/Theory/BANANNE/bannane/scripts/trainers/train_all.py \
    --id 475 \
    --logs_dir /work/submit/josemm/WORKS/Theory/BANANNE/LABS/ABLATION_SHARED \
    --seed 475 \
    --shared_latent_dim 64 \
    --hidden_dim 32 \
    --n_embedding_dim 24 \
    --fidelity_embedding_dim 8 \
