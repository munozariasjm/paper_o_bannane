#!/bin/bash
#SBATCH --job-name=trial_592
#SBATCH --output=./logs/trial_592.out
#SBATCH --error=./logs/trial_592.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

source /work/submit/josemm/SOFTWARE/anaconda3/bin/activate
conda activate envBANANNE
echo "Activated conda environment"

# Run the training script with the current set of hyperparameters
python /work/submit/josemm/WORKS/Theory/BANANNE/bannane/scripts/trainers/train_all.py \
    --id 592 \
    --logs_dir /work/submit/josemm/WORKS/Theory/BANANNE/LABS/ABLATION_SHARED \
    --seed 592 \
    --shared_latent_dim 96 \
    --hidden_dim 24 \
    --n_embedding_dim 32 \
    --fidelity_embedding_dim 16 \
