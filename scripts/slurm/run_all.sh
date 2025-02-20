#!/bin/bash
#SBATCH --job-name=bannane_inter_job
#SBATCH --output=output_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1

source /work/submit/josemm/SOFTWARE/anaconda3/bin/activate
conda activate envBANANNE
echo "Activated conda environment"

python /work/submit/josemm/WORKS/Theory/BANANNE/bannane/scripts/trainers/train_all.py

mkdir -p results/training
mkdir -p results/rmse_tables

mv *.png results/training/
mv *.csv results/rmse_tables/

echo "Finished processing"
