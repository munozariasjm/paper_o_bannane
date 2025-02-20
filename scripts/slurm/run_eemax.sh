#!/bin/bash
#SBATCH --job-name=eemax_job
#SBATCH --output=output_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=4-16

source /work/submit/josemm/SOFTWARE/anaconda3/bin/activate
conda activate envBANANNE
echo "Activated conda environment"

echo "SLURM_ARRAY_TASK_ID is: $SLURM_ARRAY_TASK_ID"

python /work/submit/josemm/WORKS/Theory/BANANNE/bannane/scripts/trainers/train_eemax.py --n_out $SLURM_ARRAY_TASK_ID

mkdir -p results/training
mkdir -p results/rmse_tables

mv *.png results/training/
mv *.csv results/rmse_tables/

echo "Finished processing n_out=$SLURM_ARRAY_TASK_ID"
