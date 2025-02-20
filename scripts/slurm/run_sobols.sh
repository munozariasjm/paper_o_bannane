#!/bin/bash
#SBATCH --job-name=bannane_inter_job
#SBATCH --output=output_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=4-16
#SBATCH --mem=64G

# Activate virtual environment (modify based on your setup)
source /work/submit/josemm/SOFTWARE/anaconda3/bin/activate
conda activate envBANANNE
echo "Activated conda environment"

echo "SLURM_ARRAY_TASK_ID is: $SLURM_ARRAY_TASK_ID"

# Run the Python script with the current SLURM_ARRAY_TASK_ID as n_out
python /work/submit/josemm/WORKS/Theory/BANANNE/bannane/scripts/trainers/train_sobol.py --N $SLURM_ARRAY_TASK_ID

echo "Finished processing n_out=$SLURM_ARRAY_TASK_ID"
