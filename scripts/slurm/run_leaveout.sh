#!/bin/bash
#SBATCH --job-name=bannane_leaveout
#SBATCH --output=leaveout_output_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G

source /work/submit/josemm/SOFTWARE/anaconda3/bin/activate
conda activate envBANANNE
echo "Activated conda environment"

COMBO=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" leaveout_combinations.txt)
echo "Processing leave–out combination: $COMBO"

python /work/submit/josemm/WORKS/Theory/BANANNE/bannane/scripts/trainers/train_leaveout.py --n_out_list $COMBO --logs_dir /work/submit/josemm/WORKS/Theory/BANANNE/LABS/LEAVEOUT

mkdir -p results/training
mkdir -p results/rmse_tables

mv *.png results/training/
mv *.csv results/rmse_tables/

echo "Finished processing leave–out combination: $COMBO"
