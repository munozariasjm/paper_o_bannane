#!/bin/bash
#SBATCH --job-name=bannane_refine
#SBATCH --output=refine_out_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=5,10,20,30,40,50,60,70,80,90,99
#SBATCH --mem=16G

source /work/submit/josemm/SOFTWARE/anaconda3/bin/activate
conda activate envBANANNE
echo "Activated conda environment: $(which python)"

FRACTION=$SLURM_ARRAY_TASK_ID

echo "Running with fraction=$FRACTION"

LOGS_DIR="results_refinement"

python /work/submit/josemm/WORKS/Theory/BANANNE/bannane/scripts/trainers/train_refinement.py \
    --fraction $FRACTION \
    --logs_dir $LOGS_DIR

mkdir -p $LOGS_DIR
mv *.png $LOGS_DIR/ 2>/dev/null
mv $LOGS_DIR/.refinement_test_predictions_frac*.csv $LOGS_DIR/ 2>/dev/null
mv $LOGS_DIR/.summary_refinement_frac*.txt $LOGS_DIR/ 2>/dev/null

echo "Finished processing fraction=$FRACTION"
