#!/bin/sh

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=84G
#SBATCH --job-name=MWClassifierParallel
#SBATCH --mail-user=abirney@uvm.edu
#SBATCH --mail-type=ALL
#SBATCH --account=cs6540
#SBATCH --cpus-per-task=8
#SBATCH --output="/users/a/b/abirney/MW_Classifier/Output/cv1/fold_%A_%a.out"
#SBATCH --error="/users/a/b/abirney/MW_Classifier/Output/cv1/fold_%A_%a.err"
#SBATCH --array=0-14%5

source activate torchGPU3

cd /users/a/b/abirney/MW_Classifier

FOLD_NUMBER=$SLURM_ARRAY_TASK_ID
python -u train_LSTM_parallel_cv.py $FOLD_NUMBER