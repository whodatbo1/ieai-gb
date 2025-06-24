#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

# Your job starts in the directory where you call sbatch
cd $HOME/ieai-gb/
# Activate your environment
source activate mechinterp
# Run your code
python experiments.py --experiment all