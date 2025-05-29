#!/bin/bash

#SBATCH --job-name=test_nb
#SBATCH --time=00:05:00        # Just 5 minutes
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=fatq
#SBATCH --gres=gpu:1
#SBATCH --output=test_notebook_%j.out

module load cuda12.3/toolkit/
module load cuDNN/cuda12.3

source ~/.bashrc
conda activate op_bench

cd /var/scratch/ave303/OP_bench

# Run a very simple quick test (e.g., import torch, check GPU, run one training step)
papermill analyse_results.ipynb analyse_results.ipynb
