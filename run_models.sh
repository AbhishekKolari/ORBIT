#!/bin/bash

#SBATCH --job-name=nb_run
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH -C A6000
#SBATCH --gres=gpu:1
#SBATCH --output=qwen2-5-vl-32b_notebook_%j.out

module load cuda12.3/toolkit
module load cuDNN/cuda12.3

source ~/.bashrc
conda activate op_bench

cd /var/scratch/ave303/OP_bench

mkdir -p output_nb_run_$SLURM_JOB_ID
cd output_nb_run_$SLURM_JOB_ID

echo "Starting at $(date)"
SECONDS=0  # built-in bash timer

papermill /var/scratch/ave303/OP_bench/opa-benchmark-smolvlm2-qwen2-5-vl.ipynb qwen2-5-vl-32b_output.ipynb

echo "Finished at $(date)"
echo "Total execution time: $(($SECONDS / 3600)) hours $(($SECONDS / 60)) minutes and $(($SECONDS % 60)) seconds"
