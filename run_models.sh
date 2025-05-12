#!/bin/bash

#SBATCH --job-name=nb_run
#SBATCH --time=02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --output=smolvlm2_notebook_%j.out

module load cuda12.3/toolkit
module load cuDNN/cuda12.3

source ~/.bashrc
conda activate op_bench

cd /var/scratch/ave303/OP_bench

mkdir -p output_nb_run_$SLURM_JOB_ID
cd output_nb_run_$SLURM_JOB_ID

echo "Starting at $(date)"
SECONDS=0  # built-in bash timer

papermill /var/scratch/ave303/OP_bench/opa-benchmark-smolvlm2-qwen2-5-vl.ipynb smolvlm2_output.ipynb

echo "Finished at $(date)"
echo "Total execution time: $(($SECONDS / 60)) minutes and $(($SECONDS % 60)) seconds"
