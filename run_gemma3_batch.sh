#!/bin/bash

#SBATCH --job-name=gemma3_batch
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH -C A6000
#SBATCH --gres=gpu:1
#SBATCH --output=gemma3_batch_%j.out

module load cuda12.3/toolkit
module load cuDNN/cuda12.3

source ~/.bashrc
conda activate op_bench

cd /var/scratch/ave303/OP_bench

mkdir -p output_gemma3_$SLURM_JOB_ID
cd output_gemma3_$SLURM_JOB_ID

echo "Starting at $(date)"
SECONDS=0

python ../run_gemma3_batch.py

echo "Finished at $(date)"
echo "Total execution time: $(($SECONDS / 3600)) hours $((($SECONDS % 3600) / 60)) minutes and $(($SECONDS % 60)) seconds" 