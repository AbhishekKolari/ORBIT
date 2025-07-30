#!/bin/bash

# ====== SLURM SETTINGS ======
#SBATCH --job-name=vlm_bench
#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH -C A6000
#SBATCH --gres=gpu:1
#SBATCH --output=vlm_bench_%j.out

# ====== USER-CONFIGURABLE VARIABLES ======
# Set your conda environment name
CONDA_ENV="op_bench"  # Change if needed
# Path to your Python script
PYTHON_SCRIPT="opa_benchmark_qwen2_5_vl.py"  # Change if needed
# Output directory for this run
OUTPUT_DIR="output_vlm_bench_$SLURM_JOB_ID"

# Required arguments for the Python script
MODEL_PATH="$1"         # e.g. /path/to/model
PROCESSOR_PATH="$2"     # e.g. /path/to/model
BENCHMARK_JSON="$3"     # e.g. /path/to/benchmark.json
DATA_DIR="$4"           # e.g. /path/to/images
OUTPUT_FILE="$5"        # e.g. results.json
BATCH_SIZE="$6"         # e.g. 5 (optional)

if [ $# -lt 5 ]; then
  echo "Usage: sbatch run_qwen_32b.sh <MODEL_PATH> <PROCESSOR_PATH> <BENCHMARK_JSON> <DATA_DIR> <OUTPUT_FILE> [BATCH_SIZE]"
  exit 1
fi

module load cuda12.3/toolkit
module load cuDNN/cuda12.3

source ~/.bashrc
conda activate "$CONDA_ENV"

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "Starting at $(date)"
SECONDS=0

python ../"$PYTHON_SCRIPT" \
  --model_path "$MODEL_PATH" \
  --processor_path "$PROCESSOR_PATH" \
  --benchmark_json "$BENCHMARK_JSON" \
  --data_dir "$DATA_DIR" \
  --output_file "$OUTPUT_FILE" \
  ${BATCH_SIZE:+--batch_size "$BATCH_SIZE"} \
  > vlm_bench_output.log 2>&1

echo "Finished at $(date)"
echo "Total execution time: $(($SECONDS / 3600)) hours $((($SECONDS % 3600) / 60)) minutes and $(($SECONDS % 60)) seconds" 