<h1 align="center"> ORBIT: An <ins>O</ins>bject Property <ins>R</ins>easoning <ins>B</ins>enchmark for Visual <ins>I</ins>nference <ins>T</ins>asks</h1>

A novel comprehensive benchmark, ORBIT, evaluates Vision-Language Models (VLMs) abilities to reason about abstract object properties across four object property dimensions, 
three reasoning complexity levels and three visual domains.

<h1 align="center"><img width="584" height="400" alt="orbit-tax-500-1" src="https://github.com/user-attachments/assets/52bd4e19-ca8f-45ab-aa44-0992726c3897" /></h1>


## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AbhishekKolari/ORBIT.git
   cd ORBIT
   ```

2. **Install the Framework**:
   
   Install Anaconda or Miniconda distribution based on Python3+ from their downloads' site once.
   ```bash
   # create the environment
   conda create -n [env_name] python=3.12

   # activate
   conda activate [env_name]

   # install dependencies
   pip install -r requirements.txt
   ```

## Usage

ORBIT provides scripts to reproduce and evaluate multiple open-source and closed-source models on the benchmark:

- **`main.py`** - Single CLI entrypoint that dispatches to the open-source evaluator (`opensource.py`) or closed-source runner (`closedsource.py`).
- **`utils.py`** - Contains main logic behind evaluating the models in `evaluate_models()` under the class `BenchmarkTester`, handling absolute/relative path confusions in `resolve_image_path()`, and metric helpers (accuracy, off-by-n, RMSE, mean error) in `compute_metric_from_results()`.
- **`opensource.py`** - Loads and evaluates open-source HuggingFace (HF) models. Contains hardocded models (different size varaints) used for evaluation as in the paper (BLIP2, Qwen2.5-VL, InternVL3, and Gemma 3) and falls back to generic HF loading for arbitrary repo IDs. Evaluation of custom or other open-source models can be included in this script. Gemma 3 is a gated model on Hf and would require a HF token to access. Accepts only full HF paths (`opengvlab/internvl3-8b`) or full custom paths if models downloaded to local machine.
- **`closedsource.py`** - Routes multimodal requests to closed-source providers (OpenAI GPT-4o, Anthropic Claude, Google Gemini). Accepts either short aliases (`gpt`, `claude`, `gemini`) or full provider model ids (`gpt-4o-mini-2024-07-18`, `claude-3.7-sonnet`, `gemini-2.0-pro`). Reads API keys from env vars.
- **`question_generator.py`** - Generate MLLM-based question triples either on existing images from `data/` or on a new set of images (organized similar to `data/`) and write them into a benchmark-style JSON (same top-level structure as ORBIT's `benchmark.json`). Uses closed-source providers (OpenAI GPT-4o, Anthropic Claude, Google Gemini). Accepts short aliases (`gpt`, `claude`, `gemini`). Supports a `--test_mode` for quick checks and `--max_images` in test mode default at 3 images.

## Environment Variables

Create .env file

## Execution

   ### Evaluate models on benchmark

   1. **Open-source models**:
      CLI 

   2. **Closed-source models**:
      CLI 

   ### ORBIT MLLMs based question generation

   CLI

## Disclaimer


<!-- ## Project Structure

- **benchmark.json**: The main benchmark dataset containing annotated questions and ground truth answers for real, animated, and AI-generated images.
- **merged_data/**: Contains subfolders for different image types (`REAL/`, `ANIMATED/`, `AI_GENERATED/`) used in the benchmark.
- **download_models.py**: Script to download and cache all required vision-language models from HuggingFace or other sources. This ensures reproducibility and consistent model versions across experiments.
- **shell_scripts/run_models.sh**: Example shell script to run model inference in batch mode.
- **ORBIT_results/**: Store the output JSON files from model runs. Each file contains the model's answers and reasoning for all benchmark questions.
- **analyse_results.ipynb**: The main analysis notebook. Loads model outputs, computes evaluation metrics (accuracy, off-by-N, MAE, RMSE), and generates plots for thesis figures. This notebook is central to the quantitative and qualitative analysis in the thesis.
- **ORBIT_analysis_plots/** and **ORBIT_model_plots/**: Contain figures generated from the analysis notebook, including accuracy curves, error distributions, and model comparison plots. These are directly used in the thesis to illustrate findings.
- **ORBIT_notebooks/opa-benchmark-<model-names>.ipynb**: Contains wrappers and utility functions for running open-source models on the benchmark.
<!-- - **pdf2bench.py**: Utility for converting PDF-based datasets into the benchmark format. -->
<!-- - **create_notebook.py**: Script to auto-generate Jupyter notebooks for new experiments or model evaluations. --> -->

<!-- ## How This Supports the Thesis Experiments

1. **Benchmark Construction**: The `benchmark.json` and `merged_data/` directories define the experimental setup, ensuring a diverse and challenging set of counting and reasoning tasks.
2. **Model Evaluation**: `download_models.py` and `opa-benchmark-<model-names>.ipynb` allow for systematic downloading, setup, and inference with a wide range of vision-language models, as required for the thesis comparison.
3. **Result Storage**: All model outputs are saved in a standardized format in `ORBIT_results/`, enabling fair and reproducible evaluation.
4. **Analysis & Visualization**: `analyse_results.ipynb` computes all key metrics reported in the thesis (accuracy, off-by-N, MAE, RMSE, error clustering, etc.) and produces publication-ready plots found in `ORBIT_analysis_plots/` and `ORBIT_model_plots/`.
5. **Reproducibility**: Scripts and notebooks are organized to allow any researcher to reproduce the thesis experiments from model download to final analysis. -->

<!-- ## Getting Started

1. **Setup and Dependencies**:  
   Install Anaconda or Miniconda distribution based on Python3+ from their downloads' site.
   ```bash 
   conda create -n [env_name] python=3.12
   ```
   Activate it and install all necessary libraries:   -->
   <!-- ```bash 
   pip install -r requirements.txt
   ```
   Create ipykernel for the use of Jupyter Notebooks:
   ```bash
   python -m ipykernel install --user --name [env_name] --display-name "[any_name]"
   ```

2. **Download models**:
   ```bash
   python download_models.py
   ```

3. **Tweak model parameters and dataset batches** in `opa-benchmark-<model-names>.ipynb`

4. **Run inference via SLURM** (change file paths accordingly):
   ```bash
   sbatch run_models.sh
   ```

5. **Analyze results**:  
   Tweak `analyse_results.ipynb` and run  
   ```bash
   sbatch analyse.sh
   ``` -->

 
