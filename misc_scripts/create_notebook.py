import json

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# VLM Benchmark for Object Property Abstraction\n",
                "\n",
                "This notebook implements a benchmark for evaluating Vision Language Models (VLMs) on object property abstraction and visual question answering (VQA) tasks. The benchmark includes three types of questions:\n",
                "\n",
                "1. Direct Recognition\n",
                "2. Property Inference\n",
                "3. Counterfactual Reasoning\n",
                "\n",
                "And three types of images:\n",
                "- REAL\n",
                "- ANIMATED\n",
                "- AI GENERATED"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Setup and Imports\n",
                "\n",
                "First, let's import the necessary libraries and set up our environment."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "import torch\n",
                "import json\n",
                "from pathlib import Path\n",
                "from PIL import Image\n",
                "import gc\n",
                "import re\n",
                "from typing import List, Dict, Any\n",
                "\n",
                "# Check if CUDA is available\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "print(f\"Using device: {device}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Benchmark Tester Class\n",
                "\n",
                "This class handles the evaluation of models against our benchmark."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "class BenchmarkTester:\n",
                "    def __init__(self, benchmark_path: str = \"benchmark.json\", data_dir: str = \".\"):\n",
                "        self.benchmark_path = Path(benchmark_path)\n",
                "        self.data_dir = Path(data_dir)\n",
                "        self.results = {}\n",
                "        \n",
                "        # Load benchmark data\n",
                "        with open(self.benchmark_path, 'r') as f:\n",
                "            self.benchmark_data = json.load(f)\n",
                "    \n",
                "    def format_question(self, question: str) -> str:\n",
                "        \"\"\"Format the question to ensure we get a numerical answer.\"\"\"\n",
                "        return f\"{question} Please provide only a number as your answer.\"\n",
                "    \n",
                "    def clean_answer(self, answer: str) -> str:\n",
                "        \"\"\"Extract only the first number from the model's answer.\"\"\"\n",
                "        numbers = re.findall(r'\\d+', answer)\n",
                "        return numbers[0] if numbers else \"\"\n",
                "    \n",
                "    def evaluate_model(self, model, processor, batch_size: int = 3, start_idx: int = 0):\n",
                "        \"\"\"Evaluate the model on the benchmark images.\"\"\"\n",
                "        images = self.benchmark_data['images']\n",
                "        total_images = len(images)\n",
                "        \n",
                "        for i in range(start_idx, total_images, batch_size):\n",
                "            batch = images[i:i + batch_size]\n",
                "            print(f\"Processing batch {i//batch_size + 1}/{(total_images-1)//batch_size + 1}\")\n",
                "            \n",
                "            for img_data in batch:\n",
                "                img_path = Path(img_data['path'])\n",
                "                if not img_path.exists():\n",
                "                    print(f\"Warning: Image not found at {img_path}\")\n",
                "                    continue\n",
                "                \n",
                "                image = Image.open(img_path)\n",
                "                \n",
                "                for q_idx, question in enumerate(img_data['questions']):\n",
                "                    formatted_question = self.format_question(question['text'])\n",
                "                    \n",
                "                    # Process image and question\n",
                "                    inputs = processor(images=image, text=formatted_question, return_tensors=\"pt\").to(device)\n",
                "                    \n",
                "                    with torch.no_grad():\n",
                "                        outputs = model.generate(\n",
                "                            **inputs,\n",
                "                            max_new_tokens=10,  # Reduced from 50\n",
                "                            num_beams=3,\n",
                "                            temperature=0.7\n",
                "                        )\n",
                "                    \n",
                "                    answer = processor.decode(outputs[0], skip_special_tokens=True)\n",
                "                    cleaned_answer = self.clean_answer(answer)\n",
                "                    \n",
                "                    # Store result\n",
                "                    result_key = f\"{img_path.stem}_q{q_idx}\"\n",
                "                    self.results[result_key] = {\n",
                "                        'question': question['text'],\n",
                "                        'model_answer': cleaned_answer,\n",
                "                        'ground_truth': question['answer']\n",
                "                    }\n",
                "                    \n",
                "                    # Clear memory\n",
                "                    del inputs, outputs\n",
                "                    torch.cuda.empty_cache()\n",
                "                    gc.collect()\n",
                "            \n",
                "            # Save checkpoint after each batch\n",
                "            self.save_results(\"checkpoint.json\")\n",
                "    \n",
                "    def save_results(self, filename: str = \"results.json\"):\n",
                "        \"\"\"Save the evaluation results to a JSON file.\"\"\"\n",
                "        with open(filename, 'w') as f:\n",
                "            json.dump(self.results, f, indent=2)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Test Fuyu Model\n",
                "\n",
                "Let's evaluate the Fuyu-8b model on our benchmark."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "def test_fuyu():\n",
                "    from transformers import AutoModelForCausalLM, AutoTokenizer\n",
                "    \n",
                "    print(\"Loading Fuyu-8b model...\")\n",
                "    model = AutoModelForCausalLM.from_pretrained(\n",
                "        \"adept/fuyu-8b\",\n",
                "        torch_dtype=torch.float16,\n",
                "        device_map=\"auto\"\n",
                "    )\n",
                "    processor = AutoTokenizer.from_pretrained(\"adept/fuyu-8b\")\n",
                "    \n",
                "    tester = BenchmarkTester()\n",
                "    tester.evaluate_model(model, processor, batch_size=3)\n",
                "    tester.save_results(\"fuyu_results.json\")\n",
                "    \n",
                "    # Clean up\n",
                "    del model, processor\n",
                "    torch.cuda.empty_cache()\n",
                "    gc.collect()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Test BLIP-2 Model\n",
                "\n",
                "Now let's evaluate the BLIP-2 model."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "def test_blip2():\n",
                "    from transformers import Blip2Processor, Blip2ForConditionalGeneration\n",
                "    \n",
                "    print(\"Loading BLIP-2 model...\")\n",
                "    model = Blip2ForConditionalGeneration.from_pretrained(\n",
                "        \"Salesforce/blip2-flan-t5-xxl\",\n",
                "        torch_dtype=torch.float16,\n",
                "        device_map=\"auto\"\n",
                "    )\n",
                "    processor = Blip2Processor.from_pretrained(\"Salesforce/blip2-flan-t5-xxl\")\n",
                "    \n",
                "    tester = BenchmarkTester()\n",
                "    tester.evaluate_model(model, processor, batch_size=3)\n",
                "    tester.save_results(\"blip2_results.json\")\n",
                "    \n",
                "    # Clean up\n",
                "    del model, processor\n",
                "    torch.cuda.empty_cache()\n",
                "    gc.collect()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Run Evaluation\n",
                "\n",
                "Now we can run our evaluation. Let's start with the Fuyu model:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "test_fuyu()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "And then the BLIP-2 model:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "test_blip2()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write the notebook content to vlm_benchmark.ipynb
with open('vlm_benchmark.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=2)

print("Notebook has been created successfully!") 