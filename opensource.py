# opensource.py
"""
Open-source runner with alias mapping.

Function:
    run_opensource(model_name, processor_path, benchmark_json, data_dir, output_file, batch_size=360)

- Supports hardcoded tested models (Qwen2.5-VL, BLIP2, InternVL3, Gemma3) using the special loading
  & config required for reproducibility.
- Falls back to a generic Hugging Face loader for any other HF repo id or local path.
"""

import gc
import torch
from pathlib import Path
from utils import BenchmarkTester

# Hardcoded aliases -> representative repo ids (lowercase)
MODEL_ALIAS_MAP = {
    "qwen2.5-vl": ["qwen/qwen2.5-vl-7b-instruct", "qwen/qwen2.5-vl-32b-instruct"],
    "blip2": ["salesforce/blip2-flan-t5-xxl", "salesforce/blip2-opt-2.7b", "salesforce/blip2-opt-6.7b"],
    "internvl3": ["opengvlab/internvl3-8b", "opengvlab/internvl3-14b"],
    "gemma3": ["google/gemma-3-27b-it"]
}

def resolve_model_alias(user_input):
    user_input_lower = user_input.lower()
    for alias in MODEL_ALIAS_MAP.keys():
        if user_input_lower.startswith(alias):  # detect variants like -7b or -32b
            return user_input
    return user_input


def run_opensource(model_name: str, processor_path: str, benchmark_json: str, data_dir: str, output_file: str, batch_size: int = 360):
    """
    Load either a hardcoded tested model (alias) or any HF model by repo id/local path.
    Calls BenchmarkTester.evaluate_model(...) which contains per-model generation/preproc branches.
    """
    resolved_name = resolve_model_alias(model_name)
    resolved_lower = resolved_name.lower()
    tester = BenchmarkTester(benchmark_json, data_dir)

    # Hardcoded model branches for reproducibility
    if resolved_lower.startswith("qwen2.5-vl"):
        # Qwen has its own model class & often requires trust_remote_code
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print(f"Loading hardcoded model Qwen2.5-VL from '{model_name}' (alias resolved to qwen2.5-vl)")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(processor_path or model_name)

    elif resolved_lower.startswith("blip2"):
        from transformers import Blip2ForConditionalGeneration, AutoProcessor
        print(f"Loading hardcoded model BLIP-2 from '{model_name}' (alias resolved to blip2)")
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        processor = AutoProcessor.from_pretrained(processor_path or model_name)

    elif resolved_lower.startswith("internvl3"):
        # InternVL often provides a chat API and custom preprocessing (handled in utils.BenchmarkTester)
        from transformers import AutoModelForCausalLM, AutoProcessor
        print(f"Loading hardcoded model InternVL3 from '{model_name}' (alias resolved to internvl3)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(processor_path or model_name)

    elif resolved_lower.startswith("gemma3"):
        # Gemma often uses AutoModelForCausalLM with bfloat16 for best perf on supported hw
        from transformers import AutoModelForCausalLM, AutoProcessor
        print(f"Loading hardcoded model Gemma3 from '{model_name}' (alias resolved to gemma3)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(processor_path or model_name)

    else:
        # Generic HF fallback: attempt to load processor + AutoModelForCausalLM
        from transformers import AutoModelForCausalLM, AutoProcessor
        print(f"No hardcoded alias resolved for '{model_name}'. Loading generically from HF or local path.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        # processor_path default fallback to model_name (repo id or local path)
        processor = AutoProcessor.from_pretrained(processor_path or model_name)

    # optional small tweak if model supports memory efficient attention
    if hasattr(model.config, 'use_memory_efficient_attention'):
        model.config.use_memory_efficient_attention = True

    try:
        tester.evaluate_model(model_name, model, processor, output_file, batch_size=batch_size)
    finally:
        # cleanup
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
