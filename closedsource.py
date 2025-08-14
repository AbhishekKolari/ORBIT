# closedsource.py
"""
Closed-source runner with alias mapping.

Function:
    run_closedsource(model_name, benchmark_json, data_dir, output_file, batch_size=360)

- Reads API keys from environment:
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
- Recognizes short aliases or full vendor model ids and routes calls appropriately.
"""

import os
import base64
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import gc
from typing import Optional

import torch
from utils import BenchmarkTester
from utils import resolve_image_path

# Attempt SDK imports; raise error if missing
try:
    import openai
except Exception:
    openai = None

try:
    import anthropic
except Exception:
    anthropic = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

# Defaults if env var not provided
DEFAULT_MODELS = {
    "gpt": "gpt-4o-mini-2024-07-18",
    "claude": "claude-3.7-sonnet",
    "gemini": "gemini-2.0-pro"
}

def _looks_like_model_id(s: str) -> bool:
    """Heuristic: model ids typically contain '-' or '/' (repo-like or vendor model names)."""
    if not s:
        return False
    return ("-" in s) or ("/" in s)

def resolve_closed_model(user_input: str):
    """
    Resolve user_input (alias or full id) into (provider, model_id).
    provider in {'gpt','claude','gemini'}.
    Returns: (provider_str, model_id)
    """
    if not user_input:
        raise ValueError("model_name is required")

    ui = user_input.strip()
    ui_lower = ui.lower()

    # Case 1: user passed a full model id (use exact string)
    if _looks_like_model_id(ui):
        # infer provider by substring
        if "gpt" in ui_lower:
            return "gpt", ui
        if "claude" in ui_lower:
            return "claude", ui
        if "gemini" in ui_lower or "google" in ui_lower:
            return "gemini", ui
        # fallback: try to choose GPT by default if ambiguous
        return "gpt", ui

    # Case 2: user passed a short alias (or something containing provider keyword)
    if "gpt" in ui_lower:
        model_id = os.getenv("OPENAI_MODEL", DEFAULT_MODELS["gpt"])
        return "gpt", model_id
    if "claude" in ui_lower or "anthropic" in ui_lower:
        model_id = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODELS["claude"])
        return "claude", model_id
    if "gemini" in ui_lower or "google" in ui_lower:
        model_id = os.getenv("GOOGLE_MODEL", DEFAULT_MODELS["gemini"])
        return "gemini", model_id

    # Last resort: default to GPT branch with default model id
    return "gpt", os.getenv("OPENAI_MODEL", DEFAULT_MODELS["gpt"])


def _image_to_base64_str(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_closedsource(model_name: str, benchmark_json: str, data_dir: str, output_file: str, batch_size: int = 360):
    """
    Run closed-source vendor multimodal models on the benchmark.
    - model_name: alias or full vendor model id (see resolve_closed_model)
    - keys are read from env variables: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
    """
    provider, resolved_model_id = resolve_closed_model(model_name)
    provider = provider.lower()
    tester = BenchmarkTester(benchmark_json, data_dir)

    # read keys from env
    openai_key = os.getenv("OPENAI_API_KEY")
    anth_key = os.getenv("ANTHROPIC_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    # configure SDKs / clients if needed
    openai_client_ready = False
    anthropic_client = None
    genai_client = None

    if provider == "gpt":
        if openai is None:
            raise RuntimeError("OpenAI SDK not installed. pip install openai")
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required for OpenAI requests")
        openai.api_key = openai_key
        openai_client_ready = True

    if provider == "claude":
        if anthropic is None:
            raise RuntimeError("Anthropic SDK not installed. pip install anthropic")
        if not anth_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is required for Anthropic requests")
        # try common constructors
        try:
            anthropic_client = anthropic.Client(api_key=anth_key)
        except Exception:
            try:
                anthropic_client = anthropic.Anthropic(api_key=anth_key)
            except Exception:
                anthropic_client = None
        if anthropic_client is None:
            raise RuntimeError("Unable to create Anthropic client with installed SDK")

    if provider == "gemini":
        if genai is None:
            raise RuntimeError("Google Generative AI SDK not installed. pip install google-generative-ai")
        if not google_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is required for Gemini requests")
        genai.configure(api_key=google_key)
        genai_client = genai.Client(api_key=google_key)

    results = []
    images = tester.benchmark['benchmark']['images'][:batch_size]
    total_images = len(images)

    for idx, image_data in enumerate(tqdm(images, desc="Processing images")):
        image_path = resolve_image_path(image_data['path'], data_dir, benchmark_json)
        if not image_path.exists():
            print(f"Warning: missing image {image_path}")
            continue

        image_results = []
        for question in image_data['questions']:
            prompt = f"{question['question']} Your response MUST be in the following format and nothing else:\n <NUMBER> [<OBJECT1>, <OBJECT2>, <OBJECT3>, ...]"
            try:
                raw_answer = ""

                if provider == "gpt":
                    # OpenAI ChatCompletion multimodal call: pass base64 image as data URI
                    b64 = _image_to_base64_str(str(image_path))
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                    resp = openai.ChatCompletion.create(
                        model=resolved_model_id,
                        messages=messages,
                        max_tokens=2000,
                        temperature=0.0
                    )
                    
                    try:
                        raw_answer = resp["choices"][0]["message"]["content"].strip()
                    except Exception:
                        try:
                            raw_answer = resp.choices[0].message.content.strip()
                        except Exception:
                            raw_answer = str(resp)

                elif provider == "claude":
                    # Anthropic: embed base64 and send as text payload (SDK shapes vary)
                    with open(image_path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode()
                    combined = f"IMAGE_BASE64:{img_b64}\n\n{prompt}"
                    # many SDKs expose completions.create
                    resp = anthropic_client.completions.create(
                        model=resolved_model_id,
                        prompt=combined,
                        max_tokens_to_sample=200,
                    )
                    raw_answer = getattr(resp, 'completion', getattr(resp, 'text', str(resp))).strip()

                elif provider == "gemini":
                    # Google: upload file handle and call generate_content
                    uploaded = genai_client.files.upload(file=str(image_path))
                    response = genai_client.models.generate_content(
                        model=resolved_model_id,
                        contents=[uploaded, prompt]
                    )
                    raw_answer = getattr(response, 'text', getattr(response, 'content', str(response))).strip()

                else:
                    raise RuntimeError(f"Unhandled provider: {provider}")

                # Parse and append result using your BenchmarkTester cleaning logic
                cleaned = tester.clean_answer(raw_answer)
                image_results.append({
                    "image_id": image_data["image_id"],
                    "image_type": image_data.get("image_type", "unknown"),
                    "question_id": question["id"],
                    "question": question["question"],
                    "ground_truth": question.get("answer"),
                    "model_answer": cleaned["count"],
                    "model_reasoning": cleaned["reasoning"],
                    "raw_answer": raw_answer,
                    "property_category": question.get("property_category")
                })

            except Exception as e:
                print(f"Error for image {image_data['image_id']} q {question['id']}: {e}")
                continue

        results.extend(image_results)

    # Save results to output_file
    if results:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Closed-source run complete: results saved to {output_file}")
    return results
