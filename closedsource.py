# closedsource.py
"""
Closed-source runner with alias mapping.

Function:
    run_closedsource(model_name, benchmark_json, data_dir, output_file, batch_size=360)

- Reads API keys from environment:
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
- Recognizes short aliases or full vendor model ids and routes calls appropriately.
- Gemini branch uses genai.Client.files.upload + client.models.generate_content (per your snippet).
"""

import os
import base64
import io
import json
import gc
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import torch
from utils import BenchmarkTester

# Attempt SDK imports lazily; raise clear error if missing when used
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

# Alias map for common closed-source models (lowercase)
CLOSED_ALIAS_MAP = {
    "gpt-4o": ["gpt-4o", "gpt-4o-mini", "gpt-4o-mini-2024-07-18"],
    "claude-3.7-sonnet": ["claude-3.7-sonnet", "claude-3.7"],
    "gemini-2.0-pro": ["gemini-2.0-pro", "gemini-2.0-flash", "gemini-2.0"]
}

def resolve_closed_alias(user_input):
    user_input_lower = user_input.lower()
    for alias in CLOSED_ALIAS_MAP.keys():
        if user_input_lower.startswith(alias):  # detect variants like -7b or -32b
            return user_input
    return user_input


def _image_to_base64_str(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_closedsource(model_name: str, benchmark_json: str, data_dir: str, output_file: str, batch_size: int = 360):
    """
    Run closed-source vendor multimodal models on the benchmark.
    - model_name: vendor model id or alias (e.g., "gpt-4o-mini-2024-07-18", "gemini-2.0-pro")
    - keys are read from env variables: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
    """
    resolved_name = resolve_closed_alias(model_name)
    resolved_lower = resolved_name.lower()
    tester = BenchmarkTester(benchmark_json, data_dir)

    # read keys from env
    openai_key = os.getenv("OPENAI_API_KEY")
    anth_key = os.getenv("ANTHROPIC_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    # configure SDKs if needed
    if resolved_lower.startswith("gpt"):
        if openai is None:
            raise RuntimeError("OpenAI SDK not installed. pip install openai")
        openai.api_key = openai_key

    if resolved_lower.startswith("claude"):
        if anthropic is None:
            raise RuntimeError("Anthropic SDK not installed. pip install anthropic")
        # instantiate a client object compatible with your installed anthropic package
        try:
            anthropic_client = anthropic.Client(api_key=anth_key)
        except Exception:
            anthropic_client = None

    if resolved_lower.startswith("gemini"):
        if genai is None:
            raise RuntimeError("Google Generative AI SDK not installed. pip install google-generative-ai")
        genai.configure(api_key=google_key)
        client = genai.Client(api_key=google_key)

    results = []
    images = tester.benchmark['benchmark']['images'][:batch_size]
    total_images = len(images)

    for idx, image_data in enumerate(tqdm(images, desc="Processing images")):
        image_path = Path(data_dir) / image_data['path']
        if not image_path.exists():
            print(f"Warning: missing image {image_path}")
            continue

        image_results = []
        for question in image_data['questions']:
            prompt = f"{question['question']} Your response MUST be in the following format and nothing else:\n <NUMBER> [<OBJECT1>, <OBJECT2>, <OBJECT3>, ...]"
            try:
                raw_answer = ""

                # GPT-4o variants (OpenAI)
                if resolved_lower.startswith("gpt"):
                    if openai is None:
                        raise RuntimeError("openai SDK not available")
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
                        model=model_name,
                        messages=messages,
                        max_tokens=2000,
                        temperature=0.0
                    )
                    # Extract assistant message content
                    raw_answer = resp.choices[0].message.content.strip()

                # Claude (Anthropic)
                elif resolved_lower.startswith("claude"):
                    if anthropic is None or anthropic_client is None:
                        raise RuntimeError("anthropic SDK/client not available")
                    with open(image_path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode()
                    combined = f"IMAGE_BASE64:{img_b64}\n\n{prompt}"
                    # NOTE: adjust call according to your installed anthropic SDK version
                    resp = anthropic_client.completions.create(
                        model=model_name,
                        prompt=combined,
                        max_tokens_to_sample=200,
                    )
                    # SDK response structure may vary; handle common patterns
                    raw_answer = getattr(resp, 'completion', getattr(resp, 'text', str(resp))).strip()

                # Gemini (Google)
                elif resolved_lower.startswith("gemini"):
                    if genai is None:
                        raise RuntimeError("google-generative-ai SDK not available")
                    # upload file handle (per your snippet)
                    uploaded = client.files.upload(file=str(image_path))
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[uploaded, prompt]
                    )
                    # response object access can differ across SDK versions
                    raw_answer = getattr(response, 'text', getattr(response, 'content', str(response))).strip()

                else:
                    # last resort: try GPT path if openai available else raise
                    if openai is not None:
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
                            model=model_name,
                            messages=messages,
                            max_tokens=2000,
                            temperature=0.0
                        )
                        raw_answer = resp.choices[0].message.content.strip()
                    else:
                        raise RuntimeError("Can't route closed-source request: unknown model and no openai SDK available.")

                # Parse and append result
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
                # continue on errors per question
                print(f"Error for image {image_data['image_id']} q {question['id']}: {e}")
                continue

        results.extend(image_results)

    # Save results to output_file in same format as opensource branch
    if results:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"Closed-source run complete: results saved to {output_file}")
    return results
