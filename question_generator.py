#!/usr/bin/env python3
"""
Question generator -> writes benchmark-style JSON.

- Provider selection: gpt4o (OpenAI), claude (Anthropic), gemini (Google Generative AI).
- Uses environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY.
- Writes output JSON matching benchmark structure (includes image_types, question_types, property_categories).
"""

import os
import argparse
import logging
import json
import base64
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Optional

from PIL import Image

# Optional provider SDK imports
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

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- Fixed benchmark fields (kept to match benchmark schema) ----------
BENCH_IMAGE_TYPES = ["REAL", "ANIMATED", "AI_GENERATED"]
BENCH_QUESTION_TYPES = {
    "Q1": "direct_recognition",
    "Q2": "property_inference",
    "Q3": "counterfactual_reasoning"
}
BENCH_PROPERTY_CATEGORIES = {
    "Q1": ["physical", "taxonomic"],
    "Q2": ["functional", "relational"],
    "Q3": ["physical", "taxonomic", "functional", "relational"]
}

IMAGE_THEMES = [
    'kitchen', 'zoo', 'biking', 'tools', 'meeting room', 'reptile zoo', 'market', 'tech', 'gym',
    'bedroom', 'classroom', 'garage', 'beach cleanup', 'camping', 'gardening', 'library',
    'wardrobe', 'salon', 'construction', 'driveway', 'laboratory', 'home setup', 'urban',
    'park', 'picnic', 'farm', 'bustop'
]

# ---------- Prompts ----------
def get_theme_prompt() -> str:
    return (
        "Look at this image and classify it into one of these themes: "
        + ", ".join(IMAGE_THEMES)
        + "\n\nSelect the single most appropriate theme from the list that best describes the main subject or setting of the image. "
        "Respond with ONLY the theme name, nothing else."
    )

def get_questions_prompt() -> str:
    return (
        "Property dimension definitions:\n"
        "Physical Properties: Materials (wood, metal, glass, plastic), States (solid, liquid, fragile, flexible), Structural characteristics (has wheels, has handle, has legs)\n"
        "Taxonomic Properties: Biological categories (mammals, reptiles, birds), Artifact categories (furniture, tools, vehicles), Food categories (fruits, vegetables, grains)\n"
        "Functional Properties: Use cases (can be worn, can hold liquid, can cut), Affordances (graspable, openable, foldable), Energy requirements (needs electricity, manual power, battery-operated)\n"
        "Relational Properties: Spatial relations (items on top of other items, inside containers), grouping relations (couple, flock)\n\n"
        "Generate exactly three creative counting questions for this image:\n"
        "1. One question about PHYSICAL or TAXONOMIC properties\n"
        "2. One question about FUNCTIONAL or RELATIONAL properties  \n"
        "3. One counterfactual question about PHYSICAL or TAXONOMIC or FUNCTIONAL or RELATIONAL properties\n\n"
        "Format each question EXACTLY like this:\n"
        "1. [Question]? [Number] ([property]) ([objects])\n"
        "2. [Question]? [Number] ([property]) ([objects])  \n"
        "3. [Question]? [Number] ([property]) ([objects])\n\n"
        "Follow this exact format with no additional text or explanations."
    )

# ---------- Utilities ----------
def validate_image(image_path: str) -> bool:
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.convert("RGB")
        return True
    except Exception as e:
        logger.error(f"Invalid image {image_path}: {e}")
        return False

def extract_image_number(filename: str) -> str:
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else filename

def natural_sort_key(filename: str):
    parts = re.split(r'(\d+)', filename)
    return [int(p) if p.isdigit() else p for p in parts]

def encode_image_data_uri(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# ---------- Parse VLM output into (question,count,property,objects) ----------
def parse_vlm_output(output: Optional[str]) -> List[Tuple[str,str,str,str]]:
    if not output:
        return []
    lines = output.strip().splitlines()
    questions = []
    for line in lines:
        s = line.strip()
        if not s or not re.match(r'^\d+\.', s):
            continue
        if '?' not in s:
            continue
        qpart, rest = s.split('?', 1)
        if '.' in qpart:
            question_text = qpart.split('.',1)[1].strip() + '?'
        else:
            continue
        rest = rest.strip()
        nums = re.findall(r'\d+', rest)
        count = nums[0] if nums else ""
        parens = re.findall(r'\(([^)]*)\)', rest)
        prop = parens[0].strip() if len(parens) >= 1 else ""
        objs = parens[1].strip() if len(parens) >= 2 else ""
        if question_text and count and prop:
            questions.append((question_text, count, prop, objs))
        else:
            logger.warning(f"Incomplete parse for line: {s}")
    return questions

# ---------- Provider wrappers ----------
def openai_generate(image_path: str, prompt: str, model_id: str = "gpt-4o-mini-2024-07-18") -> str:
    if openai is None:
        raise RuntimeError("openai package not installed")
    b64 = encode_image_data_uri(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": b64}},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    resp = openai.ChatCompletion.create(model=model_id, messages=messages, max_tokens=2000, temperature=0.0)
    try:
        # common structure
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        # fallback
        try:
            return resp.choices[0].message.content.strip()
        except Exception:
            return str(resp)

def anthropic_generate(image_path: str, prompt: str, model_id: str = "claude-3.7-sonnet") -> str:
    if anthropic is None:
        raise RuntimeError("anthropic package not installed")
    # instantiate client (supports different SDK versions)
    client = None
    try:
        client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))
    except Exception:
        try:
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        except Exception:
            client = None
    if client is None:
        raise RuntimeError("Unable to create Anthropic client for installed SDK")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    combined = f"IMAGE_BASE64:{b64}\n\n{prompt}"
    # try common invocation patterns
    try:
        resp = client.completions.create(model=model_id, prompt=combined, max_tokens_to_sample=400)
        if hasattr(resp, "completion"):
            return resp.completion.strip()
        if isinstance(resp, dict) and "completion" in resp:
            return resp["completion"].strip()
        return str(resp)
    except Exception:
        try:
            resp = client.create(prompt=combined, model=model_id)
            return getattr(resp, "text", str(resp)).strip()
        except Exception as e:
            raise RuntimeError(f"Anthropic generation failed: {e}")

def gemini_generate(image_path: str, prompt: str, model_id: str = "gemini-2.0-pro") -> str:
    if genai is None:
        raise RuntimeError("google-generative-ai package not installed")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    uploaded = client.files.upload(file=str(image_path))
    resp = client.models.generate_content(model=model_id, contents=[uploaded, prompt])
    if hasattr(resp, "text"):
        return resp.text.strip()
    if isinstance(resp, dict):
        for k in ("content","output","text"):
            if k in resp:
                return str(resp[k]).strip()
    return str(resp)

def provider_generate(provider: str, image_path: str, prompt: str, model_override: Optional[str]=None) -> str:
    provider = provider.lower()
    if "gpt" in provider:
        model_id = model_override or os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")
        return openai_generate(image_path, prompt, model_id)
    elif "claude" in provider:
        model_id = model_override or os.getenv("ANTHROPIC_MODEL", "claude-3.7-sonnet")
        return anthropic_generate(image_path, prompt, model_id)
    elif "gemini" in provider:
        model_id = model_override or os.getenv("GOOGLE_MODEL", "gemini-2.0-pro")
        return gemini_generate(image_path, prompt, model_id)
    else:
        raise ValueError(f"Unknown provider: {provider}")

# ---------- Main: build benchmark JSON ----------
def generate_benchmark_json(model_choice: str, data_dir: str, output_json: str, image_type: str="REAL", test_mode: bool=False, max_images: Optional[int]=None):
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    image_files = [f.name for f in p.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.gif','.tiff')]
    image_files = sorted(image_files, key=natural_sort_key)
    if test_mode and max_images:
        image_files = image_files[:max_images]

    result_images = []
    for idx, img_file in enumerate(image_files, start=1):
        img_path = p / img_file
        logger.info(f"Processing image {idx}/{len(image_files)}: {img_file}")

        if not validate_image(str(img_path)):
            logger.error(f"Skipping invalid image: {img_file}")
            continue

        # classify theme
        try:
            theme_prompt = get_theme_prompt()
            theme_raw = provider_generate(model_choice, str(img_path), theme_prompt)
            theme_norm = (theme_raw or "").strip().lower()
            chosen_theme = None
            for t in IMAGE_THEMES:
                if t.lower() == theme_norm:
                    chosen_theme = t
                    break
            if chosen_theme is None:
                for t in IMAGE_THEMES:
                    if any(word in theme_norm for word in t.lower().split()):
                        chosen_theme = t
                        break
            if chosen_theme is None:
                chosen_theme = IMAGE_THEMES[0]
        except Exception as e:
            logger.warning(f"Theme classification failed for {img_file}: {e}")
            chosen_theme = IMAGE_THEMES[0]

        # generate questions
        try:
            q_prompt = get_questions_prompt()
            raw_output = provider_generate(model_choice, str(img_path), q_prompt)
            parsed = parse_vlm_output(raw_output)
            questions_list = []
            if parsed:
                for qidx, (q_text, count, prop, objs) in enumerate(parsed, start=1):
                    questions_list.append({
                        "id": f"Q{qidx}",
                        "question": q_text,
                        "answer": count,
                        "property_category": prop,
                        "answer_verification": objs if objs else None,
                        "raw_answer": raw_output
                    })
            else:
                # empty question placeholders
                questions_list = []
                logger.warning(f"No parsed questions for {img_file}; raw output:\n{raw_output}")

        except Exception as e:
            logger.error(f"Question generation failed for {img_file}: {e}")
            questions_list = []

        image_entry = {
            "image_id": extract_image_number(img_file),
            "image_type": image_type,
            "theme": chosen_theme,
            "path": str(os.path.join(data_dir, img_file)),
            "questions": questions_list
        }
        result_images.append(image_entry)

    benchmark_out = {
        "benchmark": {
            "image_types": BENCH_IMAGE_TYPES,
            "question_types": BENCH_QUESTION_TYPES,
            "property_categories": BENCH_PROPERTY_CATEGORIES,
            "images": result_images,
            "metadata": {
                "generated_by": model_choice,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec='seconds'),
                "num_images": len(result_images)
            }
        }
    }

    with open(output_json, "w", encoding="utf-8") as fout:
        json.dump(benchmark_out, fout, indent=2, ensure_ascii=False)

    logger.info(f"Wrote benchmark JSON to {output_json} ({len(result_images)} images)")
    return len(result_images)

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Question generator -> benchmark JSON")
    parser.add_argument('--model', required=True, choices=['gpt','claude','gemini'], help='Provider to use')
    parser.add_argument('--data_dir', required=True, help='Directory with images')
    parser.add_argument('--output_json', required=True, help='Output JSON path')
    parser.add_argument('--image_type', default='REAL', choices=['REAL','ANIMATED','AI_GENERATED'], help='Image type tag for produced entries')
    parser.add_argument('--test_mode', action='store_true', help='Process only a few images (for quick tests)')
    parser.add_argument('--max_images', type=int, default=3, help='Max images in test mode')
    args = parser.parse_args()

    # API key checks
    if args.model == 'gpt' and not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY must be set for GPT-4o")
        return
    if args.model == 'claude' and not os.getenv('ANTHROPIC_API_KEY'):
        logger.error("ANTHROPIC_API_KEY must be set for Claude")
        return
    if args.model == 'gemini' and not os.getenv('GOOGLE_API_KEY'):
        logger.error("GOOGLE_API_KEY must be set for Gemini")
        return

    count = generate_benchmark_json(
        model_choice=args.model,
        data_dir=args.data_dir,
        output_json=args.output_json,
        image_type=args.image_type,
        test_mode=args.test_mode,
        max_images=(args.max_images if args.test_mode else None)
    )
    logger.info(f"Done — generated entries for {count} images")

if __name__ == "__main__":
    main()
