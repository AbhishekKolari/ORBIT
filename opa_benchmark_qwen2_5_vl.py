# VLM Benchmark for Object Property Abstraction
#
# This script implements a benchmark for evaluating Vision Language Models (VLMs) on object property abstraction and visual question answering (VQA) tasks.
# The benchmark includes three types of questions:
# 1. Direct Recognition
# 2. Property Inference
# 3. Counterfactual Reasoning
#
# And three types of images:
# - REAL
# - ANIMATED
# - AI GENERATED

# =====================
# Setup and Imports
# =====================

# Install required packages (commented out, handled by environment)
# %pip install transformers torch Pillow tqdm bitsandbytes accelerate
# %pip install qwen-vl-utils flash-attn #--no-build-isolation

import torch
import json
from pathlib import Path
from PIL import Image
import gc
import re
from tqdm import tqdm
from typing import List, Dict, Any
from qwen_vl_utils import process_vision_info
import time

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =====================
# Benchmark Tester Class
# =====================

class BenchmarkTester:
    def __init__(self, benchmark_path="/var/scratch/ave303/OP_bench/benchmark.json", data_dir="/var/scratch/ave303/OP_bench/"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with open(benchmark_path, 'r') as f:
            self.benchmark = json.load(f)
        self.data_dir = data_dir

    def clean_answer(self, answer):
        """Extract number and reasoning from the model's answer."""
        import re
        pattern = r'(\d+)\s*\[(.*?)\]'
        match = re.search(pattern, answer)
        if match:
            number = match.group(1)
            objects = [obj.strip() for obj in match.group(2).split(',')]
            return {
                "count": number,
                "reasoning": objects
            }
        else:
            numbers = re.findall(r'\d+', answer)
            return {
                "count": numbers[0] if numbers else "0",
                "reasoning": []
            }

    def model_generation(self, model_name, model, inputs, processor):
        """Generate answer and decode with greedy decoding."""
        outputs = None
        if model_name == "Qwen2.5-VL":
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                num_beams=1,
                early_stopping=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
            outputs = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
            ]
            answer = processor.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        else:
            print(f"Warning: Unknown model name '{model_name}' in model_generation.")
            answer = ""
        return answer, outputs

    def evaluate_model(self, model_name, model, processor, save_path, start_idx=0, batch_size=5):
        results = []
        successful_images = []
        failed_images = []
        total_questions_processed = 0
        total_questions_failed = 0
        print(f"\nEvaluating {model_name}...")
        print(f"Using device: {self.device}")
        gc.collect()
        torch.cuda.empty_cache()
        try:
            images = self.benchmark['benchmark']['images'][start_idx:start_idx + batch_size]
            total_images = len(images)
            for idx, image_data in enumerate(tqdm(images, desc="Processing images")):
                image_start_time = time.time()
                current_image_questions_failed = 0
                current_image_questions_total = 0
                try:
                    image_path = Path(self.data_dir)/image_data['path']
                    if not image_path.exists():
                        failed_images.append({
                            'image_id': image_data['image_id'],
                            'image_type': image_data.get('image_type', 'unknown'),
                            'error_type': 'file_not_found',
                            'error_message': f'Image not found at {image_path}'
                        })
                        continue
                    image = Image.open(image_path).convert("RGB")
                    image_results = []
                    for question_idx, question in enumerate(image_data['questions']):
                        current_image_questions_total += 1
                        total_questions_processed += 1
                        try:
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": image},
                                        {"type": "text", "text": f"{question['question']} Your response MUST be in the following format and nothing else:\n <NUMBER> [<OBJECT1>, <OBJECT2>, <OBJECT3>, ...]"}
                                    ]
                                },
                            ]
                            torch.cuda.empty_cache()
                            text = processor.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True
                            )
                            inputs = processor(
                                text=text,
                                images=image,
                                videos=None,
                                padding=True,
                                return_tensors="pt",
                            ).to("cuda")
                            with torch.no_grad():
                                answer, outputs = self.model_generation(model_name, model, inputs, processor)
                            cleaned_answer = self.clean_answer(answer)
                            image_results.append({
                                "image_id": image_data["image_id"],
                                "image_type": image_data.get("image_type", "unknown"),
                                "question_id": question["id"],
                                "question": question["question"],
                                "ground_truth": question["answer"],
                                "model_answer": cleaned_answer["count"],
                                "model_reasoning": cleaned_answer["reasoning"],
                                "raw_answer": answer,
                                "property_category": question["property_category"]
                            })
                            del outputs, inputs
                            torch.cuda.empty_cache()
                        except Exception as e:
                            current_image_questions_failed += 1
                            total_questions_failed += 1
                            continue
                    results.extend(image_results)
                    questions_succeeded = current_image_questions_total - current_image_questions_failed
                    if current_image_questions_failed == 0:
                        successful_images.append({
                            'image_id': image_data['image_id'],
                            'image_type': image_data.get('image_type', 'unknown'),
                            'questions_total': current_image_questions_total,
                            'questions_succeeded': questions_succeeded,
                            'processing_time': time.time() - image_start_time
                        })
                    else:
                        image_success_rate = (questions_succeeded / current_image_questions_total * 100) if current_image_questions_total > 0 else 0
                        failed_images.append({
                            'image_id': image_data['image_id'],
                            'image_type': image_data.get('image_type', 'unknown'),
                            'error_type': 'partial_failure',
                            'questions_total': current_image_questions_total,
                            'questions_failed': current_image_questions_failed,
                            'questions_succeeded': questions_succeeded,
                            'success_rate': f"{image_success_rate:.1f}%"
                        })
                    if (idx + 1) % 2 == 0 or idx == total_images - 1:
                        checkpoint_path = f"{save_path}_checkpoint.json"
                        with open(checkpoint_path, 'w') as f:
                            json.dump(results, f, indent=4)
                except Exception as e:
                    failed_images.append({
                        'image_id': image_data['image_id'],
                        'image_type': image_data.get('image_type', 'unknown'),
                        'error_type': 'complete_failure',
                        'error_message': str(e)
                    })
                    continue
            if results:
                with open(save_path, 'w') as f:
                    json.dump(results, f, indent=4)
        except Exception as e:
            if results:
                error_save_path = f"{save_path}_error_state.json"
                with open(error_save_path, 'w') as f:
                    json.dump(results, f, indent=4)
        self._print_evaluation_summary(
            model_name, total_images, successful_images, failed_images, 
            total_questions_processed, total_questions_failed, len(results)
        )
        return results

    def _print_evaluation_summary(self, model_name, total_images, successful_images, failed_images, total_questions_processed, total_questions_failed, total_results):
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY FOR {model_name.upper()}")
        print(f"{'='*60}")
        num_successful = len(successful_images)
        num_failed = len(failed_images)
        print(f"📊 IMAGE PROCESSING SUMMARY:")
        print(f"   Total images attempted: {total_images}")
        print(f"   Successfully processed: {num_successful} ({num_successful/total_images*100:.1f}%)")
        print(f"   Failed images: {num_failed} ({num_failed/total_images*100:.1f}%)")
        questions_succeeded = total_questions_processed - total_questions_failed
        print(f"\n📝 QUESTION PROCESSING SUMMARY:")
        print(f"   Total questions attempted: {total_questions_processed}")
        print(f"   Successfully processed: {questions_succeeded} ({questions_succeeded/total_questions_processed*100:.1f}%)")
        print(f"   Failed questions: {total_questions_failed} ({total_questions_failed/total_questions_processed*100:.1f}%)")
        print(f"   Results saved: {total_results}")
        if successful_images:
            print(f"\n✅ SUCCESSFUL IMAGES ({len(successful_images)}):")
            for img in successful_images:
                print(f"   • {img['image_id']} (Type: {img['image_type']}, "
                      f"Questions: {img['questions_succeeded']}/{img['questions_total']}, "
                      f"Time: {img['processing_time']:.1f}s)")
        if failed_images:
            print(f"\n❌ FAILED/PROBLEMATIC IMAGES ({len(failed_images)}):")
            for img in failed_images:
                if img['error_type'] == 'complete_failure':
                    print(f"   • {img['image_id']} (Type: {img['image_type']}) - "
                          f"COMPLETE FAILURE: {img.get('error_message', 'Unknown error')}")
                elif img['error_type'] == 'partial_failure':
                    print(f"   • {img['image_id']} (Type: {img['image_type']}) - "
                          f"PARTIAL: {img['questions_failed']}/{img['questions_total']} failed "
                          f"({img['success_rate']} success)")
                elif img['error_type'] == 'file_not_found':
                    print(f"   • {img['image_id']} (Type: {img['image_type']}) - "
                          f"FILE NOT FOUND: {img['error_message']}")
        if failed_images:
            print(f"\n📈 FAILURE ANALYSIS BY IMAGE TYPE:")
            from collections import defaultdict
            failures_by_type = defaultdict(list)
            for img in failed_images:
                failures_by_type[img['image_type']].append(img)
            for img_type, failures in failures_by_type.items():
                print(f"   • {img_type}: {len(failures)} failed images")
                for failure in failures:
                    print(f"     - {failure['image_id']} ({failure['error_type']})")
        print(f"{'='*60}\n")

# =====================
# Test Qwen2.5-VL
# =====================

def test_Qwen2_5VL():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-3B-Instruct", 
    #     load_in_8bit=True, # throws error when .to() is added
    #     torch_dtype=torch.bfloat16, 
    #     device_map="auto",
    #     low_cpu_mem_usage=True
    # )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/var/scratch/ave303/models/qwen2-5-vl-32b",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained("/var/scratch/ave303/models/qwen2-5-vl-32b")
    if hasattr(model.config, 'use_memory_efficient_attention'):
        model.config.use_memory_efficient_attention = True
    tester = BenchmarkTester()
    Qwen2_5VL_results = tester.evaluate_model(
        "Qwen2.5-VL",
        model, 
        processor, 
        "Qwen2-5-vl-32b_results.json",
        batch_size=360
    )
    del model, processor
    torch.cuda.empty_cache()
    gc.collect()

# =====================
# Main Execution
# =====================

if __name__ == '__main__':
    # Uncomment the following line to run the Qwen2.5-VL evaluation
    test_Qwen2_5VL()
    # To run SmolVLM2 evaluation, implement and call test_smolVLM2() as needed. 