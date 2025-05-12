from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer, AutoModel, AutoModelForImageTextToText, Blip2Processor, Blip2ForConditionalGeneration
import os
import torch

# Configuration
model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"  # Change this if needed
# local_model_dir = "/var/scratch/ave303/models/qwen-7b-instruct"  # Where to save
# local_model_dir = "/var/scratch/ave303/models/internvl2.5-4b"
# local_model_dir = "/var/scratch/ave303/models/sail-vl-8b"
# local_model_dir = "/var/scratch/ave303/models/blip2opt6.7b"
# local_model_dir = "/var/scratch/ave303/models/ristretto-3b"
local_model_dir = "/var/scratch/ave303/models/smolvlm2"

# Create directory if it doesn't exist
os.makedirs(local_model_dir, exist_ok=True)

print(f"Downloading model {model_name} into {local_model_dir}...")

# Then try loading again with the corrected code above
# Download model
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    cache_dir=local_model_dir,
#     torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True  # Needed for some custom models like Qwen
).eval()

# Save the model
model.save_pretrained(local_model_dir)

# Download tokenizer
processor = AutoProcessor.from_pretrained(
    model_name,
    cache_dir=local_model_dir,
    trust_remote_code=True
)
processor.save_pretrained(local_model_dir)

# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_model_dir, trust_remote_code=True, use_fast=False)
# tokenizer.save_pretrained(local_model_dir)

print(f"Model and tokenizer downloaded successfully into {local_model_dir}")
