from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, FuyuProcessor, FuyuForCausalLM, AutoTokenizer, AutoModel, AutoModelForImageTextToText, Blip2Processor, Blip2ForConditionalGeneration, Gemma3ForConditionalGeneration
import os
import torch
from dotenv import load_dotenv

# load_dotenv()

# Configuration
model_name = "Qwen/Qwen2.5-VL-32B-Instruct"  # Change this if needed
# local_model_dir = "/var/scratch/ave303/models/qwen-7b-instruct"  # Where to save
# local_model_dir = "/var/scratch/ave303/models/internvl3-8b"
# local_model_dir = "/var/scratch/ave303/models/internvl2.5-8b"
# local_model_dir = "/var/scratch/ave303/models/internvl3-14b"
# local_model_dir = "/var/scratch/ave303/models/internvl3-38b"
local_model_dir = "/var/scratch/ave303/models/qwen2-5-vl-32b"
# local_model_dir = "/var/scratch/ave303/models/qwen2-5-vl-7b"
# local_model_dir = "/var/scratch/ave303/models/spacethinker-qwen2-5-vl-3b"  # remyxai/SpaceThinker-Qwen2.5VL-3B
# local_model_dir = "/var/scratch/ave303/models/spaceom"  # remyxai/SpaceOm
# local_model_dir = "/var/scratch/ave303/models/spaceqwen2-5-vl-3b-instruct"  # remyxai/SpaceQwen2.5-VL-3B-Instruct
# local_model_dir = "/var/scratch/ave303/models/sail-vl-8b"
# local_model_dir = "/var/scratch/ave303/models/blip2opt6.7b"
# local_model_dir = "/var/scratch/ave303/models/blip2opt2.7b"
# local_model_dir = "/var/scratch/ave303/models/blip2flan-t5-xl"
# local_model_dir = "/var/scratch/ave303/models/blip2flan-t5-xxl"
# local_model_dir = "/var/scratch/ave303/models/gemma-3-27b-it"
# local_model_dir = "/var/scratch/ave303/models/ristretto-3b"
# local_model_dir = "/var/scratch/ave303/models/smolvlm2"
# local_model_dir = "/var/scratch/ave303/models/fuyu-8b"

# Create directory if it doesn't exist
os.makedirs(local_model_dir, exist_ok=True)

print(f"Downloading model {model_name} into {local_model_dir}...")

# Then try loading again with the corrected code above
# Download model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    cache_dir=local_model_dir,
    # token=os.getenv("HUGGINGFACE_TOKEN"),
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
    # token=os.getenv("HUGGINGFACE_TOKEN"),
    trust_remote_code=True
)
processor.save_pretrained(local_model_dir)

# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_model_dir, trust_remote_code=True, use_fast=False)
# tokenizer.save_pretrained(local_model_dir)

print(f"Model and tokenizer downloaded successfully into {local_model_dir}")
