# Machine Used
Macbook Pro Max M2

# Techstack
Anaconda 

# Create a new environment
conda create -n gita_env python=3.11 -y
conda activate gita_env

# Install the Mac-compatible bridge
pip install unsloth-mlx mlx-lm mlx trl datasets


# CHANGE THIS IMPORT
from unsloth_mlx import FastLanguageModel 
# from unsloth_mlx import SFTTrainer # Use this if standard TRL fails

import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Everything else stays almost identical!
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "mlx-community/Llama-3.2-1B-Instruct", # Use MLX-optimized weights
    max_seq_length = 2048,
    load_in_4bit = True,
)

# ... (rest of the dataset and training code from before)
