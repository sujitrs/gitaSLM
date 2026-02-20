"""
prepare_data.py  (FIXED)
────────────────────────
Downloads and prepares the Bhagavad Gita QA dataset from Hugging Face
into the JSONL format required by mlx-lm for fine-tuning.

FIX: Mistral-7B-Instruct-v0.3 (and many models) does NOT accept a
     standalone "system" role message — it requires strict
     user / assistant / user / assistant alternation.

     Solution: merge the system prompt into the first user message,
     which is the standard workaround for Mistral-family models.

Usage:
    python prepare_data.py

Output:
    data/train.jsonl  (~400 examples)
    data/valid.jsonl  (~50 examples)
    data/test.jsonl   (~50 examples)
"""

import json
import os
from datasets import load_dataset

# ─────────────────────────────────────────────
# Load Bhagavad Gita QA dataset from Hugging Face
# ─────────────────────────────────────────────
print("Loading Bhagavad Gita QA dataset from Hugging Face...")
ds = load_dataset("sweatSmile/Bhagavad-Gita-Vyasa-Edwin-Arnold")

SYSTEM_PROMPT = (
    "You are a knowledgeable guide on the Bhagavad Gita. "
    "Answer questions based on the teachings and wisdom of this sacred text."
)

# ─────────────────────────────────────────────
# FORMAT OPTION A — "messages" (chat) format
#   Works with: Phi-3, Llama-3, Qwen-2, OpenChat
#   System role IS supported by these models.
# ─────────────────────────────────────────────
def format_chat_with_system(example):
    """For models that support a system role (Phi-3, Llama-3, Qwen-2)."""
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }

# ─────────────────────────────────────────────
# FORMAT OPTION B — merged user message (DEFAULT )
#   Works with: Mistral-v0.1/v0.2/v0.3, Mixtral
#   Mistral's chat template requires strict user/assistant alternation.
#   Fix: prepend system prompt into the first user turn.
# ─────────────────────────────────────────────
def format_chat_merged_system(example):
    """For Mistral-family models — merges system into first user message."""
    user_content = f"{SYSTEM_PROMPT}\n\n{example['question']}"
    return {
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": example["answer"]},
        ]
    }

# ─────────────────────────────────────────────
# FORMAT OPTION C — plain "text" completion format
#   Works with ALL models as a universal fallback.
#   No chat template is applied — raw prompt/response text.
# ─────────────────────────────────────────────
def format_completion(example):
    """Universal fallback — plain prompt/completion text."""
    text = (
        f"<s>[INST] {SYSTEM_PROMPT}\n\n{example['question']} [/INST] "
        f"{example['answer']} </s>"
    )
    return {"text": text}

# ─────────────────────────────────────────────
# SELECT YOUR FORMAT HERE
#
#  - format_chat_merged_system  → Mistral, Mixtral (DEFAULT)
#  - format_chat_with_system    → Phi-3, Llama-3, Qwen-2, Gemma-2
#  - format_completion          → universal fallback (any model)
# ─────────────────────────────────────────────
FORMAT_FN   = format_chat_merged_system   # ← change this if needed
FORMAT_NAME = "chat_merged_system (Mistral-compatible)"

print(f" Formatting with: {FORMAT_NAME}")

formatted = ds["train"].map(
    FORMAT_FN,
    remove_columns=ds["train"].column_names
)

# ─────────────────────────────────────────────
# Split: 80% train / 10% valid / 10% test
# ─────────────────────────────────────────────
split    = formatted.train_test_split(test_size=0.2, seed=42)
train_ds = split["train"]
temp     = split["test"].train_test_split(test_size=0.5, seed=42)
valid_ds = temp["train"]
test_ds  = temp["test"]

# ─────────────────────────────────────────────
# Save as JSONL files (required by mlx-lm)
# ─────────────────────────────────────────────
os.makedirs("data", exist_ok=True)

for split_name, dataset in [("train", train_ds), ("valid", valid_ds), ("test", test_ds)]:
    path = f"data/{split_name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Saved {len(dataset):>3} examples → {path}")

# ─────────────────────────────────────────────
# Preview a sample
# ─────────────────────────────────────────────
print("\n Sample from train.jsonl:")
with open("data/train.jsonl") as f:
    sample = json.loads(f.readline())

if "messages" in sample:
    for msg in sample["messages"]:
        role    = msg["role"].upper()
        content = msg["content"]
        preview = content[:130] + ("..." if len(content) > 130 else "")
        print(f"  [{role}] {preview}")
else:
    print("  [TEXT]", sample["text"][:200], "...")

print("\n Dataset preparation complete!")
print(f"   Format : {FORMAT_NAME}")
print(f"   Train  : {len(train_ds)} examples")
print(f"   Valid  : {len(valid_ds)} examples")
print(f"   Test   : {len(test_ds)} examples")
print()
print(" Next: run the fine-tuning command below")
