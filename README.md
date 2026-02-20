
# Bhagwadgita Small Language Model

## Introduction
### Solution Components

1. Macbook ProMax M2
2. Python
3. mlx-lm Library
4. Kaggle Dataset of Bhagwadgita
5. Hugging Face
6. Mistral-7B-Instruct-v0.3-4bit

## Methodology

### 1. Install
pip install mlx-lm datasets huggingface_hub
huggingface-cli login

### 2. Download model
huggingface-cli download mlx-community/Mistral-7B-Instruct-v0.3-4bit

### 3. Prepare data
python prepare_data.py

### 4. Fine-tune (LoRA, ~10–20 mins on M-Max)
python -m mlx_lm.lora \
    --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
    --data ./data --train \
    --adapter-path ./gita_adapters \
    --batch-size 2 --iters 600 --num-layers 16

### 5. Run inference
python inference.py

## Installation
## Training
## Inference
## Learnings
## Troubleshooting
### 1. Error while training

#### Stacktrace
```
Error while training 
File "/opt/anaconda3/envs/bhagwadgita/lib/python3.10/site-packages/mlx_lm/lora.py", line 362, in main
    run(types.SimpleNamespace(**args))
  File "/opt/anaconda3/envs/bhagwadgita/lib/python3.10/site-packages/mlx_lm/lora.py", line 334, in run
    train_model(args, model, train_set, valid_set, training_callback)
  File "/opt/anaconda3/envs/bhagwadgita/lib/python3.10/site-packages/mlx_lm/lora.py", line 288, in train_model
    train(
  File "/opt/anaconda3/envs/bhagwadgita/lib/python3.10/site-packages/mlx_lm/tuner/trainer.py", line 259, in train
    for it, batch in zip(
  File "/opt/anaconda3/envs/bhagwadgita/lib/python3.10/site-packages/mlx_lm/tuner/trainer.py", line 132, in iterate_batches
    batch = [dataset[j] for j in batch_idx[i]]
  File "/opt/anaconda3/envs/bhagwadgita/lib/python3.10/site-packages/mlx_lm/tuner/trainer.py", line 132, in <listcomp>
    batch = [dataset[j] for j in batch_idx[i]]
  File "/opt/anaconda3/envs/bhagwadgita/lib/python3.10/site-packages/mlx_lm/tuner/datasets.py", line 168, in __getitem__
    self._proc_data[idx] = self._data.process(self._data[idx])
  File "/opt/anaconda3/envs/bhagwadgita/lib/python3.10/site-packages/mlx_lm/tuner/datasets.py", line 60, in process
    tokens = self.tokenizer.apply_chat_template(
  File "/opt/anaconda3/envs/bhagwadgita/lib/python3.10/site-packages/mlx_lm/tokenizer_utils.py", line 322, in apply_chat_template
    return self._tokenizer.apply_chat_template(*args, tokenize=tokenize, **kwargs)
  File "/opt/anaconda3/envs/bhagwadgita/lib/python3.10/site-packages/transformers/tokenization_utils_base.py", line 3029, in apply_chat_template
    rendered_chat, generation_indices = render_jinja_template(
  File "/opt/anaconda3/envs/bhagwadgita/lib/python3.10/site-packages/transformers/utils/chat_template_utils.py", line 537, in render_jinja_template
    rendered_chat = compiled_template.render(
  File "/opt/anaconda3/envs/bhagwadgita/lib/python3.10/site-packages/jinja2/environment.py", line 1295, in render
    self.environment.handle_exception()
  File "/opt/anaconda3/envs/bhagwadgita/lib/python3.10/site-packages/jinja2/environment.py", line 942, in handle_exception
    raise rewrite_traceback_stack(source=source)
  File "<template>", line 1, in top-level template code
  File "/opt/anaconda3/envs/bhagwadgita/lib/python3.10/site-packages/jinja2/sandbox.py", line 401, in call
    return __context.call(__obj, *args, **kwargs)
  File "/opt/anaconda3/envs/bhagwadgita/lib/python3.10/site-packages/transformers/utils/chat_template_utils.py", line 445, in raise_exception
    raise jinja2.exceptions.TemplateError(message)
jinja2.exceptions.TemplateError: Conversation roles must alternate user/assistant/user/assistant/..
```
#### Solution

Instead of 
```
{"role": "system",    "content": "You are a Gita guide..."}
{"role": "user",      "content": "What is Atman?"}
{"role": "assistant", "content": "..."}
```

Use in training
```
{"role": "user",      "content": "You are a Gita guide...\n\nWhat is Atman?"}
{"role": "assistant", "content": "..."}
```
### 2. Error while inferencing

#### Inference response from trained SLM
```
==========
<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>
==========
```
#### Solution 1

```
# load adapters manually, bypassing the buggy path

python -c "
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.utils import load_adapters

# Load base model cleanly first
model, tok = load('mlx-community/Mistral-7B-Instruct-v0.3-4bit')

# Apply adapters separately (only touches lora_a / lora_b tensors)
model.load_weights('./gita_adapters_v2/adapters.safetensors', strict=False)

prompt = '[INST] What is Atman according to the Bhagavad Gita? [/INST]'
print(generate(model, tok, prompt=prompt, max_tokens=300, verbose=True))
"
```

#### Steps taken to arrive at solution 1

##### A. Check if basemodel is OK
```
# Test A: base model, no adapters
python -c "
from mlx_lm import load, generate
model, tok = load('mlx-community/Mistral-7B-Instruct-v0.3-4bit')
prompt = '[INST] What is Atman? [/INST]'
print(generate(model, tok, prompt=prompt, max_tokens=50, verbose=False))
"
```

##### B. Inspect adapters
```
# Test B: inspect what the adapter actually modified
python -c "
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load

model, _ = load('mlx-community/Mistral-7B-Instruct-v0.3-4bit', adapter_path='./gita_adapters_v2')
weights = dict(tree_flatten(model.trainable_parameters()))
print('Number of adapter tensors:', len(weights))
for k, v in list(weights.items())[:5]:
    print(f'  {k}: shape={v.shape}, mean={mx.mean(mx.abs(v)).item():.6f}')
"
```
It was observed that weights were not uniform. LayerNorms should have weights near 1.0 uniformly. Values like 0.085 and 1.13 mean they've been mutated by the adapter loading — this is what's destroying every token before it even reaches the attention layers. This is a known MLX bug where load() with adapter_path incorrectly marks LayerNorm weights as trainable and merges corrupted values into them.

##### C. layers the adapter touched
```
# Test C: check what layers the adapter touched
python -c "
from safetensors import safe_open
import os
path = './gita_adapters_v2/adapters.safetensors'
print('File size:', os.path.getsize(path), 'bytes')
with safe_open(path, framework='pt') as f:
    keys = list(f.keys())
print('Total tensors:', len(keys))
print('First 5 keys:', keys[:5])
print('Last 5 keys:', keys[-5:])
# Check if embed_tokens was accidentally trained
embed_keys = [k for k in keys if 'embed' in k or 'lm_head' in k]
print('Embedding/head keys (should be EMPTY):', embed_keys)
"
```
 if embed_tokens or lm_head appears in those keys, the adapter corrupted the vocabulary mapping itself, which is why every token decodes to <unk>. That would require a targeted fix to the LoRA config to explicitly exclude those layers.

 ##### D. Check if JSONL, Tokenizer and mlx_lm's ability to load dataset is ok, 
 ```
 # 1. Check what your JSONL actually looks like
head -n 1 /Users/sj/DevManus/gitaSLM/data/train.jsonl
 

 # 2. Check how the tokenizer processes your prompt
python -c "
from mlx_lm import load
_, tok = load('mlx-community/Mistral-7B-Instruct-v0.3-4bit')
prompt = '<s>[INST] What is Atman? [/INST]'
ids = tok.encode(prompt)
print('Token IDs:', ids)
print('Decoded back:', [tok.decode([i]) for i in ids])
print('Total tokens:', len(ids))
"

# 3. Check what mlx_lm sees when it loads your dataset
python -c "
import json
with open('/Users/sj/DevManus/gitaSLM/data/train.jsonl') as f:
    row = json.loads(f.readline())
print(json.dumps(row, indent=2))
"
```

#### Solution 2
```
python -c "
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
import mlx.core as mx

model, tok = load('mlx-community/Mistral-7B-Instruct-v0.3-4bit')

# Verify LayerNorm weights BEFORE adapter load - should all be near 1.0
params = dict(model.named_modules())
ln = model.model.layers[0].input_layernorm
print('LayerNorm weight (base, should be ~1.0):', mx.mean(mx.abs(ln.weight)).item())

# Now load adapter
model.load_weights('./gita_adapters_v2/adapters.safetensors', strict=False)
print('LayerNorm weight (after adapter, if corrupted will differ):', mx.mean(mx.abs(ln.weight)).item())
"
```








# Machine Used
Macbook Pro Max M2

# Techstack
Anaconda 
https://github.com/ml-explore/mlx-examples/tree/main

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
