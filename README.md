
## Overview

This guide walks you through fine-tuning a large language model using Apple's **MLX** framework and **LoRA** (Low-Rank Adaptation) on the Bhagavad Gita QA dataset from Hugging Face. Everything runs locally on your Mac — no cloud, no GPU rental.


1. Macbook ProMax M2
2. Python Python 3.10+
3. mlx-lm Library
4. Kaggle Dataset of Bhagwadgita : `sweatSmile/Bhagavad-Gita-Vyasa-Edwin-Arnold` : `question`, `answer` — 500 QA pairs covering all 18 chapters  
5. Hugging Face
6. Base Model: Mistral-7B-Instruct-v0.3-4bit  (quantized, ~4GB)
7. lora_config.yaml (Present in this repo)
8. inference.py (Present in this repo)
9. prepare_data.py (Present in this repo)

---

## Step 1: Environment Setup

```bash
# Create and activate a virtual environment
python3 -m venv gita_mlx_env
source gita_mlx_env/bin/activate

# Install required packages
pip install --upgrade pip
pip install mlx-lm
pip install datasets huggingface_hub pandas
```

Login to Hugging Face (needed to download models):

```bash
huggingface-cli login
# Paste your token from: https://huggingface.co/settings/tokens
```

---

## Step 2: Download the Base Model

```bash
# Download a 4-bit quantized model optimized for Apple Silicon
huggingface-cli download mlx-community/Mistral-7B-Instruct-v0.3-4bit

# Alternative lighter model (good for 16GB RAM Macs):
# huggingface-cli download mlx-community/Phi-3.5-mini-instruct-4bit
```

---

## Step 3: Prepare the Dataset
Run the preparation script:

```bash
python prepare_data.py
```

---

## Step 4: Fine-Tune with LoRA (Recommended) ~10–20 mins on M-Max

LoRA trains only a small number of adapter weights, making it fast and memory-efficient.

```bash
python -m mlx_lm.lora -c /Users/sj/DevManus/gitaSLM/lora_config.yaml
```

---

## Step 5: Test Your Fine-Tuned Model

Test inference using your LoRA adapters (without fusing):

```bash
python inference.py
```

Output

```
Your question: why does Arjun seem to be perpetually confused with Krishnas teachings

Arjuna's confusion in the Bhagavad Gita arises from several factors:

1. Overwhelming Situation: Arjuna is faced with a difficult situation, having to fight against his own kinsmen and teachers. This emotional turmoil clouds his judgment and understanding.

2. Divine Wisdom: Krishna's teachings are profound and transcend human understanding. Arjuna, being a human, struggles to grasp the divine wisdom that Krishna imparts.

3. Human Limitations: Arjuna is a warrior, not a philosopher or a sage. His questions and confusion reflect the human struggle to understand the divine and the eternal.

4. Duality vs. Non-Duality: Krishna's teachings often emphasize the non-dual nature of reality, which is difficult for Arjuna, who is accustomed to perceiving the world in dualistic terms, to fully comprehend.

5. Transformation: Throughout the Gita, Arjuna's understanding evolves, reflecting the progressive nature of spiritual growth. His initial confusion gives way to understanding as Krishna guides him towards enlightenment.

In essence, Arjuna's confusion mirrors the human condition, highlighting the struggle to reconcile our human limitations with the divine wisdom that transcends them.

Your question: who is khatu shyam

Khatu Shyam is not a character or deity directly mentioned in the Bhagavad Gita. The Bhagavad Gita is a 700-verse Hindu scripture that is a part of the Indian epic Mahabharata, where the deity Krishna speaks to the warrior Arjuna about moral and spiritual duties.

Khatu Shyam is a revered form of Krishna worshipped in the town of Khatu in the Indian state of Rajasthan. This form of Krishna is believed to have appeared in the form of a dark-skinned, bearded man, and is associated with miracles and healing powers. The legend surrounding Khatu Shyam is not found in the Bhagavad Gita, but rather in local folklore and traditions.

Your question: exit

🙏 Namaste! Goodbye.
```

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

# 4. Verify the adapters loaded correctly and aren't empty, run:

ls -lh ./gita_adapters/

You should see adapters.safetensors with a non-zero file size (typically 50–200 MB for 16 LoRA layers on Mistral-7B). If it's 0 bytes or missing then there is a issue.

# Use the latest checkpoint directly in case of 0 bytes or missing adapters.safetensors
python -c "
from mlx_lm import load, generate
model, tok = load(
    'mlx-community/Mistral-7B-Instruct-v0.3-4bit',
    adapter_path='./gita_adapters/0000600_adapters.safetensors'  # adjust number
)
prompt = '<s>[INST] What is Atman? [/INST]'
print(generate(model, tok, prompt=prompt, max_tokens=300))
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







