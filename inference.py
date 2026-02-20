"""
inference.py  — FINAL WORKING VERSION
──────────────────────────────────────
Bhagavad Gita QA — MLX + LoRA fine-tuned on Mistral-7B-Instruct-v0.3

Root cause of all previous issues, documented for reference:
  1. <unk> flood — adapter_path= in load() corrupts LayerNorm weights.
     Fix: load base model first, then call model.load_weights() separately.
  2. Double <s> token — never manually prepend <s>; tokenizer adds it.
     Fix: prompt starts with [INST], not <s>[INST].
  3. system role — Mistral chat template rejects standalone system messages.
     Fix: merge system prompt into the first user turn.

Usage:
    python inference.py
    python inference.py --question "What is karma yoga?"
    python inference.py --question "Describe Arjuna's dilemma" --max-tokens 400
"""

import argparse
import mlx.core as mx
from mlx_lm import load, generate

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
BASE_MODEL      = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
ADAPTER_WEIGHTS = "./gita_adapters_v2/adapters.safetensors"

SYSTEM_PROMPT = (
    "You are a knowledgeable guide on the Bhagavad Gita. "
    "Answer questions based on the teachings and wisdom of this sacred text."
)

# ─────────────────────────────────────────────
# Load model correctly
#
# ⚠️  DO NOT use: load(model, adapter_path=...)
#     That merges adapter config into model init and corrupts LayerNorms.
#
# ✅  DO: load base model cleanly, then apply adapter weights separately.
# ─────────────────────────────────────────────
print(f"⚙️  Loading base model: {BASE_MODEL}")
model, tokenizer = load(BASE_MODEL)

print(f"🔌 Applying adapters: {ADAPTER_WEIGHTS}")
model.load_weights(ADAPTER_WEIGHTS, strict=False)
mx.eval(model.parameters())

print("✅ Ready.\n")


# ─────────────────────────────────────────────
# Prompt builder
#
# Mistral [INST] format — no <s> prefix (tokenizer adds it automatically),
# no system role (Mistral rejects it), system merged into user turn.
# ─────────────────────────────────────────────
def build_prompt(question: str) -> str:
    return f"[INST] {SYSTEM_PROMPT}\n\n{question} [/INST]"


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
def ask_gita(question: str, max_tokens: int = 350, temperature: float = 0.1) -> str:
    response = generate(
        model,
        tokenizer,
        prompt=build_prompt(question),
        max_tokens=max_tokens,
#        temperature=temperature,
        verbose=False,
    )
    # Strip any trailing special tokens
    for stop in ["</s>", "[INST]", "[/INST]"]:
        if stop in response:
            response = response[:response.index(stop)]
    return response.strip()


# ─────────────────────────────────────────────
# CLI mode
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Ask the Bhagavad Gita model")
parser.add_argument("--question",   type=str,   default=None)
parser.add_argument("--max-tokens", type=int,   default=350)
parser.add_argument("--temperature", type=float, default=0.1)
args = parser.parse_args()

if args.question:
    print(f"Q: {args.question}\n")
    print(ask_gita(args.question, args.max_tokens, args.temperature))
    exit(0)

# ─────────────────────────────────────────────
# Demo questions
# ─────────────────────────────────────────────
demo_questions = [
    "What is Atman according to the Bhagavad Gita?",
    "How did Arjuna feel when he saw his relatives arrayed for battle?",
    "What does Krishna say about desireless action (Nishkama Karma)?",
    "What are the three types of faith described in the Bhagavad Gita?",
    "What is the significance of the battlefield of Kurukshetra?",
]

print("=" * 65)
print("    🕉️   Bhagavad Gita — Fine-Tuned Model (MLX + LoRA)")
print("=" * 65)

for i, q in enumerate(demo_questions, 1):
    print(f"\n❓ [{i}] {q}")
    print("-" * 55)
    print(ask_gita(q))

# ─────────────────────────────────────────────
# Interactive loop
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("📿 Interactive mode — type your question, or 'quit' to exit.\n")

while True:
    try:
        question = input("Your question: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\n🙏 Namaste! Goodbye.")
            break
        print()
        print(ask_gita(question))
        print()
    except (KeyboardInterrupt, EOFError):
        print("\n\n🙏 Namaste! Goodbye.")
        break
