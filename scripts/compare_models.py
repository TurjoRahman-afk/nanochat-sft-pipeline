"""
Compare base vs SFT model side-by-side using nanochat's native checkpoint loader.

Usage:
    python -m scripts.compare_models
    python -m scripts.compare_models --model-tag d12
    python -m scripts.compare_models --prompts "What is 2+2?" "Tell me a joke" "Write a haiku"
    python -m scripts.compare_models --max-tokens 200 --temperature 0.7 --device cuda

Models are loaded sequentially (base first, then SFT) to stay within VRAM budget.
"""

import argparse
import textwrap
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model-tag",   type=str,   default=None,   help="Model tag (e.g. d12)")
parser.add_argument("--base-step",   type=int,   default=None,   help="Base checkpoint step (default: latest)")
parser.add_argument("--sft-step",    type=int,   default=None,   help="SFT  checkpoint step (default: latest)")
parser.add_argument("--prompts",     type=str,   nargs="+",      default=None, help="Prompts to test")
parser.add_argument("--max-tokens",  type=int,   default=150)
parser.add_argument("--temperature", type=float, default=0.6)
parser.add_argument("--top-k",       type=int,   default=50)
parser.add_argument("--device-type", type=str,   default="",     help="cuda|cpu|mps (default: autodetect)")
parser.add_argument("--seed",        type=int,   default=42)
args = parser.parse_args()

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# ---------------------------------------------------------------------------
# Default prompts
# ---------------------------------------------------------------------------
DEFAULT_PROMPTS = [
    "Hi, how are you?",
    "What is 2 + 2? Explain your reasoning.",
    "Write a short poem about the ocean.",
    "What is the capital of France?",
    "Tell me a fun fact about penguins.",
    "Write a Python function that returns the Fibonacci sequence up to n.",
    "Explain gravity to a 5-year-old.",
    "What should I cook for dinner tonight?",
]

prompts = args.prompts if args.prompts else DEFAULT_PROMPTS

device_type = autodetect_device_type() if not args.device_type else args.device_type
_, _, _, _, device = compute_init(device_type)

# ---------------------------------------------------------------------------
# Generate one response using the Engine (same path as chat_cli.py)
# ---------------------------------------------------------------------------
def generate(engine: Engine, tokenizer, prompt_text: str) -> str:
    bos             = tokenizer.get_bos_token_id()
    user_start      = tokenizer.encode_special("<|user_start|>")
    user_end        = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end   = tokenizer.encode_special("<|assistant_end|>")

    tokens = [bos, user_start] + tokenizer.encode(prompt_text) + [user_end, assistant_start]

    output_tokens = []
    # engine.generate yields (token_column, token_masks) for num_samples=1
    for token_column, _ in engine.generate(
        tokens,
        num_samples=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    ):
        token = token_column[0]
        # Stop on assistant_end or bos (engine already marks complete but we break cleanly)
        if token == assistant_end or token == bos:
            break
        output_tokens.append(token)

    return tokenizer.decode(output_tokens).strip()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DIVIDER = "=" * 72

def wrap(text: str, width: int = 68, indent: str = "  ") -> str:
    if not text:
        return indent + "(empty)"
    lines = text.splitlines()
    out = []
    for line in lines:
        if line.strip() == "":
            out.append("")
        else:
            out.extend(textwrap.wrap(line, width=width, initial_indent=indent, subsequent_indent=indent))
    return "\n".join(out)

# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------
print(f"\n{DIVIDER}")
print("  nanochat Model Comparison: Base vs SFT")
print(f"{DIVIDER}")
print(f"  Model tag       : {args.model_tag or 'auto'}")
print(f"  Device          : {device}")
print(f"  Max new tokens  : {args.max_tokens}")
print(f"  Temperature     : {args.temperature}  |  Top-k: {args.top_k}")
print(f"  Seed            : {args.seed}")
print(f"  Prompts         : {len(prompts)}")

# --- Load base ---
print(f"\n  Loading base model ...")
base_model, base_tok, base_meta = load_model(
    "base", device, phase="eval",
    model_tag=args.model_tag, step=args.base_step,
)
base_engine = Engine(base_model, base_tok)
print(f"  ✓ Base loaded  (step {base_meta.get('step', '?')}, val_bpb {base_meta.get('val_bpb', 'N/A')})")

# --- Load SFT ---
print(f"  Loading SFT model ...")
sft_model, sft_tok, sft_meta = load_model(
    "sft", device, phase="eval",
    model_tag=args.model_tag, step=args.sft_step,
)
sft_engine = Engine(sft_model, sft_tok)
print(f"  ✓ SFT  loaded  (step {sft_meta.get('step', '?')}, val_bpb {sft_meta.get('val_bpb', 'N/A')})")
print(f"\n{DIVIDER}\n")

for i, prompt in enumerate(prompts, 1):
    print(f"{'─' * 72}")
    print(f"  [{i}/{len(prompts)}] {prompt}")
    print(f"{'─' * 72}")

    base_out = generate(base_engine, base_tok, prompt)
    sft_out  = generate(sft_engine,  sft_tok,  prompt)

    print(f"\n  [BASE]\n{wrap(base_out)}\n")
    print(f"  [SFT]\n{wrap(sft_out)}\n")

print(DIVIDER)
print("  Comparison complete.")
print(DIVIDER + "\n")
