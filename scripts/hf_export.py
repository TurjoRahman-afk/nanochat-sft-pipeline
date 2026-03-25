"""
Export a nanochat checkpoint to HuggingFace PreTrainedModel format.

This creates a self-contained directory with:
  config.json          -- NanoChatConfig (all hyperparams)
  model.safetensors    -- model weights (or pytorch_model.bin as fallback)
  generation_config.json

The exported model can then be loaded with:
  from scripts.hf_export import NanoChatConfig, NanoChatForCausalLM
  model = NanoChatForCausalLM.from_pretrained("./hf_exports/d12")

Usage:
  python -m scripts.hf_export                          # exports best base model
  python -m scripts.hf_export --source sft             # exports SFT model
  python -m scripts.hf_export --source base --model-tag d12 --output-dir hf_exports/d12
"""

import argparse
import os
import json
import torch

parser = argparse.ArgumentParser(description="Export nanochat checkpoint to HuggingFace format")
parser.add_argument("--source",     type=str, default="base", help="Model source: base|sft|rl")
parser.add_argument("--model-tag",  type=str, default=None,   help="Model tag (e.g. d12)")
parser.add_argument("--step",       type=int, default=None,   help="Checkpoint step")
parser.add_argument("--device-type",type=str, default="cpu",  help="Device to load model on (cpu recommended for export)")
parser.add_argument("--output-dir", type=str, default=None,   help="Output directory (default: hf_exports/<model_tag>)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
from transformers import PretrainedConfig, PreTrainedModel, GenerationConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
import torch.nn as nn

from nanochat.common import compute_init, autodetect_device_type, get_base_dir
from nanochat.checkpoint_manager import load_model, find_largest_model, find_last_step
from nanochat.gpt import GPT, GPTConfig

# ---------------------------------------------------------------------------
# 1. NanoChatConfig — stores all GPTConfig fields as a HF PretrainedConfig
# ---------------------------------------------------------------------------

class NanoChatConfig(PretrainedConfig):
    model_type = "nanochat"

    def __init__(
        self,
        sequence_len: int   = 2048,
        vocab_size: int     = 32768,
        n_layer: int        = 12,
        n_head: int         = 6,
        n_kv_head: int      = 6,
        n_embd: int         = 768,
        window_pattern: str = "SSSL",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sequence_len   = sequence_len
        self.vocab_size     = vocab_size
        self.n_layer        = n_layer
        self.n_head         = n_head
        self.n_kv_head      = n_kv_head
        self.n_embd         = n_embd
        self.window_pattern = window_pattern
        # Required by HF for text generation
        self.max_position_embeddings = sequence_len
        self.hidden_size             = n_embd

# ---------------------------------------------------------------------------
# 2. NanoChatForCausalLM — wraps GPT as a HuggingFace PreTrainedModel
# ---------------------------------------------------------------------------

class NanoChatForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NanoChatConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _tied_weights_keys = []

    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config: NanoChatConfig):
        super().__init__(config)
        gpt_config = GPTConfig(
            sequence_len   = config.sequence_len,
            vocab_size     = config.vocab_size,
            n_layer        = config.n_layer,
            n_head         = config.n_head,
            n_kv_head      = config.n_kv_head,
            n_embd         = config.n_embd,
            window_pattern = config.window_pattern,
        )
        with torch.device("meta"):
            self.model = GPT(gpt_config)

    def _init_weights(self, module):
        # nanochat uses its own init_weights(); skip HF's default init
        pass

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor = None,
        **kwargs,
    ) -> CausalLMOutput:
        if labels is not None:
            loss = self.model(input_ids, targets=labels)
            return CausalLMOutput(loss=loss, logits=None)
        else:
            logits = self.model(input_ids)
            return CausalLMOutput(logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

# ---------------------------------------------------------------------------
# 3. Load nanochat checkpoint
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("  nanochat → HuggingFace Export")
print(f"{'='*60}")
print(f"  Source      : {args.source}")

# Resolve checkpoint path
model_dir_map = {"base": "base_checkpoints", "sft": "chatsft_checkpoints", "rl": "chatrl_checkpoints"}
base_dir = get_base_dir()
checkpoints_dir = os.path.join(base_dir, model_dir_map[args.source])
model_tag = args.model_tag or find_largest_model(checkpoints_dir)
checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
step = args.step or find_last_step(checkpoint_dir)
print(f"  Model tag   : {model_tag}")
print(f"  Step        : {step}")
print(f"  Checkpoint  : {checkpoint_dir}")

# Load via nanochat's loader (handles all patching)
device = torch.device(args.device_type)
print(f"  Device      : {device}\n")
nanochat_model, tokenizer, meta = load_model(
    args.source, device, phase="eval",
    model_tag=model_tag, step=step,
)
model_cfg = meta["model_config"]

# ---------------------------------------------------------------------------
# 4. Build HF wrapper and transfer weights
# ---------------------------------------------------------------------------

hf_config = NanoChatConfig(
    sequence_len   = model_cfg["sequence_len"],
    vocab_size     = model_cfg["vocab_size"],
    n_layer        = model_cfg["n_layer"],
    n_head         = model_cfg["n_head"],
    n_kv_head      = model_cfg["n_kv_head"],
    n_embd         = model_cfg["n_embd"],
    window_pattern = model_cfg["window_pattern"],
    # HF metadata
    architectures  = ["NanoChatForCausalLM"],
    model_type     = "nanochat",
)

hf_model = NanoChatForCausalLM(hf_config)
# Materialise from meta device then copy weights from the nanochat model
hf_model.model.to_empty(device=device)
hf_model.model.init_weights()
hf_model.model.load_state_dict(nanochat_model.state_dict(), strict=True, assign=True)
hf_model.eval()

total_params = sum(p.numel() for p in hf_model.parameters())
print(f"  Parameters  : {total_params:,}")

# ---------------------------------------------------------------------------
# 5. Save to output directory
# ---------------------------------------------------------------------------

output_dir = args.output_dir or os.path.join("hf_exports", f"{model_tag}_{args.source}_step{step}")
os.makedirs(output_dir, exist_ok=True)

print(f"\n  Saving to   : {output_dir}")
hf_model.save_pretrained(output_dir, safe_serialization=True)
print(f"  ✓ Model weights saved")

# Save generation config
gen_config = GenerationConfig(
    max_new_tokens        = 256,
    temperature           = 0.6,
    top_k                 = 50,
    do_sample             = True,
    bos_token_id          = tokenizer.get_bos_token_id(),
    eos_token_id          = tokenizer.get_bos_token_id(),  # nanochat uses BOS as stop
    pad_token_id          = tokenizer.get_bos_token_id(),
)
gen_config.save_pretrained(output_dir)
print(f"  ✓ Generation config saved")

# Save nanochat meta (training info) as extra JSON
meta_export = {
    "source"    : args.source,
    "model_tag" : model_tag,
    "step"      : step,
    "val_bpb"   : meta.get("val_bpb"),
    "total_params": total_params,
    "model_config": model_cfg,
    "exported_by" : "scripts/hf_export.py",
}
with open(os.path.join(output_dir, "nanochat_meta.json"), "w") as f:
    json.dump(meta_export, f, indent=2)
print(f"  ✓ Training metadata saved")

# Report exported file sizes
print(f"\n  Exported files:")
total_size = 0
for fname in sorted(os.listdir(output_dir)):
    fpath = os.path.join(output_dir, fname)
    size_mb = os.path.getsize(fpath) / 1024 / 1024
    total_size += size_mb
    print(f"    {fname:<35} {size_mb:>8.1f} MB")
print(f"    {'TOTAL':<35} {total_size:>8.1f} MB")

# ---------------------------------------------------------------------------
# 6. Sanity check — load back and run a forward pass
# ---------------------------------------------------------------------------

print(f"\n  Sanity check — loading back from {output_dir} ...")
loaded = NanoChatForCausalLM.from_pretrained(output_dir, config=hf_config)
loaded.eval()

bos = tokenizer.get_bos_token_id()
dummy = torch.tensor([[bos, bos + 1, bos + 2]], dtype=torch.long)
with torch.no_grad():
    out = loaded(dummy)
assert out.logits is not None and out.logits.shape == (1, 3, model_cfg["vocab_size"]), \
    f"Unexpected logits shape: {out.logits.shape}"
print(f"  ✓ Forward pass OK  →  logits shape {tuple(out.logits.shape)}")

print(f"\n  Export complete!\n")
print(f"  Load later with:")
print(f"    from scripts.hf_export import NanoChatConfig, NanoChatForCausalLM")
print(f"    model = NanoChatForCausalLM.from_pretrained(\"{output_dir}\")")
print()
