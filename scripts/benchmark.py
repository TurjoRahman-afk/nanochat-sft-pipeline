"""
Benchmark script for nanochat inference performance.
Measures latency, throughput, memory usage and outputs a report with graphs.

Usage:
    python -m scripts.benchmark
    python -m scripts.benchmark --source sft --max-tokens 200 --num-requests 50
"""

import argparse
import time
import os
import json
import torch
import statistics
from datetime import datetime

parser = argparse.ArgumentParser(description="Benchmark nanochat inference performance")
parser.add_argument("--source", type=str, default="sft", help="Model source: sft|base|rl")
parser.add_argument("--model-tag", type=str, default=None, help="Model tag to load")
parser.add_argument("--step", type=int, default=None, help="Checkpoint step to load")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate per request")
parser.add_argument("--num-requests", type=int, default=30, help="Number of requests to benchmark")
parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Directory to save results")
args = parser.parse_args()

# -------------------------------------------------------------------------
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
engine = Engine(model, tokenizer)

bos = tokenizer.get_bos_token_id()
user_start  = tokenizer.encode_special("<|user_start|>")
user_end    = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")

# -------------------------------------------------------------------------
# Benchmark prompts — varied length and topic
PROMPTS = [
    "What is 2 + 2?",
    "Explain what a neural network is in one sentence.",
    "What is the capital of France?",
    "Write a short poem about the ocean.",
    "What is the speed of light?",
    "How many days are in a leap year?",
    "What is the largest planet in the solar system?",
    "Explain recursion in programming.",
    "What is the boiling point of water in Celsius?",
    "Name three programming languages.",
]

def build_prompt_tokens(prompt_text: str) -> list[int]:
    """Build the token sequence for a single-turn prompt."""
    tokens = [bos, user_start]
    tokens.extend(tokenizer.encode(prompt_text))
    tokens.extend([user_end, assistant_start])
    return tokens

def run_single_request(prompt_tokens: list[int]) -> tuple[int, float]:
    """Run one generation request. Returns (num_tokens_generated, elapsed_seconds)."""
    t0 = time.perf_counter()
    tokens_generated = 0
    for token_column, _ in engine.generate(
        prompt_tokens,
        num_samples=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    ):
        tokens_generated += 1
    elapsed = time.perf_counter() - t0
    return tokens_generated, elapsed

# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  nanochat Inference Benchmark")
print("=" * 60)
print(f"  Model source  : {args.source}")
print(f"  Device        : {device}")
print(f"  Max tokens    : {args.max_tokens}")
print(f"  Num requests  : {args.num_requests}")
print(f"  Temperature   : {args.temperature}")
print(f"  Top-k         : {args.top_k}")
model_params = sum(p.numel() for p in model.parameters())
print(f"  Model params  : {model_params:,}")
if device_type == "cuda":
    print(f"  GPU           : {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()
print("=" * 60)

# Warm-up run (not counted)
print("\nWarming up... ", end="", flush=True)
warmup_tokens = build_prompt_tokens("Hello")
run_single_request(warmup_tokens)
if device_type == "cuda":
    torch.cuda.synchronize()
print("done.\n")

# -------------------------------------------------------------------------
# Main benchmark loop
latencies_ms   = []   # ms per request
tokens_counts  = []   # tokens generated per request
tok_per_sec_list = [] # tokens/sec per request

print(f"{'Req':>4}  {'Prompt':<40}  {'Tokens':>6}  {'Time(ms)':>9}  {'tok/s':>8}")
print("-" * 75)

for i in range(args.num_requests):
    prompt_text = PROMPTS[i % len(PROMPTS)]
    prompt_tokens = build_prompt_tokens(prompt_text)

    n_tokens, elapsed = run_single_request(prompt_tokens)
    if device_type == "cuda":
        torch.cuda.synchronize()

    latency_ms = elapsed * 1000
    tps = n_tokens / elapsed if elapsed > 0 else 0

    latencies_ms.append(latency_ms)
    tokens_counts.append(n_tokens)
    tok_per_sec_list.append(tps)

    prompt_display = (prompt_text[:37] + "...") if len(prompt_text) > 40 else prompt_text
    print(f"{i+1:>4}  {prompt_display:<40}  {n_tokens:>6}  {latency_ms:>9.1f}  {tps:>8.1f}")

if device_type == "cuda":
    torch.cuda.synchronize()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
else:
    peak_vram_mb = 0.0

# -------------------------------------------------------------------------
# Statistics
print("\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)

avg_latency   = statistics.mean(latencies_ms)
p50_latency   = statistics.median(latencies_ms)
p95_latency   = sorted(latencies_ms)[int(0.95 * len(latencies_ms))]
p99_latency   = sorted(latencies_ms)[int(0.99 * len(latencies_ms))]
min_latency   = min(latencies_ms)
max_latency   = max(latencies_ms)

avg_tps       = statistics.mean(tok_per_sec_list)
peak_tps      = max(tok_per_sec_list)
avg_tokens    = statistics.mean(tokens_counts)

total_tokens  = sum(tokens_counts)
total_time_s  = sum(l / 1000 for l in latencies_ms)
throughput    = total_tokens / total_time_s  # tokens/sec across all requests

print(f"\n  Latency (ms per request):")
print(f"    Average  : {avg_latency:>8.1f} ms")
print(f"    Median   : {p50_latency:>8.1f} ms")
print(f"    P95      : {p95_latency:>8.1f} ms")
print(f"    P99      : {p99_latency:>8.1f} ms")
print(f"    Min      : {min_latency:>8.1f} ms")
print(f"    Max      : {max_latency:>8.1f} ms")

print(f"\n  Throughput:")
print(f"    Avg tok/sec (per req) : {avg_tps:>8.1f}")
print(f"    Peak tok/sec          : {peak_tps:>8.1f}")
print(f"    Total throughput      : {throughput:>8.1f} tok/sec")
print(f"    Avg tokens generated  : {avg_tokens:>8.1f}")
print(f"    Total tokens          : {total_tokens:>8,}")

print(f"\n  Memory:")
if device_type == "cuda":
    print(f"    Peak VRAM usage : {peak_vram_mb:>8.1f} MB")
else:
    print(f"    Peak VRAM usage : N/A (CPU mode)")

print("=" * 60)

# -------------------------------------------------------------------------
# Save JSON results
os.makedirs(args.output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results = {
    "timestamp": timestamp,
    "config": {
        "source": args.source,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device_type == "cuda" else "cpu",
        "max_tokens": args.max_tokens,
        "num_requests": args.num_requests,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "model_params": model_params,
    },
    "latency_ms": {
        "avg": avg_latency, "p50": p50_latency,
        "p95": p95_latency, "p99": p99_latency,
        "min": min_latency, "max": max_latency,
    },
    "throughput": {
        "avg_tok_per_sec": avg_tps,
        "peak_tok_per_sec": peak_tps,
        "total_throughput": throughput,
        "total_tokens": total_tokens,
    },
    "memory": {
        "peak_vram_mb": peak_vram_mb,
    },
    "raw": {
        "latencies_ms": latencies_ms,
        "tokens_counts": tokens_counts,
        "tok_per_sec": tok_per_sec_list,
    }
}
json_path = os.path.join(args.output_dir, f"bench_{timestamp}.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to: {json_path}")

# -------------------------------------------------------------------------
# Plot graphs (optional — skipped gracefully if matplotlib not installed)
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — works without a display
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"nanochat Benchmark — {timestamp}", fontsize=12)

    # 1. Latency per request
    axes[0].plot(latencies_ms, color="steelblue", linewidth=1.5)
    axes[0].axhline(avg_latency, color="red", linestyle="--", label=f"avg {avg_latency:.0f}ms")
    axes[0].axhline(p95_latency, color="orange", linestyle="--", label=f"p95 {p95_latency:.0f}ms")
    axes[0].set_title("Latency per Request")
    axes[0].set_xlabel("Request #")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].legend(fontsize=8)

    # 2. Tokens per second per request
    axes[1].plot(tok_per_sec_list, color="green", linewidth=1.5)
    axes[1].axhline(avg_tps, color="red", linestyle="--", label=f"avg {avg_tps:.0f} tok/s")
    axes[1].set_title("Tokens / Second per Request")
    axes[1].set_xlabel("Request #")
    axes[1].set_ylabel("Tokens/sec")
    axes[1].legend(fontsize=8)

    # 3. Latency distribution histogram
    axes[2].hist(latencies_ms, bins=15, color="steelblue", edgecolor="white")
    axes[2].axvline(avg_latency, color="red",    linestyle="--", label=f"avg")
    axes[2].axvline(p95_latency, color="orange", linestyle="--", label=f"p95")
    axes[2].set_title("Latency Distribution")
    axes[2].set_xlabel("Latency (ms)")
    axes[2].set_ylabel("Count")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, f"bench_{timestamp}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Graph saved to   : {plot_path}")

except ImportError:
    print("\n  (matplotlib not installed — skipping graphs. Run: pip install matplotlib)")

print()
