"""
E0: Gradient Consistency Probe for FedRNK
Verifies P1: ||Δ^{Attn}||² vs ||Δ^{MLP}||²

Collects LoRA gradients from each environment, computes cross-environment
deviation norms for Attention vs MLP parameters.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import OrderedDict
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import load_agentgym_data, get_trainable_messages
from src.model import get_lora_gradients, flatten_gradients

ENV_NAMES = ["babyai", "webshop", "textcraft", "maze", "wordle"]
NUM_ACCUMULATION = 16
MAX_SEQ_LENGTH = 1024
LORA_RANK = 8
LORA_ALPHA = 16
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model_and_tokenizer(device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    if "7B" in MODEL_NAME:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def collect_gradients_for_env(env_name: str, model, tokenizer, device: str):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "model": {
            "lora": {
                "attn_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "mlp_modules": ["gate_proj", "up_proj", "down_proj"],
            }
        }
    })

    data = load_agentgym_data(env_name, max_samples=NUM_ACCUMULATION + 5)
    model.train()

    accumulated_attn = None
    accumulated_mlp = None
    total_loss = 0.0
    num_steps = 0

    for i, traj in enumerate(data[:NUM_ACCUMULATION]):
        messages, labels_mask = get_trainable_messages(traj)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        if not text.strip():
            continue

        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=MAX_SEQ_LENGTH, padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        labels = inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        model.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()

        attn_grads, mlp_grads = get_lora_gradients(model, cfg)
        attn_flat = flatten_gradients(attn_grads).cpu()
        mlp_flat = flatten_gradients(mlp_grads).cpu()

        if accumulated_attn is None:
            accumulated_attn = attn_flat
            accumulated_mlp = mlp_flat
        else:
            accumulated_attn += attn_flat
            accumulated_mlp += mlp_flat

        total_loss += loss.item()
        num_steps += 1

    model.zero_grad()

    if num_steps > 0:
        accumulated_attn /= num_steps
        accumulated_mlp /= num_steps

    return {
        "env": env_name,
        "loss": total_loss / max(num_steps, 1),
        "num_steps": num_steps,
        "attn_grad": accumulated_attn,
        "mlp_grad": accumulated_mlp,
        "attn_numel": accumulated_attn.numel() if accumulated_attn is not None else 0,
        "mlp_numel": accumulated_mlp.numel() if accumulated_mlp is not None else 0,
    }


def analyze_gradients(env_results: dict) -> dict:
    envs = list(env_results.keys())
    n = len(envs)

    attn_grads = {k: v["attn_grad"] for k, v in env_results.items()}
    mlp_grads = {k: v["mlp_grad"] for k, v in env_results.items()}

    global_attn = torch.stack([attn_grads[e] for e in envs]).mean(dim=0)
    global_mlp = torch.stack([mlp_grads[e] for e in envs]).mean(dim=0)

    attn_dev_sq = {}
    mlp_dev_sq = {}
    for e in envs:
        attn_dev_sq[e] = (attn_grads[e] - global_attn).norm().item() ** 2
        mlp_dev_sq[e] = (mlp_grads[e] - global_mlp).norm().item() ** 2

    mean_attn_dev_sq = sum(attn_dev_sq.values()) / n
    mean_mlp_dev_sq = sum(mlp_dev_sq.values()) / n

    ratio = mean_attn_dev_sq / mean_mlp_dev_sq if mean_mlp_dev_sq > 0 else float("inf")

    attn_cos = torch.zeros(n, n)
    mlp_cos = torch.zeros(n, n)
    for i, ei in enumerate(envs):
        for j, ej in enumerate(envs):
            if i == j:
                attn_cos[i, j] = 1.0
                mlp_cos[i, j] = 1.0
            else:
                attn_cos[i, j] = F.cosine_similarity(
                    attn_grads[ei].unsqueeze(0), attn_grads[ej].unsqueeze(0)
                ).item()
                mlp_cos[i, j] = F.cosine_similarity(
                    mlp_grads[ei].unsqueeze(0), mlp_grads[ej].unsqueeze(0)
                ).item()

    triu_idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    attn_avg_cos = sum(attn_cos[i, j] for i, j in triu_idx) / len(triu_idx)
    mlp_avg_cos = sum(mlp_cos[i, j] for i, j in triu_idx) / len(triu_idx)

    return {
        "mean_attn_dev_sq": mean_attn_dev_sq,
        "mean_mlp_dev_sq": mean_mlp_dev_sq,
        "ratio": ratio,
        "h1_supported": ratio < 0.5,
        "attn_avg_cosine": attn_avg_cos.item(),
        "mlp_avg_cosine": mlp_avg_cos.item(),
        "per_env_attn_dev_sq": attn_dev_sq,
        "per_env_mlp_dev_sq": mlp_dev_sq,
        "attn_cosine_matrix": attn_cos.tolist(),
        "mlp_cosine_matrix": mlp_cos.tolist(),
        "env_names": envs,
    }


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    print(f"Using device: {device}")

    print("=" * 60)
    print("  FedRNK E0: Gradient Consistency Probe")
    print("  Model:", MODEL_NAME)
    print("  LoRA rank:", LORA_RANK)
    print("  Accumulation steps:", NUM_ACCUMULATION)
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer(device)

    env_results = {}
    for env_name in ENV_NAMES:
        print(f"\n  Processing {env_name}...")
        t0 = time.time()
        result = collect_gradients_for_env(env_name, model, tokenizer, device)
        elapsed = time.time() - t0
        print(f"    loss={result['loss']:.4f}  "
              f"attn_params={result['attn_numel']}  "
              f"mlp_params={result['mlp_numel']}  "
              f"time={elapsed:.1f}s")
        env_results[env_name] = result

    print("\n" + "=" * 60)
    print("  ANALYSIS")
    print("=" * 60)

    analysis = analyze_gradients(env_results)

    print(f"\n  Mean ||Δ^attn||² = {analysis['mean_attn_dev_sq']:.6f}")
    print(f"  Mean ||Δ^mlp||²  = {analysis['mean_mlp_dev_sq']:.6f}")
    print(f"  Ratio (attn/mlp) = {analysis['ratio']:.4f}")
    print(f"  H1 supported (< 0.5)? {analysis['h1_supported']}")
    print(f"\n  Avg cosine sim - Attn: {analysis['attn_avg_cosine']:.4f}")
    print(f"  Avg cosine sim - MLP:  {analysis['mlp_avg_cosine']:.4f}")

    print(f"\n  Per-env deviation norms:")
    print(f"  {'Env':>12} | {'||Δ^attn||²':>14} | {'||Δ^mlp||²':>14} | {'Ratio':>8}")
    print(f"  {'-'*56}")
    for e in analysis["env_names"]:
        a = analysis["per_env_attn_dev_sq"][e]
        m = analysis["per_env_mlp_dev_sq"][e]
        r = a / m if m > 0 else float("inf")
        print(f"  {e:>12} | {a:>14.6f} | {m:>14.6f} | {r:>8.4f}")

    print(f"\n  Cross-env cosine similarity (Attention):")
    names = analysis["env_names"]
    header = f"  {'':>12}" + "".join(f" {n:>8}" for n in names)
    print(header)
    for i, ni in enumerate(names):
        row = f"  {ni:>12}"
        for j in range(len(names)):
            row += f" {analysis['attn_cosine_matrix'][i][j]:>8.4f}"
        print(row)

    print(f"\n  Cross-env cosine similarity (MLP):")
    print(header)
    for i, ni in enumerate(names):
        row = f"  {ni:>12}"
        for j in range(len(names)):
            row += f" {analysis['mlp_cosine_matrix'][i][j]:>8.4f}"
        print(row)

    save_dir = Path("outputs/e0_results")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_data = {
        "analysis": {k: v for k, v in analysis.items()
                     if k not in ["attn_cosine_matrix", "mlp_cosine_matrix", "per_env_attn_dev_sq", "per_env_mlp_dev_sq"]},
        "per_env": {
            e: {"loss": env_results[e]["loss"], "attn_numel": env_results[e]["attn_numel"], "mlp_numel": env_results[e]["mlp_numel"]}
            for e in ENV_NAMES
        },
    }
    with open(save_dir / "e0_summary.json", "w") as f:
        json.dump(save_data, f, indent=2)

    torch.save(env_results, save_dir / "e0_gradients.pt")

    print(f"\n  Results saved to {save_dir}/")
    print("=" * 60)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
