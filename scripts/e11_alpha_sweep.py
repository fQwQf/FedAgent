"""
E11: Loss-Power Sweep — Testing the Universal Theory

Theory predicts: w_k ∝ L_k^α where optimal α depends on gradient diversity.
  - α=0: uniform (FedAvg baseline)
  - α=1: loss-proportional (E10 method)
  - α=1.5: predicted near-optimal for c̄ ≈ 0.3-0.5
  - α=2: theoretical optimum under orthogonal gradients

This experiment tests α ∈ {0, 1, 1.5, 2} for 10 rounds each.

Usage:
    python scripts/e11_alpha_sweep.py --device cuda:4
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import load_agentgym_data, get_trainable_messages
from src.aggregation import _weighted_average_state_dicts

ALL_ENVS = ["babyai", "webshop", "textcraft", "maze", "wordle"]
HARD_ENVS = ["webshop", "textcraft", "wordle"]
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_RANK = 8
LORA_ALPHA = 16
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4
LR = 5e-5
NUM_ROUNDS = 10
SEED = 42
TRAIN_SAMPLES = 64
EVAL_SAMPLES = 32
ALPHA_VALUES = [0, 1, 1.5, 2]


def set_seed(seed):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM", bias="none"))
    return model, tokenizer


def get_lora_state(model):
    return OrderedDict({
        k: v.clone().cpu() for k, v in model.state_dict().items() if "lora_" in k})


def apply_lora_state(model, state, device):
    model.load_state_dict(
        OrderedDict({k: v.to(device) for k, v in state.items()}), strict=False)


def make_data(tokenizer):
    train_dl = {}
    eval_dl = {}
    for env in ALL_ENVS:
        data = load_agentgym_data(env, max_samples=TRAIN_SAMPLES + EVAL_SAMPLES, seed=SEED)
        for split, start, end, store in [
            ("train", 0, TRAIN_SAMPLES, train_dl),
            ("eval", TRAIN_SAMPLES, TRAIN_SAMPLES + EVAL_SAMPLES, eval_dl)
        ]:
            pts = []
            for traj in data[start:end]:
                msgs, _ = get_trainable_messages(traj)
                if not msgs:
                    continue
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                if not text.strip():
                    continue
                tok = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH,
                                padding="max_length", return_tensors="pt")
                labels = tok["input_ids"].clone()
                labels[labels == tokenizer.pad_token_id] = -100
                pts.append((tok["input_ids"][0], tok["attention_mask"][0], labels[0]))
            if pts:
                ids = torch.stack([p[0] for p in pts])
                masks = torch.stack([p[1] for p in pts])
                labels = torch.stack([p[2] for p in pts])
                store[env] = DataLoader(TensorDataset(ids, masks, labels),
                                        batch_size=BATCH_SIZE, shuffle=(split == "train"))
    return train_dl, eval_dl


def train_client(model, dataloader, device):
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    for batch in dataloader:
        ids, mask, labels = [b.to(device) for b in batch]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        opt.zero_grad()
        loss.backward()
        opt.step()


def evaluate(model, dataloader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            ids, mask, labels = [b.to(device) for b in batch]
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                total += model(input_ids=ids, attention_mask=mask, labels=labels).loss.item() * ids.size(0)
            n += ids.size(0)
    return total / max(n, 1)


def cosine_sim(v1, v2):
    dot = sum(torch.dot(v1[k].flatten().float(), v2[k].flatten().float()) for k in v1)
    n1 = sum(v1[k].float().norm()**2 for k in v1)**0.5
    n2 = sum(v2[k].float().norm()**2 for k in v2)**0.5
    return (dot / (n1 * n2 + 1e-10)).item()


def delta_norm(delta):
    return float(sum(v.float().norm()**2 for v in delta.values())**0.5)


def compute_weights(eval_losses, alpha, env_list):
    """Compute aggregation weights: w_k ∝ L_k^alpha, normalized."""
    if alpha == 0:
        return {e: 1.0 / len(env_list) for e in env_list}
    powers = {e: eval_losses[e] ** alpha for e in env_list}
    total = sum(powers.values())
    return {e: powers[e] / total for e in env_list}


def run_alpha(device, train_dl, eval_dl, alpha, num_rounds):
    method_name = f"alpha_{alpha}"
    print(f"\n{'='*60}")
    print(f"  {method_name}  (w_k ∝ L_k^{alpha}, {num_rounds}R)")
    print(f"{'='*60}")

    model, _ = load_model_and_tokenizer(device)
    global_state = get_lora_state(model)
    metrics = []

    for rnd in range(num_rounds):
        t0 = time.time()
        rd = {"round": rnd, "alpha": alpha}

        # Eval first (for weighting and logging)
        apply_lora_state(model, global_state, device)
        eval_losses = {}
        for env in ALL_ENVS:
            eval_losses[env] = evaluate(model, eval_dl[env], device)
            rd[f"{env}_eval_loss"] = eval_losses[env]

        # Compute weights: w_k ∝ L_k^alpha
        weights = compute_weights(eval_losses, alpha, ALL_ENVS)

        # Client updates
        deltas = {}
        for env in ALL_ENVS:
            apply_lora_state(model, global_state, device)
            train_client(model, train_dl[env], device)
            new_state = get_lora_state(model)
            deltas[env] = OrderedDict({
                k: new_state[k].float() - global_state[k].float() for k in global_state})

        # Aggregate
        weight_list = [weights[e] for e in ALL_ENVS]
        delta_list = [deltas[e] for e in ALL_ENVS]
        avg_delta = _weighted_average_state_dicts(delta_list, weight_list)

        # Signal retention: cos(Δ_k, Δ_avg) for each env
        for env in ALL_ENVS:
            ret = cosine_sim(deltas[env], avg_delta)
            rd[f"{env}_retention"] = ret
            rd[f"{env}_delta_norm"] = delta_norm(deltas[env])

        # Update global state
        global_state = OrderedDict({k: global_state[k] + avg_delta[k] for k in global_state})
        apply_lora_state(model, global_state, device)

        # Final eval (after aggregation)
        for env in ALL_ENVS:
            rd[f"{env}_eval_after"] = evaluate(model, eval_dl[env], device)

        hard_avg = sum(rd[f"{e}_eval_after"] for e in HARD_ENVS) / len(HARD_ENVS)
        all_avg = sum(rd[f"{e}_eval_after"] for e in ALL_ENVS) / len(ALL_ENVS)
        rd["hard_avg"] = hard_avg
        rd["avg_eval_loss"] = all_avg
        rd["time_sec"] = time.time() - t0
        metrics.append(rd)

        # Per-round output
        loss_str = "  ".join(f"{e[:3]}={rd[f'{e}_eval_after']:.3f}" for e in ALL_ENVS)
        ret_str = "  ".join(f"ret_{e[:3]}={rd[f'{e}_retention']:+.3f}" for e in ALL_ENVS)
        wt_str = "  ".join(f"w_{e[:3]}={weights[e]:.2f}" for e in ALL_ENVS)
        print(f"  R{rnd:>2}: {loss_str}  hard={hard_avg:.4f}  |  {ret_str}")
        print(f"        weights: {wt_str}")

    del model
    torch.cuda.empty_cache()
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--alphas", default="all",
                        help="Comma-separated alpha values, or 'all'")
    args = parser.parse_args()
    device = args.device
    set_seed(SEED)

    if args.alphas == "all":
        alphas = ALPHA_VALUES
    else:
        alphas = [float(a) for a in args.alphas.split(",")]

    print("=" * 60)
    print(f"  E11: Loss-Power Sweep (α ∈ {{{', '.join(str(a) for a in alphas)}}})")
    print("=" * 60)
    print()
    print("  Theory: w_k* ∝ L_k^α where:")
    print("    α=0:   uniform (FedAvg)")
    print("    α=1:   loss-proportional")
    print("    α=1.5: predicted optimal (c̄ ≈ 0.3-0.5)")
    print("    α=2:   theoretical optimum (c̄ → 0)")
    print()

    _, tokenizer = load_model_and_tokenizer(device)
    torch.cuda.empty_cache()
    print("  Tokenizing data...")
    train_dl, eval_dl = make_data(tokenizer)
    del tokenizer

    all_results = {}
    for alpha in alphas:
        all_results[alpha] = run_alpha(device, train_dl, eval_dl, alpha, args.rounds)
        save_dir = Path(f"outputs/e11_alpha_sweep/alpha_{alpha}")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(all_results[alpha], f, indent=2)

    # ============================================================
    # Final comparison
    # ============================================================
    print(f"\n{'='*100}")
    print("  E11 CONVERGENCE COMPARISON")
    print(f"{'='*100}")

    for env in ALL_ENVS + ["hard_avg", "avg_eval_loss"]:
        label = env if env in ALL_ENVS else env.replace("_", " ").upper()
        print(f"\n  {label}:")
        for alpha in alphas:
            vals = [r.get(f"{env}_eval_after" if env in ALL_ENVS else env,
                          r.get(f"{env}", None)) for r in all_results[alpha]]
            print(f"    α={alpha:<4}: " + " → ".join(f"{v:.3f}" if v else "?" for v in vals))

    # Signal retention evolution for hard envs
    print(f"\n{'='*100}")
    print("  SIGNAL RETENTION EVOLUTION (hard envs)")
    print(f"{'='*100}")

    for env in HARD_ENVS:
        print(f"\n  {env} retention:")
        for alpha in alphas:
            rets = [r[f"{env}_retention"] for r in all_results[alpha]]
            trend = "↑" if rets[-1] > rets[0] else "↓"
            print(f"    α={alpha:<4}: " + " → ".join(f"{r:+.3f}" for r in rets) +
                  f"  ({trend} {rets[0]:+.3f}→{rets[-1]:+.3f})")

    # Summary table
    print(f"\n{'='*100}")
    print("  SUMMARY: Final (R9) Results")
    print(f"{'='*100}")
    print(f"\n  {'α':>4} | {'hard_avg':>10} | {'webshop':>10} | {'textcraft':>10} | {'wordle':>10} | {'babyai':>10} | {'maze':>10}")
    print("  " + "-" * 75)
    for alpha in alphas:
        r = all_results[alpha][-1]
        vals = [r["hard_avg"]] + [r[f"{e}_eval_after"] for e in ALL_ENVS]
        print(f"  {alpha:>4} | " + " | ".join(f"{v:>10.4f}" for v in vals))

    # Key: does α=1.5 or α=2 outperform α=1?
    print(f"\n  vs α=1 (loss-proportional):")
    a1_hard = all_results[1.0][-1]["hard_avg"] if 1.0 in all_results else None
    if a1_hard:
        for alpha in alphas:
            if alpha == 1.0:
                continue
            ah = all_results[alpha][-1]["hard_avg"]
            delta = (a1_hard - ah) / a1_hard * 100
            better = "✓ BETTER" if ah < a1_hard else "✗ worse"
            print(f"    α={alpha}: hard_avg={ah:.4f} vs {a1_hard:.4f} → {delta:+.1f}% {better}")


if __name__ == "__main__":
    main()
