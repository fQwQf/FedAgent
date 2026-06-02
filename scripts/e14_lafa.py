"""
E14: Layer-Adaptive Federated Aggregation (LAFA)

Core idea: Per-layer gradient alignment c̄_l automatically determines α_l.
  - Shared layers (high alignment, c̄_l ≈ 1): α_l ≈ 0 → uniform weighting
  - Task-specific layers (low alignment, c̄_l ≈ 0): α_l ≈ 1 → loss-proportional

Formula:
  w_k^l = L_k^(1 - c̄_l) / Σ_j L_j^(1 - c̄_l)

where c̄_l = mean pairwise cosine similarity of client updates at layer l.

Compares: uniform, loss_wt (global α=1), lafa (per-layer α_l = 1 - c̄_l)

Usage:
    python scripts/e14_lafa.py --model_size 0.5B --device cuda:3
    python scripts/e14_lafa.py --model_size 1.5B --device cuda:7
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

ALL_ENVS = ["babyai", "webshop", "textcraft", "maze", "wordle"]
HARD_ENVS = ["webshop", "textcraft", "wordle"]
NUM_ROUNDS = 20
SEED = 42
TRAIN_SAMPLES = 64
EVAL_SAMPLES = 32
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4
LR = 5e-5
LORA_RANK = 8
LORA_ALPHA = 16

MODEL_MAP = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "7B": "Qwen/Qwen2.5-7B-Instruct",
}


def set_seed(seed):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(device, model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16,
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
    """Cosine similarity between two state dicts."""
    dot = sum(torch.dot(v1[k].flatten().float(), v2[k].flatten().float()) for k in v1)
    n1 = sum(v1[k].float().norm()**2 for k in v1)**0.5
    n2 = sum(v2[k].float().norm()**2 for k in v2)**0.5
    return (dot / (n1 * n2 + 1e-10)).item()


def delta_norm(delta):
    return float(sum(v.float().norm()**2 for v in delta.values())**0.5)


# ============================================================
# Layer grouping utilities
# ============================================================

def get_layer_group_key(name):
    """Map a LoRA parameter name to a (layer_idx, module_type) group.
    Returns None for non-layer parameters (e.g., lora_embedding)."""
    parts = name.split(".")
    layer_idx = None
    for p in parts:
        if p.isdigit():
            layer_idx = int(p)
            break

    # Determine module type
    if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
        module = "attn"
    elif any(x in name for x in ["gate_proj", "up_proj", "down_proj"]):
        module = "ffn"
    else:
        return None  # skip non-standard params

    return (layer_idx, module)


def compute_per_layer_alignment(deltas, env_list):
    """Compute c̄_l = mean pairwise cosine similarity for each layer group.

    For each (layer_idx, module_type) group, we concatenate all LoRA A and B
    parameters in that group into a single vector per client, then compute
    pairwise cosine similarities.

    Returns: dict mapping (layer_idx, module_type) -> c̄_l (float)
    """
    # Group parameter keys by (layer_idx, module_type)
    groups = {}
    keys = list(deltas[env_list[0]].keys())
    for key in keys:
        gk = get_layer_group_key(key)
        if gk is not None:
            groups.setdefault(gk, []).append(key)

    alignment = {}
    for gk, param_keys in groups.items():
        # Build per-env vector for this group
        env_vecs = []
        for env in env_list:
            vec = torch.cat([deltas[env][k].flatten().float() for k in param_keys])
            env_vecs.append(vec)

        # Mean pairwise cosine similarity
        cos_sum = 0.0
        count = 0
        for i in range(len(env_vecs)):
            for j in range(i + 1, len(env_vecs)):
                cos = torch.dot(env_vecs[i], env_vecs[j]) / (
                    env_vecs[i].norm() * env_vecs[j].norm() + 1e-10)
                cos_sum += cos.item()
                count += 1
        alignment[gk] = cos_sum / max(count, 1)

    return alignment


def aggregate_lafa(deltas, eval_losses, env_list):
    """LAFA aggregation: per-layer adaptive α_l = 1 - c̄_l.

    For each layer group g with alignment c̄_g:
      α_g = 1 - c̄_g
      w_k^g = L_k^α_g / Σ_j L_j^α_g

    Then aggregate: Δ_g = Σ_k w_k^g · Δ_k^g
    """
    # 1. Compute per-layer alignment
    alignment = compute_per_layer_alignment(deltas, env_list)

    # 2. Compute per-layer α
    alphas = {gk: max(0.0, 1.0 - c) for gk, c in alignment.items()}

    # 3. Compute per-layer weights for each env
    losses = {env: max(eval_losses[env], 0.01) for env in env_list}  # floor to avoid 0^x issues

    per_layer_weights = {}
    for gk, alpha in alphas.items():
        w = {}
        for env in env_list:
            w[env] = losses[env] ** alpha
        total = sum(w.values())
        for env in env_list:
            w[env] /= total
        per_layer_weights[gk] = w

    # 4. Build aggregated delta with per-layer weighting
    result = OrderedDict()
    keys = list(deltas[env_list[0]].keys())

    for key in keys:
        gk = get_layer_group_key(key)

        if gk is not None and gk in per_layer_weights:
            # Per-layer weighted average
            weights = per_layer_weights[gk]
            tensors = [deltas[env][key].float() for env in env_list]
            w_list = [weights[env] for env in env_list]
            stacked = torch.stack(tensors)
            w_tensor = torch.tensor(w_list, dtype=torch.float32)
            w_shape = [-1] + [1] * (stacked.dim() - 1)
            result[key] = (stacked * w_tensor.view(w_shape)).sum(dim=0)
        else:
            # Fallback: uniform average for non-grouped params
            tensors = [deltas[env][key].float() for env in env_list]
            result[key] = torch.stack(tensors).mean(dim=0)

    return result, alphas, per_layer_weights


def aggregate_loss_wt(deltas, eval_losses, env_list):
    """Standard loss-proportional aggregation (global α=1)."""
    losses = {env: max(eval_losses[env], 0.01) for env in env_list}
    total = sum(losses.values())
    weights = [losses[env] / total for env in env_list]

    result = OrderedDict()
    keys = list(deltas[env_list[0]].keys())
    for key in keys:
        tensors = torch.stack([deltas[env][key].float() for env in env_list])
        w = torch.tensor(weights, dtype=torch.float32)
        w_shape = [-1] + [1] * (tensors.dim() - 1)
        result[key] = (tensors * w.view(w_shape)).sum(dim=0)
    return result, None, None


def aggregate_uniform(deltas, env_list):
    """Uniform FedAvg aggregation."""
    result = OrderedDict()
    keys = list(deltas[env_list[0]].keys())
    for key in keys:
        tensors = [deltas[env][key].float() for env in env_list]
        result[key] = torch.stack(tensors).mean(dim=0)
    return result, None, None


# ============================================================
# Main experiment loop
# ============================================================

def run_method(device, train_dl, eval_dl, method, num_rounds, model_name):
    is_lafa = (method == "lafa")
    is_loss_wt = (method == "loss_wt")

    print(f"\n{'='*70}")
    print(f"  {method}  ({num_rounds}R, {model_name})")
    print(f"{'='*70}")

    model, _ = load_model_and_tokenizer(device, model_name)
    global_state = get_lora_state(model)
    metrics = []

    for rnd in range(num_rounds):
        t0 = time.time()
        rd = {"round": rnd, "method": method}

        # Eval first
        apply_lora_state(model, global_state, device)
        eval_losses = {}
        for env in ALL_ENVS:
            eval_losses[env] = evaluate(model, eval_dl[env], device)
            rd[f"{env}_eval_loss"] = eval_losses[env]

        # Client updates
        deltas = {}
        for env in ALL_ENVS:
            apply_lora_state(model, global_state, device)
            train_client(model, train_dl[env], device)
            new_state = get_lora_state(model)
            deltas[env] = OrderedDict({
                k: new_state[k].float() - global_state[k].float() for k in global_state})

        # Aggregate based on method
        if is_lafa:
            avg_delta, alphas, per_layer_weights = aggregate_lafa(deltas, eval_losses, ALL_ENVS)
        elif is_loss_wt:
            avg_delta, alphas, per_layer_weights = aggregate_loss_wt(deltas, eval_losses, ALL_ENVS)
        else:
            avg_delta, alphas, per_layer_weights = aggregate_uniform(deltas, ALL_ENVS)

        # Signal retention
        for env in ALL_ENVS:
            ret = cosine_sim(deltas[env], avg_delta)
            rd[f"{env}_retention"] = ret
            rd[f"{env}_delta_norm"] = delta_norm(deltas[env])
        rd["avg_retention"] = np.mean([rd[f"{e}_retention"] for e in ALL_ENVS])

        # Store global weights (for uniform/loss_wt)
        if is_loss_wt:
            total_loss = sum(eval_losses[e] for e in ALL_ENVS)
            for env in ALL_ENVS:
                rd[f"{env}_weight"] = eval_losses[env] / total_loss
        else:
            for env in ALL_ENVS:
                rd[f"{env}_weight"] = 1.0 / len(ALL_ENVS)

        # Store per-layer α (LAFA only)
        if is_lafa and alphas is not None:
            alpha_dict = {}
            for (layer_idx, module), alpha in sorted(alphas.items()):
                key = f"L{layer_idx}_{module}"
                alpha_dict[key] = alpha
            rd["per_layer_alphas"] = alpha_dict

            # Store effective weights for a few representative layers
            if per_layer_weights:
                # Store weights for first attn layer and last ffn layer
                attn_keys = sorted([gk for gk in per_layer_weights if gk[1] == "attn"])
                ffn_keys = sorted([gk for gk in per_layer_weights if gk[1] == "ffn"])
                if attn_keys:
                    first_attn = attn_keys[0]
                    for env in ALL_ENVS:
                        rd[f"{env}_wt_attn_L{first_attn[0]}"] = per_layer_weights[first_attn][env]
                if ffn_keys:
                    last_ffn = ffn_keys[-1]
                    for env in ALL_ENVS:
                        rd[f"{env}_wt_ffn_L{last_ffn[0]}"] = per_layer_weights[last_ffn][env]

        # Update global state
        global_state = OrderedDict({k: global_state[k] + avg_delta[k] for k in global_state})
        apply_lora_state(model, global_state, device)

        # Final eval
        for env in ALL_ENVS:
            rd[f"{env}_eval_after"] = evaluate(model, eval_dl[env], device)

        hard_avg = sum(rd[f"{e}_eval_after"] for e in HARD_ENVS) / len(HARD_ENVS)
        all_avg = sum(rd[f"{e}_eval_after"] for e in ALL_ENVS) / len(ALL_ENVS)
        rd["hard_avg"] = hard_avg
        rd["avg_eval_loss"] = all_avg
        rd["time_sec"] = time.time() - t0

        # Convergence rate
        if rnd > 0:
            prev = metrics[-1]
            for env in ALL_ENVS:
                prev_loss = prev.get(f"{env}_eval_loss", None)
                curr_loss = rd[f"{env}_eval_after"]
                if prev_loss and prev_loss > 0.05:
                    rd[f"{env}_conv_rate"] = curr_loss / prev_loss
                else:
                    rd[f"{env}_conv_rate"] = None

        metrics.append(rd)

        # Compact output
        loss_str = " ".join(f"{e[:3]}={rd[f'{e}_eval_after']:.3f}" for e in ALL_ENVS)
        ret_str = " ".join(f"r_{e[:3]}={rd[f'{e}_retention']:.3f}" for e in ALL_ENVS)
        extra = ""
        if is_lafa and alphas:
            alpha_vals = list(alphas.values())
            extra = f"α_mean={np.mean(alpha_vals):.3f} α_range=[{np.min(alpha_vals):.3f},{np.max(alpha_vals):.3f}]"
        elif is_loss_wt:
            wt_str = " ".join(f"w_{e[:3]}={rd[f'{e}_weight']:.2f}" for e in ALL_ENVS)
            extra = wt_str
        print(f"  R{rnd:>2}: {loss_str}  hard={hard_avg:.3f}  |  {ret_str}  |  {extra}")

        # Periodic cleanup
        if rnd % 5 == 4:
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return metrics


def analyze_lafa(results):
    """Analyze LAFA-specific dynamics."""
    alpha_series = {}  # (layer_idx, module) -> [alpha per round]
    for rd in results:
        pa = rd.get("per_layer_alphas", {})
        for key, alpha in pa.items():
            alpha_series.setdefault(key, []).append(alpha)

    # Extract hard_avg progression
    hard_avg_progression = [rd["hard_avg"] for rd in results]

    # Webshop retention progression
    webshop_retention = [rd.get("webshop_retention", None) for rd in results]

    return {
        "alpha_series": alpha_series,
        "hard_avg_progression": hard_avg_progression,
        "webshop_retention": webshop_retention,
        "final_hard_avg": hard_avg_progression[-1],
        "final_alpha_mean": np.mean([v[-1] for v in alpha_series.values()]) if alpha_series else None,
        "final_alpha_range": (min(v[-1] for v in alpha_series.values()),
                              max(v[-1] for v in alpha_series.values())) if alpha_series else None,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--model_size", default="0.5B", choices=["0.5B", "1.5B"])
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--method", default="all", choices=["all", "uniform", "loss_wt", "lafa"])
    args = parser.parse_args()

    device = args.device
    model_name = MODEL_MAP[args.model_size]
    set_seed(SEED)

    print("=" * 70)
    print(f"  E14: Layer-Adaptive Federated Aggregation (LAFA)")
    print(f"  Model: {model_name}  Rounds: {args.rounds}  Device: {device}")
    print("=" * 70)

    # Load tokenizer and prepare data
    _, tokenizer = load_model_and_tokenizer(device, model_name)
    torch.cuda.empty_cache()
    print("  Tokenizing data...")
    train_dl, eval_dl = make_data(tokenizer)
    del tokenizer
    torch.cuda.empty_cache()

    methods = (args.method.split(",") if args.method != "all"
               else ["uniform", "loss_wt", "lafa"])

    all_results = {}
    for m in methods:
        all_results[m] = run_method(device, train_dl, eval_dl, m, args.rounds, model_name)
        save_dir = Path(f"outputs/e14_lafa/{args.model_size}/{m}")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(all_results[m], f, indent=2)

    # ================================================================
    # ANALYSIS
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  E14 RESULTS COMPARISON ({args.model_size}, {args.rounds}R)")
    print(f"{'='*70}")

    # Final hard_avg comparison
    for m in methods:
        final = all_results[m][-1]
        print(f"\n  {m}:")
        print(f"    hard_avg = {final['hard_avg']:.4f}")
        print(f"    avg_loss = {final['avg_eval_loss']:.4f}")
        for e in ALL_ENVS:
            print(f"    {e:10s} = {final[f'{e}_eval_after']:.4f}  "
                  f"(retention={final[f'{e}_retention']:.3f})")

    # LAFA vs baselines
    if "lafa" in all_results:
        lafa_hard = all_results["lafa"][-1]["hard_avg"]
        for m in methods:
            if m != "lafa":
                m_hard = all_results[m][-1]["hard_avg"]
                delta = lafa_hard - m_hard
                pct = delta / m_hard * 100 if m_hard != 0 else 0
                print(f"\n  LAFA vs {m}: {delta:+.4f} ({pct:+.1f}%) on hard_avg")

    # LAFA per-layer α analysis
    if "lafa" in all_results:
        lafa_analysis = analyze_lafa(all_results["lafa"])
        print(f"\n  LAFA per-layer α dynamics:")
        if lafa_analysis["final_alpha_mean"] is not None:
            lo, hi = lafa_analysis["final_alpha_range"]
            print(f"    Final α: mean={lafa_analysis['final_alpha_mean']:.3f}  "
                  f"range=[{lo:.3f}, {hi:.3f}]")

        # Show α evolution for representative layers
        alpha_series = lafa_analysis["alpha_series"]
        if alpha_series:
            # Group by module type, pick first and last layer of each
            attn_keys = sorted([k for k in alpha_series if "attn" in k])
            ffn_keys = sorted([k for k in alpha_series if "ffn" in k])
            selected = []
            if attn_keys:
                selected.extend([attn_keys[0], attn_keys[-1]])
            if ffn_keys:
                selected.extend([ffn_keys[0], ffn_keys[-1]])

            print(f"\n    α evolution (selected layers):")
            for key in selected:
                series = alpha_series[key]
                print(f"      {key}: {series[0]:.3f} → {series[-1]:.3f}")

    # Hard_avg progression
    print(f"\n  hard_avg progression:")
    for m in methods:
        progression = [rd["hard_avg"] for rd in all_results[m]]
        # Show every 5th round
        sampled = [(i, progression[i]) for i in range(0, len(progression), 5)]
        sampled_str = "  ".join(f"R{i}={v:.3f}" for i, v in sampled)
        print(f"    {m:>8}: {sampled_str}")

    # Save analysis
    analysis_dir = Path(f"outputs/e14_lafa/{args.model_size}")
    analysis = {
        m: {
            "final_hard_avg": all_results[m][-1]["hard_avg"],
            "final_avg_loss": all_results[m][-1]["avg_eval_loss"],
        }
        for m in methods
    }
    if "lafa" in all_results:
        lafa_a = analyze_lafa(all_results["lafa"])
        analysis["lafa_detail"] = {
            "final_alpha_mean": lafa_a["final_alpha_mean"],
            "final_alpha_range": list(lafa_a["final_alpha_range"]) if lafa_a["final_alpha_range"] else None,
        }
    with open(analysis_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\n  Results saved to outputs/e14_lafa/{args.model_size}/")


if __name__ == "__main__":
    main()
