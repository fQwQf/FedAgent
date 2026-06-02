"""
E13: Rate Equalization Theory Validation

Tests whether loss-proportional aggregation achieves rate equalization
across tasks, and whether this scales to larger models.

Key predictions:
  1. Loss-weighted: cross-task convergence rate std → 0 (equalization)
  2. Uniform: cross-task rate std stays high (divergent rates)
  3. Self-reinforcing dynamics: hardest task's weight & retention increase
  4. Phase transition: occurs when easiest task loss drops below threshold
  5. Scale invariance: dynamics similar at 0.5B and 1.5B

Usage:
    python scripts/e13_rate_equalization.py --model_size 1.5B --device cuda:0
    python scripts/e13_rate_equalization.py --model_size 0.5B --device cuda:0
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
    dot = sum(torch.dot(v1[k].flatten().float(), v2[k].flatten().float()) for k in v1)
    n1 = sum(v1[k].float().norm()**2 for k in v1)**0.5
    n2 = sum(v2[k].float().norm()**2 for k in v2)**0.5
    return (dot / (n1 * n2 + 1e-10)).item()


def delta_norm(delta):
    return float(sum(v.float().norm()**2 for v in delta.values())**0.5)


def run_method(device, train_dl, eval_dl, method, num_rounds, model_name):
    use_loss_wt = (method == "loss_wt")

    print(f"\n{'='*70}")
    print(f"  {method}  ({'loss-proportional' if use_loss_wt else 'uniform'}, "
          f"{num_rounds}R, {model_name})")
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

        # Compute weights
        if use_loss_wt:
            total = sum(eval_losses[e] for e in ALL_ENVS)
            weights = {e: eval_losses[e] / total for e in ALL_ENVS}
        else:
            weights = {e: 1.0 / len(ALL_ENVS) for e in ALL_ENVS}

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

        # Signal retention
        for env in ALL_ENVS:
            ret = cosine_sim(deltas[env], avg_delta)
            rd[f"{env}_retention"] = ret
            rd[f"{env}_delta_norm"] = delta_norm(deltas[env])
        rd["avg_retention"] = np.mean([rd[f"{e}_retention"] for e in ALL_ENVS])

        # Store weights
        for env in ALL_ENVS:
            rd[f"{env}_weight"] = weights[env]

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

        # Convergence rate (if not first round)
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
        wt_str = " ".join(f"w_{e[:3]}={weights[e]:.2f}" for e in ALL_ENVS) if use_loss_wt else ""
        rate_str = ""
        if rnd > 0:
            rates = [rd.get(f"{e}_conv_rate") for e in ALL_ENVS]
            valid_rates = [r for r in rates if r is not None]
            if valid_rates:
                rate_str = f"rate_std={np.std(valid_rates):.3f}"
        print(f"  R{rnd:>2}: {loss_str}  hard={hard_avg:.3f}  |  {ret_str}  |  {wt_str}  |  {rate_str}")

        # Periodic cleanup
        if rnd % 5 == 4:
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return metrics


def analyze_rate_equalization(results, method_name):
    """Analyze rate equalization dynamics."""
    tasks = ALL_ENVS
    n_rounds = len(results)

    # Extract convergence rates
    task_rates = {t: [] for t in tasks}
    for r in range(1, n_rounds):
        rd = results[r]
        for t in tasks:
            rate = rd.get(f"{t}_conv_rate")
            if rate is not None:
                task_rates[t].append(rate)

    # Cross-task rate std at each round
    cross_task_stds = []
    for i in range(min(len(task_rates[t]) for t in tasks)):
        round_rates = [task_rates[t][i] for t in tasks]
        cross_task_stds.append(np.std(round_rates))

    # Weight dynamics (for loss_wt)
    weight_series = {t: [] for t in tasks}
    for rd in results:
        for t in tasks:
            w = rd.get(f"{t}_weight")
            if w is not None:
                weight_series[t].append(w)

    # Retention dynamics
    retention_series = {t: [] for t in tasks}
    for rd in results:
        for t in tasks:
            retention_series[t].append(rd[f"{t}_retention"])

    return {
        "task_mean_rates": {t: np.mean(task_rates[t]) for t in tasks if task_rates[t]},
        "cross_task_rate_std_mean": np.mean(cross_task_stds) if cross_task_stds else None,
        "cross_task_rate_std_per_round": cross_task_stds,
        "weight_series": weight_series,
        "retention_series": retention_series,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_size", default="1.5B", choices=["0.5B", "1.5B"])
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--method", default="all", choices=["all", "uniform", "loss_wt"])
    args = parser.parse_args()

    device = args.device
    model_name = MODEL_MAP[args.model_size]
    set_seed(SEED)

    print("=" * 70)
    print(f"  E13: Rate Equalization Theory Validation")
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
               else ["uniform", "loss_wt"])

    all_results = {}
    all_analysis = {}
    for m in methods:
        all_results[m] = run_method(device, train_dl, eval_dl, m, args.rounds, model_name)
        save_dir = Path(f"outputs/e13_rate_equalization/{args.model_size}/{m}")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(all_results[m], f, indent=2)
        all_analysis[m] = analyze_rate_equalization(all_results[m], m)

    # ================================================================
    # ANALYSIS: Rate Equalization Comparison
    # ================================================================
    print(f"\n{'='*70}")
    print("  RATE EQUALIZATION ANALYSIS")
    print(f"{'='*70}")

    for m in methods:
        a = all_analysis[m]
        print(f"\n  {m} - Mean convergence rates per task:")
        for t in ALL_ENVS:
            rate = a["task_mean_rates"].get(t, None)
            if rate:
                print(f"    {t:10s}: {rate:.4f}")
        if a["cross_task_rate_std_mean"] is not None:
            print(f"  Cross-task rate std (mean): {a['cross_task_rate_std_mean']:.4f}")

    # Compare rate stds
    if len(methods) == 2:
        stds = {m: all_analysis[m]["cross_task_rate_std_mean"] for m in methods}
        if all(v is not None for v in stds.values()):
            ratio = max(stds.values()) / min(stds.values())
            better = min(stds, key=stds.get)
            print(f"\n  Rate equalization ratio: {ratio:.1f}x  ({better} more equalized)")

    # ================================================================
    # ANALYSIS: Self-Reinforcing Dynamics
    # ================================================================
    print(f"\n{'='*70}")
    print("  SELF-REINFORCING DYNAMICS (webshop = hardest)")
    print(f"{'='*70}")

    if "loss_wt" in all_analysis:
        ws_weights = all_analysis["loss_wt"]["weight_series"].get("webshop", [])
        ws_ret = all_analysis["loss_wt"]["retention_series"].get("webshop", [])
        uf_ws_ret = all_analysis.get("uniform", {}).get("retention_series", {}).get("webshop", [])

        if ws_weights:
            print(f"\n  Webshop weight:  {ws_weights[0]:.3f} → {ws_weights[-1]:.3f}  "
                  f"(change: {ws_weights[-1]-ws_weights[0]:+.3f})")
            print(f"  Webshop retention: {ws_ret[0]:.3f} → {ws_ret[-1]:.3f}  "
                  f"(change: {ws_ret[-1]-ws_ret[0]:+.3f})")

        if uf_ws_ret:
            gap_start = ws_ret[0] - uf_ws_ret[0] if ws_ret else 0
            gap_end = ws_ret[-1] - uf_ws_ret[-1] if ws_ret else 0
            print(f"  LW-UF retention gap: {gap_start:+.3f} → {gap_end:+.3f}  "
                  f"(widened by {gap_end-gap_start:+.3f})")

    # ================================================================
    # ANALYSIS: Phase Transition Detection
    # ================================================================
    print(f"\n{'='*70}")
    print("  PHASE TRANSITION DETECTION")
    print(f"{'='*70}")

    if "loss_wt" in all_results:
        res = all_results["loss_wt"]
        # Find round where easiest task loss < threshold
        for t in ALL_ENVS:
            for r, rd in enumerate(res):
                if rd[f"{t}_eval_loss"] < 0.5:
                    print(f"  {t}: first drops below 0.5 at R{r}")
                    break

        # Find round where webshop weight starts increasing consistently
        ws_wts = [rd.get("webshop_weight", 0) for rd in res]
        for r in range(1, len(ws_wts) - 2):
            if ws_wts[r+1] > ws_wts[r] and ws_wts[r+2] > ws_wts[r+1]:
                print(f"  Weight phase transition at R{r} (weight: {ws_wts[r]:.3f} → {ws_wts[r+2]:.3f})")
                break

    # ================================================================
    # Final Summary
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS ({args.model_size}, {args.rounds}R)")
    print(f"{'='*70}")

    for m in methods:
        final = all_results[m][-1]
        print(f"\n  {m}:")
        print(f"    hard_avg = {final['hard_avg']:.4f}")
        for e in ALL_ENVS:
            print(f"    {e:10s} = {final[f'{e}_eval_after']:.4f}  "
                  f"(retention={final[f'{e}_retention']:.3f})")

    # Save analysis
    analysis_dir = Path(f"outputs/e13_rate_equalization/{args.model_size}")
    analysis_summary = {}
    for m in methods:
        a = all_analysis[m]
        analysis_summary[m] = {
            "task_mean_rates": {k: float(v) for k, v in a["task_mean_rates"].items()},
            "cross_task_rate_std_mean": float(a["cross_task_rate_std_mean"]) if a["cross_task_rate_std_mean"] else None,
        }
    with open(analysis_dir / "analysis.json", "w") as f:
        json.dump(analysis_summary, f, indent=2)

    print(f"\n  Results saved to outputs/e13_rate_equalization/{args.model_size}/")


if __name__ == "__main__":
    main()
