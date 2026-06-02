"""
E12: SOTA Comparison — Is Loss-Proportional Competitive?

Methods (all 10 rounds):
  1. fedavg      — Standard FedAvg (reuse E11 alpha_0)
  2. loss_wt     — Loss-proportional w_k ∝ L_k (reuse E11 alpha_1)
  3. fedprox     — FedProx with proximal term μ||θ-θ_global||²
  4. grad_align  — Gradient-alignment weighting w_k ∝ cos(Δ_k, Δ_avg)

Usage:
    python scripts/e12_sota_compare.py --device cuda:7 --methods fedprox,grad_align
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
FEDPROX_MU = 0.1


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


def train_client_fedprox(model, dataloader, device, global_state, mu=FEDPROX_MU):
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    for batch in dataloader:
        ids, mask, labels = [b.to(device) for b in batch]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            ce_loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        prox_loss = 0.0
        for name, param in model.named_parameters():
            if "lora_" in name and param.requires_grad:
                prox_loss += ((param - global_state[name].to(device)) ** 2).sum()
        loss = ce_loss + (mu / 2) * prox_loss
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


def run_fedprox(device, train_dl, eval_dl, num_rounds):
    method_name = "fedprox"
    mu = FEDPROX_MU
    print(f"\n{'='*60}")
    print(f"  {method_name}  (mu={mu}, {num_rounds}R)")
    print(f"{'='*60}")

    model, _ = load_model_and_tokenizer(device)
    global_state = get_lora_state(model)
    metrics = []

    for rnd in range(num_rounds):
        t0 = time.time()
        rd = {"round": rnd, "method": method_name, "mu": mu}

        apply_lora_state(model, global_state, device)
        eval_losses = {}
        for env in ALL_ENVS:
            eval_losses[env] = evaluate(model, eval_dl[env], device)
            rd[f"{env}_eval_loss"] = eval_losses[env]

        weights = {e: 1.0 / len(ALL_ENVS) for e in ALL_ENVS}

        deltas = {}
        for env in ALL_ENVS:
            apply_lora_state(model, global_state, device)
            global_state_dev = {k: v.to(device) for k, v in global_state.items()}
            train_client_fedprox(model, train_dl[env], device, global_state_dev, mu)
            new_state = get_lora_state(model)
            deltas[env] = OrderedDict({
                k: new_state[k].float() - global_state[k].float() for k in global_state})

        weight_list = [weights[e] for e in ALL_ENVS]
        delta_list = [deltas[e] for e in ALL_ENVS]
        avg_delta = _weighted_average_state_dicts(delta_list, weight_list)

        for env in ALL_ENVS:
            ret = cosine_sim(deltas[env], avg_delta)
            rd[f"{env}_retention"] = ret
            rd[f"{env}_delta_norm"] = delta_norm(deltas[env])

        global_state = OrderedDict({k: global_state[k] + avg_delta[k] for k in global_state})
        apply_lora_state(model, global_state, device)

        for env in ALL_ENVS:
            rd[f"{env}_eval_after"] = evaluate(model, eval_dl[env], device)

        hard_avg = sum(rd[f"{e}_eval_after"] for e in HARD_ENVS) / len(HARD_ENVS)
        all_avg = sum(rd[f"{e}_eval_after"] for e in ALL_ENVS) / len(ALL_ENVS)
        rd["hard_avg"] = hard_avg
        rd["avg_eval_loss"] = all_avg
        rd["time_sec"] = time.time() - t0
        metrics.append(rd)

        loss_str = "  ".join(f"{e[:3]}={rd[f'{e}_eval_after']:.3f}" for e in ALL_ENVS)
        ret_str = "  ".join(f"ret_{e[:3]}={rd[f'{e}_retention']:+.3f}" for e in ALL_ENVS)
        print(f"  R{rnd:>2}: {loss_str}  hard={hard_avg:.4f}  |  {ret_str}")

    del model
    torch.cuda.empty_cache()
    return metrics


def run_grad_align(device, train_dl, eval_dl, num_rounds):
    method_name = "grad_align"
    print(f"\n{'='*60}")
    print(f"  {method_name}  (w_k ∝ cos(Δ_k, Δ_avg), {num_rounds}R)")
    print(f"{'='*60}")

    model, _ = load_model_and_tokenizer(device)
    global_state = get_lora_state(model)
    metrics = []

    for rnd in range(num_rounds):
        t0 = time.time()
        rd = {"round": rnd, "method": method_name}

        apply_lora_state(model, global_state, device)
        eval_losses = {}
        for env in ALL_ENVS:
            eval_losses[env] = evaluate(model, eval_dl[env], device)
            rd[f"{env}_eval_loss"] = eval_losses[env]

        deltas = {}
        for env in ALL_ENVS:
            apply_lora_state(model, global_state, device)
            train_client(model, train_dl[env], device)
            new_state = get_lora_state(model)
            deltas[env] = OrderedDict({
                k: new_state[k].float() - global_state[k].float() for k in global_state})

        # Step 1: compute uniform average as reference
        uniform_weights = [1.0 / len(ALL_ENVS)] * len(ALL_ENVS)
        uniform_avg = _weighted_average_state_dicts(
            [deltas[e] for e in ALL_ENVS], uniform_weights)

        # Step 2: weight proportional to alignment with uniform average
        alignments = {}
        for env in ALL_ENVS:
            alignments[env] = cosine_sim(deltas[env], uniform_avg)

        # Shift to ensure all positive: w_k = cos_k - min(cos) + epsilon
        min_align = min(alignments.values())
        shifted = {e: alignments[e] - min_align + 0.01 for e in ALL_ENVS}
        total_shifted = sum(shifted.values())
        weights = {e: shifted[e] / total_shifted for e in ALL_ENVS}

        # Re-aggregate with alignment weights
        weight_list = [weights[e] for e in ALL_ENVS]
        delta_list = [deltas[e] for e in ALL_ENVS]
        avg_delta = _weighted_average_state_dicts(delta_list, weight_list)

        for env in ALL_ENVS:
            ret = cosine_sim(deltas[env], avg_delta)
            rd[f"{env}_retention"] = ret
            rd[f"{env}_delta_norm"] = delta_norm(deltas[env])
            rd[f"{env}_alignment"] = alignments[env]

        global_state = OrderedDict({k: global_state[k] + avg_delta[k] for k in global_state})
        apply_lora_state(model, global_state, device)

        for env in ALL_ENVS:
            rd[f"{env}_eval_after"] = evaluate(model, eval_dl[env], device)

        hard_avg = sum(rd[f"{e}_eval_after"] for e in HARD_ENVS) / len(HARD_ENVS)
        all_avg = sum(rd[f"{e}_eval_after"] for e in ALL_ENVS) / len(ALL_ENVS)
        rd["hard_avg"] = hard_avg
        rd["avg_eval_loss"] = all_avg
        rd["time_sec"] = time.time() - t0
        metrics.append(rd)

        loss_str = "  ".join(f"{e[:3]}={rd[f'{e}_eval_after']:.3f}" for e in ALL_ENVS)
        wt_str = "  ".join(f"w_{e[:3]}={weights[e]:.2f}" for e in ALL_ENVS)
        print(f"  R{rnd:>2}: {loss_str}  hard={hard_avg:.4f}  |  {wt_str}")

    del model
    torch.cuda.empty_cache()
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--methods", default="all",
                        help="Comma-separated: fedprox,grad_align,all")
    args = parser.parse_args()
    device = args.device
    set_seed(SEED)

    if args.methods == "all":
        methods = ["fedprox", "grad_align"]
    else:
        methods = args.methods.split(",")

    print("=" * 60)
    print("  E12: SOTA Comparison for LLM Agent FL")
    print("=" * 60)
    print()
    print("  FedAvg + Loss-proportional results reused from E11.")
    print("  Running: {}".format(", ".join(methods)))
    print()

    _, tokenizer = load_model_and_tokenizer(device)
    torch.cuda.empty_cache()
    print("  Tokenizing data...")
    train_dl, eval_dl = make_data(tokenizer)
    del tokenizer

    all_results = {}

    # Load E11 results for baselines
    e11_dir = Path("outputs/e11_alpha_sweep")
    if (e11_dir / "alpha_0" / "metrics.json").exists():
        all_results["fedavg"] = json.load(open(e11_dir / "alpha_0" / "metrics.json"))
        print("  Loaded E11 fedavg (alpha=0) results.")
    if (e11_dir / "alpha_1" / "metrics.json").exists():
        all_results["loss_wt"] = json.load(open(e11_dir / "alpha_1" / "metrics.json"))
        print("  Loaded E11 loss_wt (alpha=1) results.")

    # Run new methods
    runners = {
        "fedprox": run_fedprox,
        "grad_align": run_grad_align,
    }
    for method in methods:
        if method in runners:
            all_results[method] = runners[method](device, train_dl, eval_dl, args.rounds)
            save_dir = Path("outputs/e12_sota_compare/{}".format(method))
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / "metrics.json", "w") as f:
                json.dump(all_results[method], f, indent=2)

    # ============================================================
    # Final comparison
    # ============================================================
    all_methods = ["fedavg", "loss_wt"] + [m for m in methods if m not in ["fedavg", "loss_wt"]]
    available = [m for m in all_methods if m in all_results]

    print(f"\n{'='*100}")
    print("  E12 FINAL COMPARISON")
    print(f"{'='*100}")

    print(f"\n  R9 (final) hard_avg:")
    for m in available:
        r = all_results[m][-1]
        hard_avg = r.get("hard_avg", sum(r[f"{e}_eval_after"] for e in HARD_ENVS) / 3)
        print(f"    {m:>12}: hard_avg={hard_avg:.4f}")

    print(f"\n  Per-env (R9):")
    header = "  {:>12} | ".format("method") + " | ".join("{:>8}".format(e[:6]) for e in ALL_ENVS)
    print(header)
    print("  " + "-" * 70)
    for m in available:
        r = all_results[m][-1]
        vals = [r[f"{e}_eval_after"] for e in ALL_ENVS]
        print("  {:>12} | ".format(m) + " | ".join("{:>8.4f}".format(v) for v in vals))

    print(f"\n  Webshop retention (R0→R9):")
    for m in available:
        data = all_results[m]
        if "webshop_retention" in data[0]:
            r0 = data[0]["webshop_retention"]
            r9 = data[-1]["webshop_retention"]
            print(f"    {m:>12}: {r0:+.3f} -> {r9:+.3f}")

    print(f"\n  Hard env retention at R9:")
    for m in available:
        r = all_results[m][-1]
        rets = [r.get(f"{e}_retention", 0) for e in HARD_ENVS]
        print(f"    {m:>12}: " + " ".join("{:+.3f}".format(x) for x in rets))

    # The key verdict
    print(f"\n{'='*100}")
    print("  VERDICT: Is loss-proportional competitive?")
    print(f"{'='*100}")
    if "loss_wt" in all_results and "fedprox" in all_results:
        lw_hard = all_results["loss_wt"][-1].get("hard_avg",
                  sum(all_results["loss_wt"][-1][f"{e}_eval_after"] for e in HARD_ENVS) / 3)
        fp_hard = all_results["fedprox"][-1].get("hard_avg",
                  sum(all_results["fedprox"][-1][f"{e}_eval_after"] for e in HARD_ENVS) / 3)
        fa_hard = all_results["fedavg"][-1].get("hard_avg",
                  sum(all_results["fedavg"][-1][f"{e}_eval_after"] for e in HARD_ENVS) / 3)

        print()
        print("  fedavg:      hard_avg = {:.4f}".format(fa_hard))
        print("  loss_wt:     hard_avg = {:.4f}  ({:+.1f}% vs fedavg)".format(
            lw_hard, (fa_hard - lw_hard) / fa_hard * 100))
        print("  fedprox:     hard_avg = {:.4f}  ({:+.1f}% vs fedavg)".format(
            fp_hard, (fa_hard - fp_hard) / fa_hard * 100))

        if lw_hard < fp_hard:
            print("\n  >>> loss_wt BEATS fedprox on hard tasks: direction WORTH pursuing")
        else:
            gap = (lw_hard - fp_hard) / fp_hard * 100
            print("\n  >>> fedprox beats loss_wt by {:.1f}%: direction needs rethinking".format(gap))


if __name__ == "__main__":
    main()
