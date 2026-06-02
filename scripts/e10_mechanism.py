"""
E10: Mechanism-to-Outcome Linkage

10-round training with per-round signal retention measurement.
Links the gradient reversal mechanism (E9) to convergence outcomes (E7/E8).

Methods:
  1. uniform  -- Standard FedAvg (equal weights), 10 rounds
  2. loss_wt  -- Loss-proportional weighting, 10 rounds
  3. 3hard    -- Only hard envs, FedAvg, 10 rounds

At each round, measures:
  - Per-env eval loss (convergence)
  - Per-env signal retention: cos(Δ_k, Δ_avg) (mechanism)
  - Per-env gradient norm (for norm ratio analysis)

Key predictions:
  - Uniform: webshop retention stays negative → slow convergence
  - Loss_wt: webshop retention becomes positive → faster convergence
  - 3hard: all envs have high retention → fastest hard-env convergence
  - Loss_wt closes >50% of the 3hard gap over 10 rounds

Usage:
    python scripts/e10_mechanism.py --device cuda:4
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
EASY_ENVS = ["babyai", "maze"]
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


def run_method(device, train_dl, eval_dl, method, num_rounds):
    env_list = HARD_ENVS if method == "3hard" else ALL_ENVS
    use_loss_wt = (method == "loss_wt")

    print(f"\n{'='*60}")
    print(f"  {method}  ({'loss-proportional' if use_loss_wt else 'uniform'}, {num_rounds}R)")
    print(f"{'='*60}")

    model, _ = load_model_and_tokenizer(device)
    global_state = get_lora_state(model)
    metrics = []

    for rnd in range(num_rounds):
        t0 = time.time()
        rd = {"round": rnd, "method": method}

        # Eval first (for loss-weighting and logging)
        apply_lora_state(model, global_state, device)
        eval_losses = {}
        for env in ALL_ENVS:
            eval_losses[env] = evaluate(model, eval_dl[env], device)
            rd[f"{env}_eval_loss"] = eval_losses[env]

        # Compute weights
        if use_loss_wt:
            total = sum(eval_losses[e] for e in env_list)
            weights = {e: eval_losses[e] / total for e in env_list}
        else:
            weights = {e: 1.0 / len(env_list) for e in env_list}

        # Client updates
        deltas = {}
        for env in env_list:
            apply_lora_state(model, global_state, device)
            train_client(model, train_dl[env], device)
            new_state = get_lora_state(model)
            deltas[env] = OrderedDict({
                k: new_state[k].float() - global_state[k].float() for k in global_state})

        # Aggregate
        weight_list = [weights[e] for e in env_list]
        delta_list = [deltas[e] for e in env_list]
        avg_delta = _weighted_average_state_dicts(delta_list, weight_list)

        # Signal retention: cos(Δ_k, Δ_avg) for each env
        for env in env_list:
            ret = cosine_sim(deltas[env], avg_delta)
            rd[f"{env}_retention"] = ret
            rd[f"{env}_delta_norm"] = delta_norm(deltas[env])
        rd[f"avg_retention"] = np.mean([rd[f"{e}_retention"] for e in env_list])

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

        # Compact per-round output
        ret_str = "  ".join(f"ret_{e[:3]}={rd[f'{e}_retention']:+.3f}" for e in env_list)
        loss_str = "  ".join(f"{e[:3]}={rd[f'{e}_eval_after']:.3f}" for e in ALL_ENVS)
        wt_str = "  ".join(f"w_{e[:3]}={weights[e]:.2f}" for e in env_list) if use_loss_wt else ""
        print(f"  R{rnd:>2}: {loss_str}  hard={hard_avg:.4f}  |  {ret_str}  {wt_str}")

    del model
    torch.cuda.empty_cache()
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--method", default="all",
                        choices=["all", "uniform", "loss_wt", "3hard"])
    args = parser.parse_args()
    device = args.device
    set_seed(SEED)

    print("=" * 60)
    print("  E10: Mechanism-to-Outcome Linkage (10R)")
    print("=" * 60)

    _, tokenizer = load_model_and_tokenizer(device)
    torch.cuda.empty_cache()
    print("  Tokenizing data...")
    train_dl, eval_dl = make_data(tokenizer)
    del tokenizer

    methods = (args.method.split(",") if args.method != "all"
               else ["uniform", "loss_wt", "3hard"])

    all_results = {}
    for m in methods:
        all_results[m] = run_method(device, train_dl, eval_dl, m, args.rounds)
        save_dir = Path(f"outputs/e10_mechanism/{m}")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(all_results[m], f, indent=2)

    # ============================================================
    # Final comparison
    # ============================================================
    print(f"\n{'='*100}")
    print("  E10 CONVERGENCE (eval_after at each round)")
    print(f"{'='*100}")

    for env in ALL_ENVS + ["hard_avg", "avg_eval_loss"]:
        label = env if env in ALL_ENVS else env.replace("_", " ").upper()
        print(f"\n  {label}:")
        for m in methods:
            vals = [r.get(f"{env}_eval_after" if env in ALL_ENVS else env, r.get(f"{env}", None))
                    for r in all_results[m]]
            print(f"    {m:>10}: " + " → ".join(f"{v:.3f}" if v else "?" for v in vals))

    # Signal retention evolution
    print(f"\n{'='*100}")
    print("  SIGNAL RETENTION EVOLUTION")
    print(f"{'='*100}")

    for env in ALL_ENVS:
        print(f"\n  {env} retention [cos(Δ_k, Δ_avg)]:")
        for m in methods:
            env_list = HARD_ENVS if m == "3hard" else ALL_ENVS
            if env not in env_list:
                continue
            rets = [r[f"{env}_retention"] for r in all_results[m]]
            neg_count = sum(1 for r in rets if r < 0)
            print(f"    {m:>10}: " + " → ".join(f"{r:+.3f}" for r in rets) +
                  f"  (neg={neg_count}/{len(rets)})")

    # Summary: hard env performance gap
    print(f"\n{'='*100}")
    print("  HARD ENV GAP ANALYSIS")
    print(f"{'='*100}")

    if "3hard" in all_results and "uniform" in all_results:
        for e in HARD_ENVS:
            u_final = all_results["uniform"][-1][f"{e}_eval_after"]
            h_final = all_results["3hard"][-1][f"{e}_eval_after"]
            if "loss_wt" in all_results:
                w_final = all_results["loss_wt"][-1][f"{e}_eval_after"]
                gap_closed = (u_final - w_final) / (u_final - h_final) * 100 if u_final != h_final else 0
                print(f"  {e:>10}: uniform={u_final:.4f}  loss_wt={w_final:.4f}  "
                      f"3hard={h_final:.4f}  gap_closed={gap_closed:.0f}%")

    # Key: correlation between retention and improvement
    print(f"\n{'='*100}")
    print("  RETENTION → IMPROVEMENT CORRELATION")
    print(f"{'='*100}")

    for m in methods:
        rets = []
        imps = []
        for i in range(1, len(all_results[m])):
            for env in (HARD_ENVS if m == "3hard" else ALL_ENVS):
                ret = all_results[m][i-1].get(f"{env}_retention", None)
                if ret is not None:
                    l_before = all_results[m][i-1].get(f"{env}_eval_after",
                                all_results[m][i-1].get(f"{env}_eval_loss", None))
                    l_after = all_results[m][i].get(f"{env}_eval_after",
                              all_results[m][i].get(f"{env}_eval_loss", None))
                    if l_before and l_after and l_before > 0:
                        rets.append(ret)
                        imps.append((l_before - l_after) / l_before)

        if len(rets) > 5:
            corr = np.corrcoef(rets, imps)[0, 1]
            print(f"  {m:>10}: r(retention, improvement) = {corr:.3f}  (n={len(rets)})")


if __name__ == "__main__":
    main()
