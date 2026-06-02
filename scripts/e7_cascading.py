"""
E7: Cascading Knowledge Transfer Validation

Tests the hypothesis: easy-client mastery enables hard-client phase transitions.

Methods:
  1. fed_5env_nat     -- 5 envs, all data, FedAvg (reproduces E6 NAT baseline)
  2. fed_3hard_nat    -- ONLY hard envs (webshop/textcraft/wordle), all data, FedAvg
  3. local_each_nat   -- No federation, each env trains alone with all data
  4. fed_5env_nat_10r -- 10 rounds of 5-env FedAvg (to see phase transition fully)
  5. fed_3hard_nat_10r-- 10 rounds of 3-hard-env FedAvg

Key predictions:
  - fed_5env_nat >> fed_3hard_nat for hard envs (easy clients help!)
  - fed_3hard_nat ≈ local_each_nat for hard envs (no benefit without easy clients)
  - Phase transition at R2-R3 only appears in fed_5env_nat, NOT in fed_3hard_nat
  - 10-round runs show the pattern is stable and continues

Usage:
    python scripts/e7_cascading.py --device cuda:4
    python scripts/e7_cascading.py --device cuda:4 --method fed_3hard_nat
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import time
import random
import argparse
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn.functional as F
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
SEED = 42
TRAIN_SAMPLES = 64
EVAL_SAMPLES = 32


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM", bias="none",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def tokenize_trajectory(tokenizer, traj):
    messages, _ = get_trainable_messages(traj)
    if not messages:
        return None
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    if not text.strip():
        return None
    tok = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH,
                    padding="max_length", return_tensors="pt")
    input_ids = tok["input_ids"]
    attention_mask = tok["attention_mask"]
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return input_ids[0], attention_mask[0], labels[0]


def prepare_data(tokenizer, env_list=None):
    if env_list is None:
        env_list = ALL_ENVS
    train_tensors = {}
    eval_loaders = {}
    for env in env_list:
        data = load_agentgym_data(env, max_samples=TRAIN_SAMPLES + EVAL_SAMPLES, seed=SEED)
        train_data = data[:TRAIN_SAMPLES]
        eval_data = data[TRAIN_SAMPLES:TRAIN_SAMPLES + EVAL_SAMPLES]

        train_pts = []
        for traj in train_data:
            pt = tokenize_trajectory(tokenizer, traj)
            if pt is not None:
                train_pts.append(pt)
        train_tensors[env] = train_pts

        eval_pts = []
        for traj in eval_data:
            pt = tokenize_trajectory(tokenizer, traj)
            if pt is not None:
                eval_pts.append(pt)
        if eval_pts:
            ids = torch.stack([p[0] for p in eval_pts])
            masks = torch.stack([p[1] for p in eval_pts])
            labels = torch.stack([p[2] for p in eval_pts])
            eval_loaders[env] = DataLoader(
                TensorDataset(ids, masks, labels), batch_size=BATCH_SIZE, shuffle=False
            )
        print(f"  {env}: {len(train_pts)} train, {len(eval_pts)} eval")

    return train_tensors, eval_loaders


def get_lora_state(model):
    return OrderedDict({
        k: v.clone().cpu() for k, v in model.state_dict().items() if "lora_" in k
    })


def apply_lora_state(model, lora_state, device):
    model.load_state_dict(
        OrderedDict({k: v.to(device) for k, v in lora_state.items()}), strict=False
    )


def compute_lora_delta(old_state, new_state):
    return OrderedDict({
        k: new_state[k].float() - old_state[k].float() for k in old_state
    })


def make_dataloader(tensors):
    if not tensors:
        return None
    ids = torch.stack([t[0] for t in tensors])
    masks = torch.stack([t[1] for t in tensors])
    labels = torch.stack([t[2] for t in tensors])
    return DataLoader(TensorDataset(ids, masks, labels),
                      batch_size=BATCH_SIZE, shuffle=True)


def train_one_epoch(model, dataloader, device):
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    for batch in dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels)
        optimizer.zero_grad()
        outputs.loss.backward()
        optimizer.step()


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                labels=labels)
            total_loss += outputs.loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
    return total_loss / max(total_samples, 1)


# ============================================================
# Method 1: FedAvg with ALL 5 envs (reproduces E6 NAT)
# ============================================================
def run_fed_5env(device, train_tensors, eval_loaders, num_rounds):
    label = f"fed_5env_nat{'' if num_rounds == 5 else f'_{num_rounds}r'}"
    print(f"\n{'='*60}")
    print(f"  {label}: 5-env FedAvg, all data, {num_rounds} rounds")
    print(f"{'='*60}")

    model, _ = load_model_and_tokenizer(device)
    global_state = get_lora_state(model)
    all_metrics = []

    for rnd in range(num_rounds):
        t0 = time.time()
        round_data = {"round": rnd, "method": label}
        apply_lora_state(model, global_state, device)

        deltas = []
        for env in ALL_ENVS:
            apply_lora_state(model, global_state, device)
            dl = make_dataloader(train_tensors[env])
            if dl is not None:
                train_one_epoch(model, dl, device)
            updated_state = get_lora_state(model)
            deltas.append(compute_lora_delta(global_state, updated_state))

        global_state = apply_delta_to_state(global_state,
            _weighted_average_state_dicts(deltas, [1.0]*len(deltas)))
        apply_lora_state(model, global_state, device)

        for env in ALL_ENVS:
            round_data[f"{env}_eval_loss"] = evaluate(model, eval_loaders[env], device)

        # Compute per-group averages
        easy_avg = sum(round_data[f"{e}_eval_loss"] for e in EASY_ENVS) / len(EASY_ENVS)
        hard_avg = sum(round_data[f"{e}_eval_loss"] for e in HARD_ENVS) / len(HARD_ENVS)
        all_avg = sum(round_data[f"{e}_eval_loss"] for e in ALL_ENVS) / len(ALL_ENVS)
        round_data["easy_avg"] = easy_avg
        round_data["hard_avg"] = hard_avg
        round_data["avg_eval_loss"] = all_avg
        round_data["time_sec"] = time.time() - t0
        all_metrics.append(round_data)

        env_str = "  ".join(f"{e}={round_data[f'{e}_eval_loss']:.3f}" for e in ALL_ENVS)
        print(f"  R{rnd}: {env_str}  avg={all_avg:.4f}  easy={easy_avg:.4f}  hard={hard_avg:.4f}")

    del model
    torch.cuda.empty_cache()
    return all_metrics


# ============================================================
# Method 2: FedAvg with ONLY hard envs (no easy env transfer)
# ============================================================
def run_fed_3hard(device, train_tensors, eval_loaders, num_rounds):
    label = f"fed_3hard_nat{'' if num_rounds == 5 else f'_{num_rounds}r'}"
    print(f"\n{'='*60}")
    print(f"  {label}: 3-hard-env FedAvg, all data, {num_rounds} rounds")
    print(f"  NO easy env knowledge transfer!")
    print(f"{'='*60}")

    model, _ = load_model_and_tokenizer(device)
    global_state = get_lora_state(model)
    all_metrics = []

    for rnd in range(num_rounds):
        t0 = time.time()
        round_data = {"round": rnd, "method": label}
        apply_lora_state(model, global_state, device)

        deltas = []
        for env in HARD_ENVS:
            apply_lora_state(model, global_state, device)
            dl = make_dataloader(train_tensors[env])
            if dl is not None:
                train_one_epoch(model, dl, device)
            updated_state = get_lora_state(model)
            deltas.append(compute_lora_delta(global_state, updated_state))

        global_state = apply_delta_to_state(global_state,
            _weighted_average_state_dicts(deltas, [1.0]*len(deltas)))
        apply_lora_state(model, global_state, device)

        # Evaluate on ALL envs (including easy, as a probe)
        for env in ALL_ENVS:
            round_data[f"{env}_eval_loss"] = evaluate(model, eval_loaders[env], device)

        easy_avg = sum(round_data[f"{e}_eval_loss"] for e in EASY_ENVS) / len(EASY_ENVS)
        hard_avg = sum(round_data[f"{e}_eval_loss"] for e in HARD_ENVS) / len(HARD_ENVS)
        all_avg = sum(round_data[f"{e}_eval_loss"] for e in ALL_ENVS) / len(ALL_ENVS)
        round_data["easy_avg"] = easy_avg
        round_data["hard_avg"] = hard_avg
        round_data["avg_eval_loss"] = all_avg
        round_data["time_sec"] = time.time() - t0
        all_metrics.append(round_data)

        env_str = "  ".join(f"{e}={round_data[f'{e}_eval_loss']:.3f}" for e in ALL_ENVS)
        print(f"  R{rnd}: {env_str}  avg={all_avg:.4f}  easy={easy_avg:.4f}  hard={hard_avg:.4f}")

    del model
    torch.cuda.empty_cache()
    return all_metrics


# ============================================================
# Method 3: Local training (no federation)
# ============================================================
def run_local(device, train_tensors, eval_loaders, num_rounds):
    label = "local_each_nat"
    print(f"\n{'='*60}")
    print(f"  {label}: No federation, each env trains independently")
    print(f"{'='*60}")

    all_metrics = []

    for env in ALL_ENVS:
        print(f"\n  --- Training {env} locally ---")
        model, _ = load_model_and_tokenizer(device)
        global_state = get_lora_state(model)

        for rnd in range(num_rounds):
            apply_lora_state(model, global_state, device)
            dl = make_dataloader(train_tensors[env])
            if dl is not None:
                train_one_epoch(model, dl, device)
            global_state = get_lora_state(model)

        # Evaluate
        round_data = {"round": num_rounds - 1, "method": label}
        for e in ALL_ENVS:
            round_data[f"{e}_eval_loss"] = evaluate(model, eval_loaders[e], device)

        easy_avg = sum(round_data[f"{e}_eval_loss"] for e in EASY_ENVS) / len(EASY_ENVS)
        hard_avg = sum(round_data[f"{e}_eval_loss"] for e in HARD_ENVS) / len(HARD_ENVS)
        all_avg = sum(round_data[f"{e}_eval_loss"] for e in ALL_ENVS) / len(ALL_ENVS)
        round_data["easy_avg"] = easy_avg
        round_data["hard_avg"] = hard_avg
        round_data["avg_eval_loss"] = all_avg
        all_metrics.append(round_data)

        env_str = "  ".join(f"{e}={round_data[f'{e}_eval_loss']:.3f}" for e in ALL_ENVS)
        print(f"  {env} final: {env_str}  avg={all_avg:.4f}")

        del model
        torch.cuda.empty_cache()

    # Aggregate: for local training, we report per-env final results
    # The "method-level" result uses each env's locally-trained model for its own eval
    return all_metrics


def apply_delta_to_state(state, delta):
    return OrderedDict({k: state[k] + delta[k] for k in state})


def summarize_round(round_data, env_list=None):
    if env_list is None:
        env_list = ALL_ENVS
    losses = [round_data[f"{e}_eval_loss"] for e in env_list]
    avg = sum(losses) / len(losses)
    std = (sum((l - avg)**2 for l in losses) / len(losses)) ** 0.5
    return avg, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--method", choices=["all", "fed_5env_nat", "fed_3hard_nat",
                                            "local_each_nat"], default="all")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)
    num_rounds = args.rounds

    print("=" * 60)
    print(f"  E7: Cascading Knowledge Transfer Validation")
    print(f"  Device: {device}  |  Rounds: {num_rounds}")
    print(f"  Hypothesis: easy-client mastery enables hard-client phase transitions")
    print("=" * 60)

    print("\n  Loading tokenizer & model...")
    _, tokenizer = load_model_and_tokenizer(device)
    del _
    torch.cuda.empty_cache()

    print("  Pre-tokenizing data (all 5 envs)...")
    train_tensors, eval_loaders = prepare_data(tokenizer)
    del tokenizer

    all_results = {}
    methods = args.method.split(",") if args.method != "all" else [
        "fed_5env_nat", "fed_3hard_nat", "local_each_nat"
    ]

    for method in methods:
        if method == "fed_5env_nat":
            all_results[method] = run_fed_5env(device, train_tensors, eval_loaders, num_rounds)
        elif method == "fed_3hard_nat":
            all_results[method] = run_fed_3hard(device, train_tensors, eval_loaders, num_rounds)
        elif method == "local_each_nat":
            all_results[method] = run_local(device, train_tensors, eval_loaders, num_rounds)

        save_dir = Path(f"outputs/e7_cascading/{method}")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(all_results[method], f, indent=2)

    # ============================================================
    # Final Comparison
    # ============================================================
    print(f"\n{'='*100}")
    print("  E7 FINAL COMPARISON")
    print(f"{'='*100}")

    # Header
    print(f"{'Method':>22} | {'Avg':>8} | {'Easy':>8} | {'Hard':>8} |", end="")
    for e in ALL_ENVS:
        print(f" {e[:5]:>7}", end="")
    print()
    print("-" * 100)

    for method in methods:
        if method == "local_each_nat":
            # For local: each env evaluated on its own model
            env_results = all_results[method]
            per_env = {}
            for i, env in enumerate(ALL_ENVS):
                per_env[f"{env}_eval_loss"] = env_results[i][f"{env}_eval_loss"]
            avg = sum(per_env[f"{e}_eval_loss"] for e in ALL_ENVS) / len(ALL_ENVS)
            easy = sum(per_env[f"{e}_eval_loss"] for e in EASY_ENVS) / len(EASY_ENVS)
            hard = sum(per_env[f"{e}_eval_loss"] for e in HARD_ENVS) / len(HARD_ENVS)
            print(f"{method:>22} | {avg:>8.4f} | {easy:>8.4f} | {hard:>8.4f} |", end="")
            for e in ALL_ENVS:
                print(f" {per_env[f'{e}_eval_loss']:>7.4f}", end="")
            print("  (local)")
        else:
            d = all_results[method][-1]
            print(f"{method:>22} | {d['avg_eval_loss']:>8.4f} | {d['easy_avg']:>8.4f} | {d['hard_avg']:>8.4f} |", end="")
            for e in ALL_ENVS:
                print(f" {d[f'{e}_eval_loss']:>7.4f}", end="")
            print()

    # ============================================================
    # Per-round phase transition analysis
    # ============================================================
    print(f"\n{'='*100}")
    print("  PHASE TRANSITION ANALYSIS (per-round hard env loss)")
    print(f"{'='*100}")

    for env in HARD_ENVS:
        print(f"\n  {env}:")
        for method in methods:
            if method == "local_each_nat":
                print(f"    {method:>22}: (single final point, no per-round data)")
                continue
            data = all_results[method]
            losses = [d[f"{env}_eval_loss"] for d in data]
            abs_imps = [losses[i] - losses[i+1] for i in range(len(losses)-1)]
            loss_str = " → ".join(f"{l:.3f}" for l in losses)
            imp_str = " → ".join(f"{d:+.3f}" for d in abs_imps)
            print(f"    {method:>22}: {loss_str}")
            print(f"    {'':>22}  imp: {imp_str}")
            if len(abs_imps) >= 3 and abs_imps[1] < abs_imps[0] and abs_imps[-1] > abs_imps[1]:
                ratio = abs_imps[-1] / max(abs_imps[1], 0.001)
                print(f"    {'':>22}  ★ U-SHAPE detected! Late/early ratio: {ratio:.1f}x")

    # ============================================================
    # KEY COMPARISON: Hard env with/without easy clients
    # ============================================================
    print(f"\n{'='*100}")
    print("  KEY TEST: Does removing easy clients hurt hard envs?")
    print(f"{'='*100}")

    if "fed_5env_nat" in all_results and "fed_3hard_nat" in all_results:
        d5 = all_results["fed_5env_nat"][-1]
        d3 = all_results["fed_3hard_nat"][-1]
        print(f"\n  Hard env final losses (R{num_rounds-1}):")
        for env in HARD_ENVS:
            l5 = d5[f"{env}_eval_loss"]
            l3 = d3[f"{env}_eval_loss"]
            delta = (l3 - l5) / l5 * 100
            verdict = "✓ easy clients HELP" if delta > 5 else ("≈ similar" if abs(delta) < 5 else "✗ no benefit")
            print(f"    {env:>10}: 5env={l5:.4f}  3hard={l3:.4f}  delta={delta:+.1f}%  {verdict}")

        hard_5 = d5["hard_avg"]
        hard_3 = d3["hard_avg"]
        delta = (hard_3 - hard_5) / hard_5 * 100
        print(f"\n    HARD AVG:  5env={hard_5:.4f}  3hard={hard_3:.4f}  delta={delta:+.1f}%")

    if "local_each_nat" in all_results:
        env_results = all_results["local_each_nat"]
        print(f"\n  Local vs Federated (hard envs):")
        for i, env in enumerate(ALL_ENVS):
            local_loss = env_results[i][f"{env}_eval_loss"]
            if "fed_5env_nat" in all_results:
                fed_loss = all_results["fed_5env_nat"][-1][f"{env}_eval_loss"]
                delta = (local_loss - fed_loss) / fed_loss * 100
                print(f"    {env:>10}: local={local_loss:.4f}  fed5={fed_loss:.4f}  fed benefit={delta:+.1f}%")


if __name__ == "__main__":
    main()
