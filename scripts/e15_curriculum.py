"""
E15: Curriculum-Guided Federated Training for LLM Agents

Direction B: Train tasks in curriculum order (easy → medium → hard).
Rationale: Easy tasks establish basic agent capabilities; harder tasks
build on this foundation without gradient conflict from simultaneous
training on all tasks.

Curriculum:
  Phase 1 (R0-R6): babyai, maze          (easy: navigation, simple reasoning)
  Phase 2 (R7-R12): + wordle             (medium: language puzzle)
  Phase 3 (R13-R19): + webshop, textcraft (hard: complex reasoning)

Baseline: All 5 tasks trained simultaneously from R0.

Usage:
    python scripts/e15_curriculum.py --model_size 0.5B --device cuda:3
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

# Curriculum definition
EASY_ENVS = ["babyai", "maze"]
MEDIUM_ENVS = ["wordle"]
HARD_ENVS_CURR = ["webshop", "textcraft"]

# Phase boundaries
PHASE_1_END = 6    # R0-R6: easy only
PHASE_2_END = 12   # R7-R12: easy + medium
PHASE_3_END = 20   # R13-R19: all tasks

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


def get_active_tasks(round_num):
    """Return active tasks for this round based on curriculum."""
    if round_num < PHASE_1_END:
        return EASY_ENVS
    elif round_num < PHASE_2_END:
        return EASY_ENVS + MEDIUM_ENVS
    else:
        return ALL_ENVS


def run_curriculum(device, train_dl, eval_dl, num_rounds, model_name):
    print(f"\n{'='*70}")
    print(f"  Curriculum  (easy→medium→hard, {num_rounds}R, {model_name})")
    print(f"{'='*70}")
    print(f"  Phase 1 (R0-R{PHASE_1_END-1}): {EASY_ENVS}")
    print(f"  Phase 2 (R{PHASE_1_END}-R{PHASE_2_END-1}): {EASY_ENVS + MEDIUM_ENVS}")
    print(f"  Phase 3 (R{PHASE_2_END}-R{num_rounds-1}): {ALL_ENVS}")
    print(f"{'='*70}")

    model, _ = load_model_and_tokenizer(device, model_name)
    global_state = get_lora_state(model)
    metrics = []

    for rnd in range(num_rounds):
        t0 = time.time()
        rd = {"round": rnd, "method": "curriculum"}
        active_tasks = get_active_tasks(rnd)

        # Phase indicator
        if rnd < PHASE_1_END:
            phase = 1
        elif rnd < PHASE_2_END:
            phase = 2
        else:
            phase = 3
        rd["phase"] = phase
        rd["active_tasks"] = active_tasks

        # Eval ALL tasks (to track progress on inactive tasks too)
        apply_lora_state(model, global_state, device)
        eval_losses = {}
        for env in ALL_ENVS:
            eval_losses[env] = evaluate(model, eval_dl[env], device)
            rd[f"{env}_eval_loss"] = eval_losses[env]

        # Client updates: only active tasks train
        deltas = {}
        for env in active_tasks:
            apply_lora_state(model, global_state, device)
            train_client(model, train_dl[env], device)
            new_state = get_lora_state(model)
            deltas[env] = OrderedDict({
                k: new_state[k].float() - global_state[k].float() for k in global_state})

        # Aggregate: uniform average of active task deltas
        delta_list = [deltas[env] for env in active_tasks]
        avg_delta = OrderedDict()
        keys = list(global_state.keys())
        for key in keys:
            tensors = [d[key].float() for d in delta_list]
            avg_delta[key] = torch.stack(tensors).mean(dim=0)

        # Update global state
        global_state = OrderedDict({k: global_state[k] + avg_delta[k] for k in global_state})
        apply_lora_state(model, global_state, device)

        # Final eval on ALL tasks
        for env in ALL_ENVS:
            rd[f"{env}_eval_after"] = evaluate(model, eval_dl[env], device)

        hard_avg = sum(rd[f"{e}_eval_after"] for e in HARD_ENVS) / len(HARD_ENVS)
        all_avg = sum(rd[f"{e}_eval_after"] for e in ALL_ENVS) / len(ALL_ENVS)
        rd["hard_avg"] = hard_avg
        rd["avg_eval_loss"] = all_avg
        rd["time_sec"] = time.time() - t0

        metrics.append(rd)

        # Compact output
        loss_str = " ".join(f"{e[:3]}={rd[f'{e}_eval_after']:.3f}" for e in ALL_ENVS)
        print(f"  R{rnd:>2} [P{phase}]: {loss_str}  hard={hard_avg:.3f}  active={active_tasks}")

        if rnd % 5 == 4:
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return metrics


def run_baseline(device, train_dl, eval_dl, num_rounds, model_name):
    """Baseline: all tasks trained simultaneously from R0 (uniform FedAvg)."""
    print(f"\n{'='*70}")
    print(f"  Baseline  (all tasks simultaneous, {num_rounds}R, {model_name})")
    print(f"{'='*70}")

    model, _ = load_model_and_tokenizer(device, model_name)
    global_state = get_lora_state(model)
    metrics = []

    for rnd in range(num_rounds):
        t0 = time.time()
        rd = {"round": rnd, "method": "baseline"}

        # Eval
        apply_lora_state(model, global_state, device)
        eval_losses = {}
        for env in ALL_ENVS:
            eval_losses[env] = evaluate(model, eval_dl[env], device)
            rd[f"{env}_eval_loss"] = eval_losses[env]

        # Client updates: ALL tasks
        deltas = {}
        for env in ALL_ENVS:
            apply_lora_state(model, global_state, device)
            train_client(model, train_dl[env], device)
            new_state = get_lora_state(model)
            deltas[env] = OrderedDict({
                k: new_state[k].float() - global_state[k].float() for k in global_state})

        # Uniform aggregate
        delta_list = [deltas[env] for env in ALL_ENVS]
        avg_delta = OrderedDict()
        for key in global_state.keys():
            tensors = [d[key].float() for d in delta_list]
            avg_delta[key] = torch.stack(tensors).mean(dim=0)

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

        loss_str = " ".join(f"{e[:3]}={rd[f'{e}_eval_after']:.3f}" for e in ALL_ENVS)
        print(f"  R{rnd:>2}: {loss_str}  hard={hard_avg:.3f}")

        if rnd % 5 == 4:
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--model_size", default="0.5B", choices=["0.5B", "1.5B"])
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--method", default="all", choices=["all", "curriculum", "baseline"])
    args = parser.parse_args()

    device = args.device
    model_name = MODEL_MAP[args.model_size]
    set_seed(SEED)

    print("=" * 70)
    print(f"  E15: Curriculum-Guided Federated Training")
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
               else ["curriculum", "baseline"])

    all_results = {}
    for m in methods:
        if m == "curriculum":
            all_results[m] = run_curriculum(device, train_dl, eval_dl, args.rounds, model_name)
        else:
            all_results[m] = run_baseline(device, train_dl, eval_dl, args.rounds, model_name)

        save_dir = Path(f"outputs/e15_curriculum/{args.model_size}/{m}")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(all_results[m], f, indent=2)

    # ================================================================
    # ANALYSIS
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  E15 RESULTS COMPARISON ({args.model_size}, {args.rounds}R)")
    print(f"{'='*70}")

    for m in methods:
        final = all_results[m][-1]
        print(f"\n  {m}:")
        print(f"    hard_avg = {final['hard_avg']:.4f}")
        print(f"    avg_loss = {final['avg_eval_loss']:.4f}")
        for e in ALL_ENVS:
            print(f"    {e:10s} = {final[f'{e}_eval_after']:.4f}")

    # Compare curriculum vs baseline
    if "curriculum" in all_results and "baseline" in all_results:
        curr_hard = all_results["curriculum"][-1]["hard_avg"]
        base_hard = all_results["baseline"][-1]["hard_avg"]
        delta = curr_hard - base_hard
        pct = delta / base_hard * 100 if base_hard != 0 else 0
        print(f"\n  Curriculum vs Baseline: {delta:+.4f} ({pct:+.1f}%) on hard_avg")

        # Track hard task progression
        print(f"\n  Hard tasks progression (webshop + textcraft + wordle):")
        for m in ["baseline", "curriculum"]:
            if m in all_results:
                hard_series = [sum(d[f"{e}_eval_after"] for e in HARD_ENVS) / len(HARD_ENVS)
                               for d in all_results[m]]
                sampled = [(i, hard_series[i]) for i in range(0, len(hard_series), 5)]
                sampled_str = "  ".join(f"R{i}={v:.3f}" for i, v in sampled)
                print(f"    {m:>10}: {sampled_str}")

    # Track inactive task performance in curriculum
    if "curriculum" in all_results:
        print(f"\n  Inactive task eval in curriculum (before they join):")
        for env in ALL_ENVS:
            first_active = None
            for i, d in enumerate(all_results["curriculum"]):
                if env in d["active_tasks"]:
                    first_active = i
                    break
            if first_active and first_active > 0:
                before = all_results["curriculum"][first_active - 1][f"{env}_eval_after"]
                after = all_results["curriculum"][first_active][f"{env}_eval_after"]
                print(f"    {env:10s}: before join (R{first_active-1})={before:.3f}, "
                      f"after first train (R{first_active})={after:.3f}")

    print(f"\n  Results saved to outputs/e15_curriculum/{args.model_size}/")


if __name__ == "__main__":
    main()
