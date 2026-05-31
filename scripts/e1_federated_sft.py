"""
E1: Federated SFT with p-Debiasing (Optimized)

Compares FedAvg, FedDebias, and Local on 5 AgentGym environments.
Pre-tokenizes all data for speed. Uses Qwen2.5-1.5B-Instruct + LoRA rank=8.

Usage:
    python scripts/e1_federated_sft.py --method fedavg
    python scripts/e1_federated_sft.py --method feddebias
    python scripts/e1_federated_sft.py --method local
    python scripts/e1_federated_sft.py --method all
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import load_agentgym_data, get_trainable_messages
from src.aggregation import _average_state_dicts, _weighted_average_state_dicts


ENV_NAMES = ["babyai", "webshop", "textcraft", "maze", "wordle"]
SUCCESS_RATES = {"babyai": 0.83, "webshop": 0.73, "textcraft": 0.67, "maze": 0.28, "wordle": 0.05}
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_RANK = 8
LORA_ALPHA = 16
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 4
LR = 5e-5
NUM_ROUNDS = 5
TRAIN_SAMPLES = 64
EVAL_SAMPLES = 32
SEED = 42


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
    model.print_trainable_parameters()
    return model, tokenizer


def pretokenize_data(tokenizer):
    train_loaders, eval_loaders = {}, {}
    for env in ENV_NAMES:
        all_data = load_agentgym_data(env)
        rng = random.Random(SEED)
        rng.shuffle(all_data)
        train_raw = all_data[:TRAIN_SAMPLES]
        eval_raw = all_data[TRAIN_SAMPLES:TRAIN_SAMPLES + EVAL_SAMPLES]

        train_loaders[env] = _tokenize_dataset(tokenizer, train_raw)
        eval_loaders[env] = _tokenize_dataset(tokenizer, eval_raw)
        print(f"  {env}: {len(train_raw)} train, {len(eval_raw)} eval")

    return train_loaders, eval_loaders


def _tokenize_dataset(tokenizer, data):
    texts = []
    for traj in data:
        messages, _ = get_trainable_messages(traj)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        if text.strip():
            texts.append(text)

    if not texts:
        return None

    tokenized = tokenizer(
        texts, truncation=True, max_length=MAX_SEQ_LENGTH,
        padding="max_length", return_tensors="pt",
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    dataset = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def train_one_epoch(model, dataloader, device):
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    total_loss, num_steps = 0.0, 0

    for batch in dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        total_loss += loss.item()
        num_steps += 1

    return total_loss / max(num_steps, 1)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss, num_steps = 0.0, 0

    for batch in dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        with torch.no_grad():
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            total_loss += loss.item()
            num_steps += 1

    return total_loss / max(num_steps, 1)


def get_lora_state(model):
    return OrderedDict({
        k: v.clone().cpu() for k, v in model.state_dict().items() if "lora_" in k
    })


def apply_lora_state(model, lora_state, device):
    model.load_state_dict(
        OrderedDict({k: v.to(device) for k, v in lora_state.items()}),
        strict=False,
    )


def compute_lora_delta(old_state, new_state):
    return OrderedDict({
        k: new_state[k].float() - old_state[k].float()
        for k in old_state if k in new_state
    })


def apply_delta_to_state(base_state, delta):
    return OrderedDict({
        k: base_state[k] + delta.get(k, torch.zeros_like(base_state[k]))
        for k in base_state
    })


def summarize_round(round_data):
    env_losses = [round_data.get(f"{e}_global_eval_loss", float("nan")) for e in ENV_NAMES]
    avg = sum(env_losses) / len(env_losses)
    std = (sum((l - avg) ** 2 for l in env_losses) / len(env_losses)) ** 0.5
    round_data["avg_eval_loss"] = avg
    round_data["eval_loss_std"] = std
    return avg, std


def run_federated(method, device, train_loaders, eval_loaders):
    print(f"\n{'='*60}")
    print(f"  Method: {method}  |  Rounds: {NUM_ROUNDS}  |  Device: {device}")
    print(f"{'='*60}")

    model, _ = load_model_and_tokenizer(device)
    global_state = get_lora_state(model)
    all_metrics = []

    for rnd in range(NUM_ROUNDS):
        t0 = time.time()
        round_data = {"round": rnd, "method": method}
        deltas, weights = [], []

        for env in ENV_NAMES:
            apply_lora_state(model, global_state, device)
            train_loss = train_one_epoch(model, train_loaders[env], device)
            updated_state = get_lora_state(model)
            delta = compute_lora_delta(global_state, updated_state)
            deltas.append(delta)
            weights.append(SUCCESS_RATES[env])
            round_data[f"{env}_train_loss"] = train_loss

        if method == "fedavg":
            agg_delta = _average_state_dicts(deltas)
        elif method == "feddebias":
            agg_delta = _weighted_average_state_dicts(deltas, weights)
        else:
            raise ValueError(f"Unknown method: {method}")

        global_state = apply_delta_to_state(global_state, agg_delta)
        apply_lora_state(model, global_state, device)

        for env in ENV_NAMES:
            eval_loss = evaluate(model, eval_loaders[env], device)
            round_data[f"{env}_global_eval_loss"] = eval_loss
            print(f"    {env:>12}: train={round_data[f'{env}_train_loss']:.4f}  eval={eval_loss:.4f}")

        avg, std = summarize_round(round_data)
        round_data["time_sec"] = time.time() - t0
        all_metrics.append(round_data)
        print(f"  Round {rnd:2d}: avg={avg:.4f} +/- {std:.4f}  ({round_data['time_sec']:.0f}s)")

    del model
    torch.cuda.empty_cache()
    return all_metrics


def run_local(device, train_loaders, eval_loaders):
    print(f"\n{'='*60}")
    print(f"  Method: local (independent)  |  Rounds: {NUM_ROUNDS}  |  Device: {device}")
    print(f"{'='*60}")

    env_models, shared_tok = {}, None
    for env in ENV_NAMES:
        m, t = load_model_and_tokenizer(device)
        env_models[env] = m
        shared_tok = t

    all_metrics = []
    for rnd in range(NUM_ROUNDS):
        t0 = time.time()
        round_data = {"round": rnd, "method": "local"}

        for env in ENV_NAMES:
            train_loss = train_one_epoch(env_models[env], train_loaders[env], device)
            eval_loss = evaluate(env_models[env], eval_loaders[env], device)
            round_data[f"{env}_train_loss"] = train_loss
            round_data[f"{env}_global_eval_loss"] = eval_loss
            print(f"    {env:>12}: train={train_loss:.4f}  eval={eval_loss:.4f}")

        avg, std = summarize_round(round_data)
        round_data["time_sec"] = time.time() - t0
        all_metrics.append(round_data)
        print(f"  Round {rnd:2d}: avg={avg:.4f} +/- {std:.4f}  ({round_data['time_sec']:.0f}s)")

    for m in env_models.values():
        del m
    torch.cuda.empty_cache()
    return all_metrics


def main():
    global NUM_ROUNDS

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["fedavg", "feddebias", "local", "all"], default="all")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    args = parser.parse_args()

    NUM_ROUNDS = args.rounds
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    print("=" * 60)
    print("  E1: Federated SFT with p-Debiasing")
    print(f"  Model: {MODEL_NAME}  |  Device: {device}")
    print(f"  Rounds: {NUM_ROUNDS}  |  Train/eval: {TRAIN_SAMPLES}/{EVAL_SAMPLES}")
    print(f"  p_k = {SUCCESS_RATES}")
    print("=" * 60)

    print("\n  Loading tokenizer...")
    tok_model, tokenizer = load_model_and_tokenizer(device)
    del tok_model
    torch.cuda.empty_cache()

    print("\n  Pre-tokenizing data...")
    train_loaders, eval_loaders = pretokenize_data(tokenizer)
    del tokenizer

    methods = ["fedavg", "feddebias", "local"] if args.method == "all" else [args.method]
    all_results = {}

    for method in methods:
        if method == "local":
            all_results[method] = run_local(device, train_loaders, eval_loaders)
        else:
            all_results[method] = run_federated(method, device, train_loaders, eval_loaders)

        save_dir = Path(f"outputs/e1_{method}")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(all_results[method], f, indent=2)
        print(f"  Saved to {save_dir}/metrics.json")

    print(f"\n{'='*60}")
    print("  FINAL COMPARISON")
    print(f"{'='*60}")
    header = f"{'Round':>5}"
    for m in methods:
        header += f" | {'avg_'+m:>14} {'std_'+m:>10}"
    print(header)
    print("-" * len(header))
    for rnd in range(NUM_ROUNDS):
        row = f"{rnd:>5}"
        for m in methods:
            d = all_results[m][rnd]
            row += f" | {d['avg_eval_loss']:>14.4f} {d['eval_loss_std']:>10.4f}"
        print(row)


if __name__ == "__main__":
    main()
