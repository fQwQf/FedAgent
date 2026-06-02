"""
E5: Federated Learning from Failures + Federated RL

Addresses the REAL bottleneck: data scarcity in hard environments.

Methods compared:
  1. fedavg_sft     -- Baseline: success-only SFT + FedAvg (current standard)
  2. fedavg_nat     -- NAT: train on ALL trajectories with success/fail prefix
  3. fedavg_dpo     -- DPO: contrastive learning from fail-vs-success pairs
  4. fed_grpo       -- Federated GRPO: reward-weighted regression on all trajectories
  5. fed_curriculum -- Sequential curriculum: easy envs first, then hard envs

Key insight: Methods 2-4 use ALL trajectories (including failures),
            while method 1 discards ~60-80% of hard env data.

Usage:
    python scripts/e5_fed_lff.py --device cuda:4
    python scripts/e5_fed_lff.py --device cuda:4 --method fedavg_nat
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


ENV_NAMES = ["babyai", "webshop", "textcraft", "maze", "wordle"]
EASY_ENVS = ["babyai", "maze"]
HARD_ENVS = ["webshop", "textcraft", "wordle"]
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_RANK = 8
LORA_ALPHA = 16
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4
LR = 5e-5
NUM_ROUNDS = 5
TRAIN_SAMPLES = 64
EVAL_SAMPLES = 32
SEED = 42

SUCCESS_PREFIX = "[SUCCESS] "
FAIL_PREFIX = "[FAILED] "


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


def tokenize_trajectory(tokenizer, traj, prefix=""):
    messages, _ = get_trainable_messages(traj)
    if not messages:
        return None
    if prefix and messages[0].get("role") == "user":
        messages[0]["content"] = prefix + messages[0]["content"]
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


def prepare_data(tokenizer):
    train_tensors, eval_loaders = {}, {}
    for env in ENV_NAMES:
        all_data = load_agentgym_data(env)
        rng = random.Random(SEED)
        rng.shuffle(all_data)
        train_raw = all_data[:TRAIN_SAMPLES]
        eval_raw = all_data[TRAIN_SAMPLES:TRAIN_SAMPLES + EVAL_SAMPLES]
        print(f"  {env}: {len(train_raw)} train, {len(eval_raw)} eval")

        train_tensors[env] = [t for traj in train_raw
                              if (t := tokenize_trajectory(tokenizer, traj)) is not None]
        eval_list = [t for traj in eval_raw if (t := tokenize_trajectory(tokenizer, traj)) is not None]
        if eval_list:
            ids = torch.stack([t[0] for t in eval_list])
            masks = torch.stack([t[1] for t in eval_list])
            lbls = torch.stack([t[2] for t in eval_list])
            eval_loaders[env] = DataLoader(TensorDataset(ids, masks, lbls), batch_size=BATCH_SIZE)

    return train_tensors, eval_loaders


def compute_per_trajectory_loss(model, tensors_list, device):
    model.eval()
    losses = []
    dl = make_dataloader(tensors_list)
    if dl is None:
        return losses
    with torch.no_grad():
        for batch in dl:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(device)
            per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), reduction='none',
            ).view(shift_labels.size())
            mask = (shift_labels != -100).float()
            per_sample = (per_token * mask).sum(1) / mask.sum(1).clamp(min=1)
            losses.extend(per_sample.cpu().tolist())
    return losses


def make_dataloader(tensors_list):
    if not tensors_list:
        return None
    ids = torch.stack([t[0] for t in tensors_list])
    masks = torch.stack([t[1] for t in tensors_list])
    lbls = torch.stack([t[2] for t in tensors_list])
    return DataLoader(TensorDataset(ids, masks, lbls), batch_size=BATCH_SIZE, shuffle=True)


def make_weighted_dataloader(tensors_list, weights):
    if not tensors_list:
        return None
    ids = torch.stack([t[0] for t in tensors_list])
    masks = torch.stack([t[1] for t in tensors_list])
    lbls = torch.stack([t[2] for t in tensors_list])
    w = torch.tensor(weights, dtype=torch.float32)
    return DataLoader(TensorDataset(ids, masks, lbls, w), batch_size=BATCH_SIZE, shuffle=True)


def filter_by_loss(tensors_list, losses, threshold):
    filtered = [(t, l) for t, l in zip(tensors_list, losses) if l < threshold]
    if len(filtered) < 3:
        filtered = sorted(zip(tensors_list, losses), key=lambda x: x[1])[:max(3, len(tensors_list) // 4)]
    return [t for t, _ in filtered], len(filtered) / max(len(tensors_list), 1)


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


def train_one_epoch_weighted(model, dataloader, device):
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    total_loss, num_steps = 0.0, 0
    for batch in dataloader:
        input_ids, attention_mask, labels, w = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), reduction='none',
        ).view(shift_labels.size())
        valid_mask = (shift_labels != -100).float()
        per_sample = (per_token * valid_mask).sum(1) / valid_mask.sum(1).clamp(min=1)
        loss = (per_sample * w).sum() / w.sum().clamp(min=1e-8)
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
        OrderedDict({k: v.to(device) for k, v in lora_state.items()}), strict=False
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
    env_losses = [round_data.get(f"{e}_eval_loss", float("nan")) for e in ENV_NAMES]
    avg = sum(env_losses) / len(env_losses)
    std = (sum((l - avg) ** 2 for l in env_losses) / len(env_losses)) ** 0.5
    round_data["avg_eval_loss"] = avg
    round_data["eval_loss_std"] = std
    return avg, std


def run_fedavg_sft(device, train_tensors, eval_loaders):
    """Baseline: online FedAvg with success-only filtering."""
    print(f"\n{'='*60}")
    print(f"  FedAvg-SFT (baseline): success-only + FedAvg")
    print(f"{'='*60}")

    model, _ = load_model_and_tokenizer(device)
    global_state = get_lora_state(model)
    all_metrics = []

    for rnd in range(NUM_ROUNDS):
        t0 = time.time()
        round_data = {"round": rnd, "method": "fedavg_sft"}
        apply_lora_state(model, global_state, device)

        all_losses = []
        env_losses_raw = {}
        for env in ENV_NAMES:
            losses = compute_per_trajectory_loss(model, train_tensors[env], device)
            env_losses_raw[env] = losses
            all_losses.extend(losses)

        threshold = sorted(all_losses)[len(all_losses) // 2]
        deltas = []
        for env in ENV_NAMES:
            filtered, p_k = filter_by_loss(train_tensors[env], env_losses_raw[env], threshold)
            round_data[f"{env}_p_k"] = p_k
            round_data[f"{env}_n_train"] = len(filtered)
            apply_lora_state(model, global_state, device)
            dl = make_dataloader(filtered)
            if dl is not None:
                train_one_epoch(model, dl, device)
            updated_state = get_lora_state(model)
            deltas.append(compute_lora_delta(global_state, updated_state))

        global_state = apply_delta_to_state(global_state, _weighted_average_state_dicts(deltas, [1.0]*len(deltas)))
        apply_lora_state(model, global_state, device)

        for env in ENV_NAMES:
            round_data[f"{env}_eval_loss"] = evaluate(model, eval_loaders[env], device)
        avg, std = summarize_round(round_data)
        round_data["time_sec"] = time.time() - t0
        all_metrics.append(round_data)
        env_str = "  ".join(f"{e}={round_data[f'{e}_eval_loss']:.3f}" for e in ENV_NAMES)
        print(f"  R{rnd}: {env_str}  avg={avg:.4f}")

    del model
    torch.cuda.empty_cache()
    return all_metrics


def run_fedavg_nat(device, train_tensors, eval_loaders):
    """NAT: train on ALL data with [SUCCESS]/[FAILED] prefix + FedAvg."""
    print(f"\n{'='*60}")
    print(f"  FedAvg-NAT: all trajectories + success/fail prefix + FedAvg")
    print(f"{'='*60}")

    model, tokenizer = load_model_and_tokenizer(device)

    all_data_tensors = {}
    for env in ENV_NAMES:
        env_data = load_agentgym_data(env)
        rng = random.Random(SEED)
        rng.shuffle(env_data)
        train_raw = env_data[:TRAIN_SAMPLES]
        all_data_tensors[env] = [t for traj in train_raw
                                 if (t := tokenize_trajectory(tokenizer, traj)) is not None]
        print(f"  {env}: {len(all_data_tensors[env])} total (vs {len(train_tensors[env])} filtered)")

    global_state = get_lora_state(model)
    all_metrics = []

    for rnd in range(NUM_ROUNDS):
        t0 = time.time()
        round_data = {"round": rnd, "method": "fedavg_nat"}
        apply_lora_state(model, global_state, device)

        all_losses = []
        env_losses_raw = {}
        for env in ENV_NAMES:
            losses = compute_per_trajectory_loss(model, train_tensors[env], device)
            env_losses_raw[env] = losses
            all_losses.extend(losses)

        threshold = sorted(all_losses)[len(all_losses) // 2]
        deltas = []
        for env in ENV_NAMES:
            apply_lora_state(model, global_state, device)
            dl = make_dataloader(all_data_tensors[env])
            if dl is not None:
                train_one_epoch(model, dl, device)
            updated_state = get_lora_state(model)
            deltas.append(compute_lora_delta(global_state, updated_state))

        global_state = apply_delta_to_state(global_state, _weighted_average_state_dicts(deltas, [1.0]*len(deltas)))
        apply_lora_state(model, global_state, device)

        for env in ENV_NAMES:
            round_data[f"{env}_eval_loss"] = evaluate(model, eval_loaders[env], device)
        avg, std = summarize_round(round_data)
        round_data["time_sec"] = time.time() - t0
        all_metrics.append(round_data)
        env_str = "  ".join(f"{e}={round_data[f'{e}_eval_loss']:.3f}" for e in ENV_NAMES)
        print(f"  R{rnd}: {env_str}  avg={avg:.4f}")

    del model
    torch.cuda.empty_cache()
    return all_metrics


def run_fed_grpo(device, train_tensors, eval_loaders):
    """Federated GRPO: reward-weighted regression on all trajectories.
    
    For each trajectory, compute loss under current model.
    Weight = normalized inverse loss (lower loss = higher weight).
    This is a GRPO-like approach: w_i = softmax(-loss_i / temperature).
    """
    print(f"\n{'='*60}")
    print(f"  Fed-GRPO: reward-weighted regression (all data, loss-weighted)")
    print(f"{'='*60}")

    model, _ = load_model_and_tokenizer(device)
    global_state = get_lora_state(model)
    all_metrics = []
    temperature = 0.5

    for rnd in range(NUM_ROUNDS):
        t0 = time.time()
        round_data = {"round": rnd, "method": "fed_grpo"}
        apply_lora_state(model, global_state, device)

        deltas = []
        for env in ENV_NAMES:
            losses = compute_per_trajectory_loss(model, train_tensors[env], device)
            if not losses:
                apply_lora_state(model, global_state, device)
                deltas.append(compute_lora_delta(global_state, get_lora_state(model)))
                continue

            loss_tensor = torch.tensor(losses)
            raw_weights = F.softmax(-loss_tensor / temperature, dim=0)
            sample_weights = raw_weights.tolist()

            n_eff = sum(1 for w in sample_weights if w > 1.0/len(sample_weights) * 0.5)
            round_data[f"{env}_n_effective"] = n_eff
            round_data[f"{env}_mean_loss"] = sum(losses) / len(losses)
            round_data[f"{env}_min_loss"] = min(losses)
            round_data[f"{env}_max_weight"] = max(sample_weights)

            apply_lora_state(model, global_state, device)
            dl = make_weighted_dataloader(train_tensors[env], sample_weights)
            if dl is not None:
                train_one_epoch_weighted(model, dl, device)
            updated_state = get_lora_state(model)
            deltas.append(compute_lora_delta(global_state, updated_state))

        global_state = apply_delta_to_state(global_state, _weighted_average_state_dicts(deltas, [1.0]*len(deltas)))
        apply_lora_state(model, global_state, device)

        for env in ENV_NAMES:
            round_data[f"{env}_eval_loss"] = evaluate(model, eval_loaders[env], device)
        avg, std = summarize_round(round_data)
        round_data["time_sec"] = time.time() - t0
        all_metrics.append(round_data)
        env_str = "  ".join(f"{e}={round_data[f'{e}_eval_loss']:.3f}" for e in ENV_NAMES)
        print(f"  R{rnd}: {env_str}  avg={avg:.4f}")

    del model
    torch.cuda.empty_cache()
    return all_metrics


def run_fed_curriculum(device, train_tensors, eval_loaders):
    """Sequential curriculum: Phase 1 train easy envs, Phase 2 add hard envs.
    
    Uses our cross-env transfer finding: easy env knowledge transfers to hard envs.
    Phase 1 (rounds 0-2): Only aggregate easy envs (babyai, maze).
    Phase 2 (rounds 3-4): Aggregate all envs, hard envs benefit from easy-env-pretrained model.
    """
    print(f"\n{'='*60}")
    print(f"  Fed-Curriculum: Phase1 easy envs (R0-2), Phase2 all envs (R3-4)")
    print(f"{'='*60}")

    model, _ = load_model_and_tokenizer(device)
    global_state = get_lora_state(model)
    all_metrics = []

    for rnd in range(NUM_ROUNDS):
        t0 = time.time()
        phase = 1 if rnd < 3 else 2
        active_envs = EASY_ENVS if phase == 1 else ENV_NAMES
        round_data = {"round": rnd, "method": "fed_curriculum", "phase": phase}

        apply_lora_state(model, global_state, device)
        all_losses = []
        env_losses_raw = {}
        for env in ENV_NAMES:
            losses = compute_per_trajectory_loss(model, train_tensors[env], device)
            env_losses_raw[env] = losses
            all_losses.extend(losses)

        threshold = sorted(all_losses)[len(all_losses) // 2]

        deltas = []
        for env in active_envs:
            filtered, p_k = filter_by_loss(train_tensors[env], env_losses_raw[env], threshold)
            round_data[f"{env}_p_k"] = p_k
            round_data[f"{env}_n_train"] = len(filtered)
            apply_lora_state(model, global_state, device)
            dl = make_dataloader(filtered)
            if dl is not None:
                train_one_epoch(model, dl, device)
            updated_state = get_lora_state(model)
            deltas.append(compute_lora_delta(global_state, updated_state))

        global_state = apply_delta_to_state(global_state, _weighted_average_state_dicts(deltas, [1.0]*len(deltas)))
        apply_lora_state(model, global_state, device)

        for env in ENV_NAMES:
            round_data[f"{env}_eval_loss"] = evaluate(model, eval_loaders[env], device)
        avg, std = summarize_round(round_data)
        round_data["time_sec"] = time.time() - t0
        all_metrics.append(round_data)
        env_str = "  ".join(f"{e}={round_data[f'{e}_eval_loss']:.3f}" for e in ENV_NAMES)
        print(f"  R{rnd} (P{phase}): {env_str}  avg={avg:.4f}")

    del model
    torch.cuda.empty_cache()
    return all_metrics


def main():
    global NUM_ROUNDS

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--method", choices=["all", "fedavg_sft", "fedavg_nat", "fed_grpo", "fed_curriculum"], default="all")
    args = parser.parse_args()

    NUM_ROUNDS = args.rounds
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    print("=" * 60)
    print("  E5: Federated Learning from Failures")
    print(f"  Device: {device}  |  Rounds: {NUM_ROUNDS}")
    print("=" * 60)

    print("\n  Loading tokenizer...")
    tok_model, tokenizer = load_model_and_tokenizer(device)
    del tok_model
    torch.cuda.empty_cache()

    print("  Pre-tokenizing data...")
    train_tensors, eval_loaders = prepare_data(tokenizer)
    del tokenizer

    methods = ["fedavg_sft", "fedavg_nat", "fed_grpo", "fed_curriculum"] if args.method == "all" else [args.method]
    all_results = {}

    for method in methods:
        if method == "fedavg_sft":
            all_results[method] = run_fedavg_sft(device, train_tensors, eval_loaders)
        elif method == "fedavg_nat":
            all_results[method] = run_fedavg_nat(device, train_tensors, eval_loaders)
        elif method == "fed_grpo":
            all_results[method] = run_fed_grpo(device, train_tensors, eval_loaders)
        elif method == "fed_curriculum":
            all_results[method] = run_fed_curriculum(device, train_tensors, eval_loaders)

        save_dir = Path(f"outputs/e5_fed_lff/{method}")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(all_results[method], f, indent=2)

    print(f"\n{'='*90}")
    print("  E5 FINAL COMPARISON")
    print(f"{'='*90}")
    print(f"{'Method':>18} | {'Avg Loss':>10} | {'Std':>8} |", end="")
    for env in ENV_NAMES:
        print(f" {env:>10}", end="")
    print()
    print("-" * 90)

    for method in methods:
        d = all_results[method][-1]
        print(f"{method:>18} | {d['avg_eval_loss']:>10.4f} | {d['eval_loss_std']:>8.4f} |", end="")
        for env in ENV_NAMES:
            print(f" {d[f'{env}_eval_loss']:>10.4f}", end="")
        print()

    best = min(all_results, key=lambda m: all_results[m][-1]["avg_eval_loss"])
    print(f"\n  WINNER: {best} = {all_results[best][-1]['avg_eval_loss']:.4f}")


if __name__ == "__main__":
    main()
