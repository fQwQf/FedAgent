"""
E6: Plausibility-Gated Federated Learning (PGFL)

Tests the theory: optimal failure utilization shifts from filtering to 
full utilization as model quality improves during federated training.

Methods:
  1. fedavg_sft     -- Baseline: loss-threshold filtered SFT + FedAvg
  2. fedavg_nat     -- All data, uniform weights + FedAvg
  3. fedavg_pgfl    -- Plausibility-gated weights + FedAvg
  4. fedavg_nat_noise  -- All data (incl. synthetic garbage) + FedAvg
  5. fedavg_pgfl_noise -- Plausibility-gated (incl. synthetic garbage) + FedAvg

Key predictions:
  - PGFL ≈ NAT on clean data (all trajectories are plausible)
  - NAT+noise WORSE than NAT (garbage trajectories hurt)
  - PGFL+noise ≈ NAT (plausibility gate filters out garbage)
  - Plausibility weights increase monotonically over rounds
  - Hard envs benefit most from PGFL (they need all plausible data)

Usage:
    python scripts/e6_pgfl.py --device cuda:4
    python scripts/e6_pgfl.py --device cuda:4 --method fedavg_pgfl
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

# PGFL hyperparameters
PGFL_TEMPERATURE = 0.3  # Temperature for plausibility gate
PGFL_THRESHOLD = 0.0    # Log-ratio threshold (0 = same plausibility as successes)

# Noise injection
NOISE_FRACTION = 0.25   # Fraction of trajectories to corrupt
NOISE_TOKEN_PROB = 0.4  # Probability of replacing each action token


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


def make_noisy_trajectory(traj, tokenizer, noise_prob=NOISE_TOKEN_PROB):
    """Create a noisy version of a trajectory by corrupting action tokens.
    
    This simulates an 'implausible failure' — a trajectory with wrong actions
    that the model should learn to avoid, not imitate.
    """
    import copy
    noisy = copy.deepcopy(traj)
    vocab_size = len(tokenizer)
    
    for conv in noisy["conversations"]:
        if conv.get("loss") is True and conv["from"] == "gpt":
            tokens = tokenizer.encode(conv["value"], add_special_tokens=False)
            corrupted = []
            for t in tokens:
                if random.random() < noise_prob:
                    corrupted.append(random.randint(100, min(vocab_size - 1, 50000)))
                else:
                    corrupted.append(t)
            conv["value"] = tokenizer.decode(corrupted)
    
    return noisy


def prepare_data(tokenizer, inject_noise=False):
    """Prepare training and evaluation data.
    
    If inject_noise=True, replaces NOISE_FRACTION of trajectories with
    corrupted versions (simulating online failures).
    """
    train_tensors = {}
    noise_flags = {}  # Track which trajectories are noisy
    eval_loaders = {}
    
    for env in ENV_NAMES:
        all_data = load_agentgym_data(env)
        rng = random.Random(SEED)
        rng.shuffle(all_data)
        train_raw = all_data[:TRAIN_SAMPLES]
        eval_raw = all_data[TRAIN_SAMPLES:TRAIN_SAMPLES + EVAL_SAMPLES]
        
        if inject_noise:
            # Replace a fraction of trajectories with noisy versions
            n_noisy = int(NOISE_FRACTION * len(train_raw))
            noisy_indices = rng.sample(range(len(train_raw)), n_noisy)
            train_processed = []
            is_noisy = []
            for i, traj in enumerate(train_raw):
                if i in noisy_indices:
                    noisy_traj = make_noisy_trajectory(traj, tokenizer)
                    t = tokenize_trajectory(tokenizer, noisy_traj)
                    train_processed.append(t if t is not None else tokenize_trajectory(tokenizer, traj))
                    is_noisy.append(True)
                else:
                    t = tokenize_trajectory(tokenizer, traj)
                    train_processed.append(t)
                    is_noisy.append(False)
        else:
            train_processed = [tokenize_trajectory(tokenizer, traj) for traj in train_raw]
            is_noisy = [False] * len(train_raw)
        
        train_tensors[env] = [t for t in train_processed if t is not None]
        noise_flags[env] = is_noisy[:len(train_tensors[env])]
        
        eval_list = [t for t in (tokenize_trajectory(tokenizer, traj) for traj in eval_raw) if t is not None]
        if eval_list:
            ids = torch.stack([t[0] for t in eval_list])
            masks = torch.stack([t[1] for t in eval_list])
            lbls = torch.stack([t[2] for t in eval_list])
            eval_loaders[env] = DataLoader(TensorDataset(ids, masks, lbls), batch_size=BATCH_SIZE)
        
        n_noisy_count = sum(noise_flags[env])
        print(f"  {env}: {len(train_tensors[env])} train ({n_noisy_count} noisy), {len(eval_list)} eval")
    
    return train_tensors, noise_flags, eval_loaders


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


def compute_plausibility_weights(losses, success_threshold, temperature=PGFL_TEMPERATURE):
    """Compute plausibility-gated weights for all trajectories.
    
    For each trajectory τ:
      - Compute plausibility ratio: ρ(τ) = exp(-loss(τ)) / exp(-success_threshold)
        = exp(success_threshold - loss(τ))
      - Apply sigmoid gate: w(τ) = σ((log ρ - η) / T)
        = σ((success_threshold - loss(τ) - η) / T)
    
    Successes (loss < threshold) get high weights automatically.
    Failures (loss > threshold) get weights proportional to their plausibility.
    """
    log_plausibility_ratios = [(success_threshold - l) for l in losses]
    
    # Sigmoid gate
    weights = [torch.sigmoid(torch.tensor((lr - PGFL_THRESHOLD) / temperature)).item() 
               for lr in log_plausibility_ratios]
    
    return weights


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


# ============================================================
# Method 1: FedAvg-SFT (baseline, filtered)
# ============================================================
def run_fedavg_sft(device, train_tensors, eval_loaders):
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


# ============================================================
# Method 2: FedAvg-NAT (all data, uniform)
# ============================================================
def run_fedavg_nat(device, train_tensors, eval_loaders):
    print(f"\n{'='*60}")
    print(f"  FedAvg-NAT: all data, uniform weights + FedAvg")
    print(f"{'='*60}")

    model, _ = load_model_and_tokenizer(device)
    global_state = get_lora_state(model)
    all_metrics = []

    for rnd in range(NUM_ROUNDS):
        t0 = time.time()
        round_data = {"round": rnd, "method": "fedavg_nat"}
        apply_lora_state(model, global_state, device)

        deltas = []
        for env in ENV_NAMES:
            apply_lora_state(model, global_state, device)
            dl = make_dataloader(train_tensors[env])
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


# ============================================================
# Method 3: PGFL (plausibility-gated weights)
# ============================================================
def run_fedavg_pgfl(device, train_tensors, eval_loaders, label="fedavg_pgfl"):
    print(f"\n{'='*60}")
    print(f"  PGFL: plausibility-gated weights (T={PGFL_TEMPERATURE}) + FedAvg")
    print(f"{'='*60}")

    model, _ = load_model_and_tokenizer(device)
    global_state = get_lora_state(model)
    all_metrics = []

    for rnd in range(NUM_ROUNDS):
        t0 = time.time()
        round_data = {"round": rnd, "method": label, "temperature": PGFL_TEMPERATURE}
        apply_lora_state(model, global_state, device)

        deltas = []
        for env in ENV_NAMES:
            losses = compute_per_trajectory_loss(model, train_tensors[env], device)
            if not losses:
                apply_lora_state(model, global_state, device)
                deltas.append(compute_lora_delta(global_state, get_lora_state(model)))
                continue

            # Use median loss as success threshold
            success_threshold = sorted(losses)[len(losses) // 2]
            
            # Compute plausibility-gated weights
            weights = compute_plausibility_weights(losses, success_threshold, PGFL_TEMPERATURE)
            
            # Log statistics
            n_high = sum(1 for w in weights if w > 0.5)
            n_low = sum(1 for w in weights if w < 0.1)
            mean_w = sum(weights) / len(weights)
            round_data[f"{env}_n_high_plaus"] = n_high
            round_data[f"{env}_n_low_plaus"] = n_low
            round_data[f"{env}_mean_weight"] = mean_w
            round_data[f"{env}_success_thresh"] = success_threshold

            apply_lora_state(model, global_state, device)
            dl = make_weighted_dataloader(train_tensors[env], weights)
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
        n_high_total = sum(round_data.get(f"{e}_n_high_plaus", 0) for e in ENV_NAMES)
        print(f"  R{rnd}: {env_str}  avg={avg:.4f}  high_plaus={n_high_total}/320")

    del model
    torch.cuda.empty_cache()
    return all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--method", choices=["all", "fedavg_sft", "fedavg_nat", "fedavg_pgfl",
                                            "fedavg_nat_noise", "fedavg_pgfl_noise"], default="all")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    print("=" * 60)
    print(f"  E6: Plausibility-Gated Federated Learning (PGFL)")
    print(f"  Device: {device}  |  Rounds: {args.rounds}  |  T={PGFL_TEMPERATURE}")
    print("=" * 60)

    print("\n  Loading tokenizer...")
    tok_model, tokenizer = load_model_and_tokenizer(device)
    del tok_model
    torch.cuda.empty_cache()

    print("  Pre-tokenizing clean data...")
    train_tensors_clean, _, eval_loaders = prepare_data(tokenizer, inject_noise=False)
    
    print("\n  Pre-tokenizing noisy data...")
    train_tensors_noisy, noise_flags, _ = prepare_data(tokenizer, inject_noise=True)
    
    del tokenizer

    all_results = {}
    methods = args.method.split(",") if args.method != "all" else [
        "fedavg_sft", "fedavg_nat", "fedavg_pgfl",
        "fedavg_nat_noise", "fedavg_pgfl_noise"
    ]

    for method in methods:
        if method == "fedavg_sft":
            all_results[method] = run_fedavg_sft(device, train_tensors_clean, eval_loaders)
        elif method == "fedavg_nat":
            all_results[method] = run_fedavg_nat(device, train_tensors_clean, eval_loaders)
        elif method == "fedavg_pgfl":
            all_results[method] = run_fedavg_pgfl(device, train_tensors_clean, eval_loaders, label="fedavg_pgfl")
        elif method == "fedavg_nat_noise":
            all_results[method] = run_fedavg_nat(device, train_tensors_noisy, eval_loaders)
        elif method == "fedavg_pgfl_noise":
            all_results[method] = run_fedavg_pgfl(device, train_tensors_noisy, eval_loaders, label="fedavg_pgfl_noise")

        save_dir = Path(f"outputs/e6_pgfl/{method}")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(all_results[method], f, indent=2)

    print(f"\n{'='*100}")
    print("  E6 FINAL COMPARISON")
    print(f"{'='*100}")
    print(f"{'Method':>22} | {'Avg Loss':>10} | {'Std':>8} |", end="")
    for env in ENV_NAMES:
        print(f" {env:>10}", end="")
    print()
    print("-" * 100)

    for method in methods:
        d = all_results[method][-1]
        print(f"{method:>22} | {d['avg_eval_loss']:>10.4f} | {d['eval_loss_std']:>8.4f} |", end="")
        for env in ENV_NAMES:
            print(f" {d[f'{env}_eval_loss']:>10.4f}", end="")
        print()

    best = min(all_results, key=lambda m: all_results[m][-1]["avg_eval_loss"])
    print(f"\n  WINNER: {best} = {all_results[best][-1]['avg_eval_loss']:.4f}")

    # Print plausibility statistics for PGFL methods
    for method in ["fedavg_pgfl", "fedavg_pgfl_noise"]:
        if method in all_results:
            print(f"\n  {method} plausibility evolution:")
            for rnd_data in all_results[method]:
                r = rnd_data["round"]
                stats = []
                for env in ENV_NAMES:
                    nh = rnd_data.get(f"{env}_n_high_plaus", "?")
                    mw = rnd_data.get(f"{env}_mean_weight", "?")
                    if isinstance(nh, int):
                        stats.append(f"{env[:4]}:high={nh}/64,w={mw:.3f}")
                if stats:
                    print(f"    R{r}: {' | '.join(stats)}")


if __name__ == "__main__":
    main()
