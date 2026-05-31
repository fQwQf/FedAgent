"""
E2: Combined Online Simulation + Offline Gradient Norm Experiments

Modes:
  online_fedavg    -- simulate online: success filter + FedAvg aggregation
  online_feddebias -- simulate online: success filter + dynamic p_k weighting
  offline_fednorm  -- offline: all data + inverse gradient norm weighting
  all              -- run all three

Usage:
    python scripts/e2_online_offline.py --mode all --device cuda:4
    python scripts/e2_online_offline.py --mode online_feddebias --device cuda:2
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
from src.aggregation import _average_state_dicts, _weighted_average_state_dicts


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


def tokenize_trajectory(tokenizer, traj):
    messages, _ = get_trainable_messages(traj)
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

        train_tensors[env] = []
        for traj in train_raw:
            t = tokenize_trajectory(tokenizer, traj)
            if t is not None:
                train_tensors[env].append(t)

        eval_list = [t for traj in eval_raw if (t := tokenize_trajectory(tokenizer, traj)) is not None]
        if eval_list:
            ids = torch.stack([t[0] for t in eval_list])
            masks = torch.stack([t[1] for t in eval_list])
            lbls = torch.stack([t[2] for t in eval_list])
            eval_loaders[env] = DataLoader(
                TensorDataset(ids, masks, lbls), batch_size=BATCH_SIZE
            )

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


def filter_by_loss(tensors_list, losses, threshold):
    filtered = [(t, l) for t, l in zip(tensors_list, losses) if l < threshold]
    if len(filtered) < 3:
        filtered = sorted(zip(tensors_list, losses), key=lambda x: x[1])[:max(3, len(tensors_list) // 4)]
    return [t for t, _ in filtered], len(filtered) / len(tensors_list)


def make_dataloader(tensors_list):
    if not tensors_list:
        return None
    ids = torch.stack([t[0] for t in tensors_list])
    masks = torch.stack([t[1] for t in tensors_list])
    lbls = torch.stack([t[2] for t in tensors_list])
    return DataLoader(TensorDataset(ids, masks, lbls), batch_size=BATCH_SIZE, shuffle=True)


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


def run_online(method, device, train_tensors, eval_loaders):
    print(f"\n{'='*60}")
    print(f"  ONLINE SIMULATION: {method}  |  Rounds: {NUM_ROUNDS}")
    print(f"{'='*60}")

    model, _ = load_model_and_tokenizer(device)
    global_state = get_lora_state(model)
    all_metrics = []

    for rnd in range(NUM_ROUNDS):
        t0 = time.time()
        round_data = {"round": rnd, "method": method}

        apply_lora_state(model, global_state, device)

        all_losses = []
        env_losses_raw = {}
        for env in ENV_NAMES:
            losses = compute_per_trajectory_loss(model, train_tensors[env], device)
            env_losses_raw[env] = losses
            all_losses.extend(losses)

        threshold = sorted(all_losses)[len(all_losses) // 2]
        round_data["threshold"] = threshold

        deltas, weights = [], []
        for env in ENV_NAMES:
            filtered, p_k = filter_by_loss(train_tensors[env], env_losses_raw[env], threshold)
            round_data[f"{env}_p_k"] = p_k
            round_data[f"{env}_n_train"] = len(filtered)

            apply_lora_state(model, global_state, device)
            dl = make_dataloader(filtered)
            if dl is not None:
                train_loss = train_one_epoch(model, dl, device)
            else:
                train_loss = float("nan")
            round_data[f"{env}_train_loss"] = train_loss

            updated_state = get_lora_state(model)
            delta = compute_lora_delta(global_state, updated_state)
            deltas.append(delta)
            weights.append(p_k)

        p_k_str = "  ".join(f"{e}={round_data[f'{e}_p_k']:.3f}" for e in ENV_NAMES)
        print(f"  Round {rnd}: p_k = {p_k_str}")

        if method == "online_fedavg":
            agg_delta = _average_state_dicts(deltas)
        elif method == "online_feddebias":
            safe_weights = [max(w, 0.01) for w in weights]
            agg_delta = _weighted_average_state_dicts(deltas, safe_weights)
        else:
            raise ValueError(f"Unknown method: {method}")

        global_state = apply_delta_to_state(global_state, agg_delta)
        apply_lora_state(model, global_state, device)

        for env in ENV_NAMES:
            eval_loss = evaluate(model, eval_loaders[env], device)
            round_data[f"{env}_eval_loss"] = eval_loss

        avg, std = summarize_round(round_data)
        round_data["time_sec"] = time.time() - t0
        all_metrics.append(round_data)

        env_str = "  ".join(f"{e}={round_data[f'{e}_eval_loss']:.3f}" for e in ENV_NAMES)
        print(f"           eval: {env_str}  avg={avg:.4f}")

    del model
    torch.cuda.empty_cache()
    return all_metrics


def run_offline_fednorm(device, train_tensors, eval_loaders):
    print(f"\n{'='*60}")
    print(f"  OFFLINE FEDNORM  |  Rounds: {NUM_ROUNDS}")
    print(f"{'='*60}")

    model, _ = load_model_and_tokenizer(device)
    global_state = get_lora_state(model)
    all_metrics = []

    train_dls = {env: make_dataloader(train_tensors[env]) for env in ENV_NAMES}

    for rnd in range(NUM_ROUNDS):
        t0 = time.time()
        round_data = {"round": rnd, "method": "offline_fednorm"}
        deltas, norms = [], []

        for env in ENV_NAMES:
            apply_lora_state(model, global_state, device)
            train_loss = train_one_epoch(model, train_dls[env], device)
            round_data[f"{env}_train_loss"] = train_loss

            updated_state = get_lora_state(model)
            delta = compute_lora_delta(global_state, updated_state)
            norm = sum(v.float().norm().item() ** 2 for v in delta.values()) ** 0.5
            deltas.append(delta)
            norms.append(norm)
            round_data[f"{env}_delta_norm"] = norm

        inv_norms = [1.0 / max(n, 1e-8) for n in norms]
        agg_delta = _weighted_average_state_dicts(deltas, inv_norms)

        norm_str = "  ".join(f"{e}={round_data[f'{e}_delta_norm']:.4f}" for e in ENV_NAMES)
        print(f"  Round {rnd}: delta_norms = {norm_str}")

        global_state = apply_delta_to_state(global_state, agg_delta)
        apply_lora_state(model, global_state, device)

        for env in ENV_NAMES:
            eval_loss = evaluate(model, eval_loaders[env], device)
            round_data[f"{env}_eval_loss"] = eval_loss

        avg, std = summarize_round(round_data)
        round_data["time_sec"] = time.time() - t0
        all_metrics.append(round_data)

        env_str = "  ".join(f"{e}={round_data[f'{e}_eval_loss']:.3f}" for e in ENV_NAMES)
        print(f"           eval: {env_str}  avg={avg:.4f}")

    del model
    torch.cuda.empty_cache()
    return all_metrics


def main():
    global NUM_ROUNDS

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["online_fedavg", "online_feddebias", "offline_fednorm", "all"], default="all")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    args = parser.parse_args()

    NUM_ROUNDS = args.rounds

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    print("=" * 60)
    print("  E2: Online Simulation + Offline Gradient Norm")
    print(f"  Model: {MODEL_NAME}  |  Device: {device}  |  Rounds: {NUM_ROUNDS}")
    print("=" * 60)

    print("\n  Loading tokenizer...")
    tok_model, tokenizer = load_model_and_tokenizer(device)
    del tok_model
    torch.cuda.empty_cache()

    print("  Pre-tokenizing data...")
    train_tensors, eval_loaders = prepare_data(tokenizer)
    del tokenizer

    modes = ["online_fedavg", "online_feddebias", "offline_fednorm"] if args.mode == "all" else [args.mode]
    all_results = {}

    for mode in modes:
        if mode.startswith("online"):
            all_results[mode] = run_online(mode, device, train_tensors, eval_loaders)
        else:
            all_results[mode] = run_offline_fednorm(device, train_tensors, eval_loaders)

        save_dir = Path(f"outputs/e2_{mode}")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(all_results[mode], f, indent=2)
        print(f"  Saved to {save_dir}/metrics.json")

    print(f"\n{'='*60}")
    print("  E2 FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Round':>5}", end="")
    for m in modes:
        print(f" | {'avg':>8} {'std':>8}", end="")
    print()
    for rnd in range(NUM_ROUNDS):
        print(f"{rnd:>5}", end="")
        for m in modes:
            d = all_results[m][rnd]
            print(f" | {d['avg_eval_loss']:>8.4f} {d['eval_loss_std']:>8.4f}", end="")
        print()


if __name__ == "__main__":
    main()
