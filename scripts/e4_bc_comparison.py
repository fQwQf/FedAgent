"""
E4: Option B (Per-client Adaptive Alpha) vs Option C (Reward-Weighted SFT)

Option B: alpha_k = 1 - hat_p_k, where hat_p_k = ||g_k|| / max_k||g_k||
  -> Easy envs (high p_k, large norm): alpha_k ~ 0 -> FedAvg (no normalization)
  -> Hard envs (low p_k, small norm): alpha_k ~ 1 -> FedNorm (full normalization)
  Weight: w_k = ||g_k||^{-alpha_k}

Option C: Soft filtering with temperature tau instead of hard success/fail
  -> Weight each trajectory by exp(-loss / tau) instead of binary keep/discard
  -> tau -> 0: hard filter (keep only low-loss), equivalent to current approach
  -> tau -> inf: keep all, equal weight (standard SFT, no 1/p_k amplification)
  -> Intermediate tau: partial 1/p_k effect, controlled by temperature

Baseline: FedAvg (alpha=0, hard filter)

Usage:
    python scripts/e4_bc_comparison.py --device cuda:4
    python scripts/e4_bc_comparison.py --device cuda:4 --method option_b
    python scripts/e4_bc_comparison.py --device cuda:4 --method option_c
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


def make_dataloader(tensors_list):
    if not tensors_list:
        return None
    ids = torch.stack([t[0] for t in tensors_list])
    masks = torch.stack([t[1] for t in tensors_list])
    lbls = torch.stack([t[2] for t in tensors_list])
    return DataLoader(TensorDataset(ids, masks, lbls), batch_size=BATCH_SIZE, shuffle=True)


def train_one_epoch_weighted(model, tensors_list, sample_weights, device):
    """Train with per-sample weights (for Option C soft filtering)."""
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    if not tensors_list:
        return float("nan")

    ids = torch.stack([t[0] for t in tensors_list])
    masks = torch.stack([t[1] for t in tensors_list])
    lbls = torch.stack([t[2] for t in tensors_list])
    weights = torch.tensor(sample_weights, dtype=torch.float32)

    dataset = TensorDataset(ids, masks, lbls, weights)
    dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    total_loss, num_steps = 0.0, 0
    for batch in dl:
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


def filter_by_loss(tensors_list, losses, threshold):
    filtered = [(t, l) for t, l in zip(tensors_list, losses) if l < threshold]
    if len(filtered) < 3:
        filtered = sorted(zip(tensors_list, losses), key=lambda x: x[1])[:max(3, len(tensors_list) // 4)]
    return [t for t, _ in filtered], len(filtered) / max(len(tensors_list), 1)


def run_option_b(device, train_tensors, eval_loaders):
    """Option B: Per-client adaptive alpha_k = 1 - hat_p_k.
    
    hat_p_k estimated from gradient norms: hat_p_k = ||g_k|| / max(||g_k||).
    Weight: w_k = ||g_k||^{-alpha_k} where alpha_k = 1 - hat_p_k.
    
    Intuition:
      - high p_k env (large norm) -> alpha_k ~ 0 -> uniform weight (FedAvg-like)
      - low p_k env (small norm) -> alpha_k ~ 1 -> inverse-norm weight (FedNorm-like)
    """
    print(f"\n{'='*60}")
    print(f"  OPTION B: Per-client Adaptive Alpha  |  Rounds: {NUM_ROUNDS}")
    print(f"  alpha_k = 1 - ||g_k||/max(||g||)")
    print(f"  w_k = ||g_k||^{{-alpha_k}}")
    print(f"{'='*60}")

    model, _ = load_model_and_tokenizer(device)
    global_state = get_lora_state(model)
    all_metrics = []

    for rnd in range(NUM_ROUNDS):
        t0 = time.time()
        round_data = {"round": rnd, "method": "option_b_adaptive"}

        apply_lora_state(model, global_state, device)

        all_losses = []
        env_losses_raw = {}
        for env in ENV_NAMES:
            losses = compute_per_trajectory_loss(model, train_tensors[env], device)
            env_losses_raw[env] = losses
            all_losses.extend(losses)

        threshold = sorted(all_losses)[len(all_losses) // 2]
        round_data["threshold"] = threshold

        deltas, norms = [], []
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
            norm = sum(v.float().norm().item() ** 2 for v in delta.values()) ** 0.5
            deltas.append(delta)
            norms.append(norm)
            round_data[f"{env}_delta_norm"] = norm

        max_norm = max(norms)
        hat_pks = [n / max_norm for n in norms]
        alphas_k = [1.0 - hp for hp in hat_pks]
        weights = [n ** (-a) if a > 0 else 1.0 for n, a in zip(norms, alphas_k)]

        for i, env in enumerate(ENV_NAMES):
            round_data[f"{env}_hat_pk"] = hat_pks[i]
            round_data[f"{env}_alpha_k"] = alphas_k[i]
            round_data[f"{env}_weight"] = weights[i]

        agg_delta = _weighted_average_state_dicts(deltas, weights)

        detail = "  ".join(
            f"{e}: p_k={round_data[f'{e}_p_k']:.3f} "
            f"||g||={round_data[f'{e}_delta_norm']:.4f} "
            f"hat_p={round_data[f'{e}_hat_pk']:.3f} "
            f"a_k={round_data[f'{e}_alpha_k']:.3f} "
            f"w={round_data[f'{e}_weight']:.4f}"
            for e in ENV_NAMES
        )
        print(f"  Round {rnd}: {detail}")

        global_state = apply_delta_to_state(global_state, agg_delta)
        apply_lora_state(model, global_state, device)

        for env in ENV_NAMES:
            eval_loss = evaluate(model, eval_loaders[env], device)
            round_data[f"{env}_eval_loss"] = eval_loss

        avg, std = summarize_round(round_data)
        round_data["time_sec"] = time.time() - t0
        all_metrics.append(round_data)

        env_str = "  ".join(f"{e}={round_data[f'{e}_eval_loss']:.3f}" for e in ENV_NAMES)
        print(f"           eval: {env_str}  avg={avg:.4f}  std={std:.4f}")

    del model
    torch.cuda.empty_cache()
    return all_metrics


def run_option_c(device, train_tensors, eval_loaders, tau_values):
    """Option C: Reward-weighted SFT with soft filtering.
    
    Instead of hard binary filter (keep if loss < threshold), use
    soft weights: w_i = exp(-loss_i / tau)
    
    tau -> 0: hard filter (approximate current approach)
    tau -> inf: uniform weights (standard SFT, no 1/p_k effect)
    """

    results_by_tau = {}

    for tau in tau_values:
        method_name = f"option_c_tau{tau:.3f}"
        print(f"\n{'='*60}")
        print(f"  OPTION C: Soft Filtering  tau={tau}  |  Rounds: {NUM_ROUNDS}")
        print(f"  w_i = exp(-loss_i / {tau})")
        print(f"{'='*60}")

        model, _ = load_model_and_tokenizer(device)
        global_state = get_lora_state(model)
        all_metrics = []

        for rnd in range(NUM_ROUNDS):
            t0 = time.time()
            round_data = {"round": rnd, "method": method_name, "tau": tau}

            apply_lora_state(model, global_state, device)

            env_losses_raw = {}
            for env in ENV_NAMES:
                losses = compute_per_trajectory_loss(model, train_tensors[env], device)
                env_losses_raw[env] = losses

            deltas, norms = [], []
            for env in ENV_NAMES:
                losses = env_losses_raw[env]
                if tau < 1e-6:
                    sample_weights = [1.0 if l < sorted(losses)[len(losses)//2] else 0.0 for l in losses]
                else:
                    max_loss = max(losses) if losses else 1.0
                    raw_w = [torch.exp(torch.tensor(-l / tau)).item() for l in losses]
                    w_sum = sum(raw_w)
                    sample_weights = [w / w_sum * len(raw_w) for w in raw_w] if w_sum > 0 else [1.0] * len(losses)

                n_effective = sum(1 for w in sample_weights if w > 0.1)
                p_k_est = n_effective / len(sample_weights) if sample_weights else 0
                round_data[f"{env}_p_k_est"] = p_k_est
                round_data[f"{env}_n_effective"] = n_effective
                round_data[f"{env}_mean_weight"] = sum(sample_weights) / len(sample_weights)

                apply_lora_state(model, global_state, device)
                train_loss = train_one_epoch_weighted(model, train_tensors[env], sample_weights, device)
                round_data[f"{env}_train_loss"] = train_loss

                updated_state = get_lora_state(model)
                delta = compute_lora_delta(global_state, updated_state)
                norm = sum(v.float().norm().item() ** 2 for v in delta.values()) ** 0.5
                deltas.append(delta)
                norms.append(norm)
                round_data[f"{env}_delta_norm"] = norm

            agg_delta = _weighted_average_state_dicts(
                deltas, [1.0] * len(deltas)
            )

            norm_str = "  ".join(f"{e}={round_data[f'{e}_delta_norm']:.4f}" for e in ENV_NAMES)
            pk_str = "  ".join(f"{e}={round_data[f'{e}_p_k_est']:.3f}" for e in ENV_NAMES)
            print(f"  Round {rnd}: norms=[{norm_str}]")
            print(f"          p_k_eff=[{pk_str}]")

            global_state = apply_delta_to_state(global_state, agg_delta)
            apply_lora_state(model, global_state, device)

            for env in ENV_NAMES:
                eval_loss = evaluate(model, eval_loaders[env], device)
                round_data[f"{env}_eval_loss"] = eval_loss

            avg, std = summarize_round(round_data)
            round_data["time_sec"] = time.time() - t0
            all_metrics.append(round_data)

            env_str = "  ".join(f"{e}={round_data[f'{e}_eval_loss']:.3f}" for e in ENV_NAMES)
            print(f"           eval: {env_str}  avg={avg:.4f}  std={std:.4f}")

        del model
        torch.cuda.empty_cache()
        results_by_tau[tau] = all_metrics

    return results_by_tau


def run_fedavg_baseline(device, train_tensors, eval_loaders):
    """FedAvg baseline with hard filtering (same as E3 alpha=0 online)."""
    print(f"\n{'='*60}")
    print(f"  BASELINE: FedAvg (hard filter)  |  Rounds: {NUM_ROUNDS}")
    print(f"{'='*60}")

    model, _ = load_model_and_tokenizer(device)
    global_state = get_lora_state(model)
    all_metrics = []

    for rnd in range(NUM_ROUNDS):
        t0 = time.time()
        round_data = {"round": rnd, "method": "fedavg_baseline"}

        apply_lora_state(model, global_state, device)

        all_losses = []
        env_losses_raw = {}
        for env in ENV_NAMES:
            losses = compute_per_trajectory_loss(model, train_tensors[env], device)
            env_losses_raw[env] = losses
            all_losses.extend(losses)

        threshold = sorted(all_losses)[len(all_losses) // 2]
        round_data["threshold"] = threshold

        deltas = []
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

        agg_delta = _weighted_average_state_dicts(deltas, [1.0] * len(deltas))

        global_state = apply_delta_to_state(global_state, agg_delta)
        apply_lora_state(model, global_state, device)

        for env in ENV_NAMES:
            eval_loss = evaluate(model, eval_loaders[env], device)
            round_data[f"{env}_eval_loss"] = eval_loss

        avg, std = summarize_round(round_data)
        round_data["time_sec"] = time.time() - t0
        all_metrics.append(round_data)

        env_str = "  ".join(f"{e}={round_data[f'{e}_eval_loss']:.3f}" for e in ENV_NAMES)
        print(f"  Round {rnd}: eval: {env_str}  avg={avg:.4f}  std={std:.4f}")

    del model
    torch.cuda.empty_cache()
    return all_metrics


def main():
    global NUM_ROUNDS

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--method", choices=["all", "option_b", "option_c", "baseline"], default="all")
    parser.add_argument("--taus", type=str, default="0.1,0.5,1.0,2.0",
                        help="Comma-separated tau values for Option C")
    args = parser.parse_args()

    NUM_ROUNDS = args.rounds
    tau_values = [float(t) for t in args.taus.split(",")]
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    print("=" * 60)
    print("  E4: Option B vs Option C vs FedAvg Baseline")
    print(f"  Model: {MODEL_NAME}  |  Device: {device}  |  Rounds: {NUM_ROUNDS}")
    print(f"  Option B: per-client alpha_k = 1 - hat_p_k")
    print(f"  Option C: soft filtering tau in {tau_values}")
    print("=" * 60)

    print("\n  Loading tokenizer...")
    tok_model, tokenizer = load_model_and_tokenizer(device)
    del tok_model
    torch.cuda.empty_cache()

    print("  Pre-tokenizing data...")
    train_tensors, eval_loaders = prepare_data(tokenizer)
    del tokenizer

    all_results = {}

    if args.method in ["all", "baseline"]:
        all_results["fedavg_baseline"] = run_fedavg_baseline(device, train_tensors, eval_loaders)

    if args.method in ["all", "option_b"]:
        all_results["option_b"] = run_option_b(device, train_tensors, eval_loaders)

    if args.method in ["all", "option_c"]:
        c_results = run_option_c(device, train_tensors, eval_loaders, tau_values)
        for tau, metrics in c_results.items():
            all_results[f"option_c_tau{tau:.3f}"] = metrics

    for key, metrics in all_results.items():
        save_dir = Path(f"outputs/e4_comparison/{key}")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"\n{'='*90}")
    print("  E4 FINAL COMPARISON")
    print(f"{'='*90}")
    print(f"{'Method':>25} | {'Avg Loss':>10} | {'Std':>8} |", end="")
    for env in ENV_NAMES:
        print(f" {env:>10}", end="")
    print()
    print("-" * 90)

    for key in all_results:
        d = all_results[key][-1]
        label = key.replace("option_c_tau", "C(tau=").replace("option_b", "B(adapt)").replace("fedavg_baseline", "FedAvg")
        if key.startswith("option_c"):
            label += ")"
        print(f"{label:>25} | {d['avg_eval_loss']:>10.4f} | {d['eval_loss_std']:>8.4f} |", end="")
        for env in ENV_NAMES:
            print(f" {d[f'{env}_eval_loss']:>10.4f}", end="")
        print()

    best_key = min(all_results, key=lambda k: all_results[k][-1]["avg_eval_loss"])
    best = all_results[best_key][-1]
    print(f"\n  WINNER: {best_key} with avg={best['avg_eval_loss']:.4f} std={best['eval_loss_std']:.4f}")

    best_bal = min(all_results, key=lambda k: all_results[k][-1]["eval_loss_std"])
    bal = all_results[best_bal][-1]
    print(f"  BEST BALANCE: {best_bal} with std={bal['eval_loss_std']:.4f}")


if __name__ == "__main__":
    main()
