"""
E8: Loss-Weighted Aggregation for LLM Agent FL

Tests: Can we fix the gradient dilution problem revealed by E7?

Methods:
  1. fed_5env_uniform   -- Standard FedAvg (equal weights) [E7 baseline]
  2. fed_5env_loss_wt   -- Loss-proportional weighting: w_k = L_k / sum(L_j)
  3. fed_3hard_uniform  -- Only hard envs, standard FedAvg [E7 baseline]

Prediction: fed_5env_loss_wt should match or beat fed_3hard_uniform on hard envs,
while also training easy envs (unlike fed_3hard which ignores them).

Usage:
    python scripts/e8_loss_weighted.py --device cuda:4
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
from src.aggregation import _weighted_average_state_dicts

ALL_ENVS = ["babyai", "webshop", "textcraft", "maze", "wordle"]
HARD_ENVS = ["webshop", "textcraft", "wordle"]
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_RANK = 8
LORA_ALPHA = 16
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4
LR = 5e-5
NUM_ROUNDS = 5
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


def prepare_data(tokenizer):
    train_tensors = {}
    eval_loaders = {}
    for env in ALL_ENVS:
        data = load_agentgym_data(env, max_samples=TRAIN_SAMPLES + EVAL_SAMPLES, seed=SEED)
        train_data = data[:TRAIN_SAMPLES]
        eval_data = data[TRAIN_SAMPLES:TRAIN_SAMPLES + EVAL_SAMPLES]

        train_pts = [pt for traj in train_data
                     if (pt := tokenize_trajectory(tokenizer, traj)) is not None]
        train_tensors[env] = train_pts

        eval_pts = [pt for traj in eval_data
                    if (pt := tokenize_trajectory(tokenizer, traj)) is not None]
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


def run_fed_method(device, train_tensors, eval_loaders, method, num_rounds):
    use_loss_weight = (method == "fed_5env_loss_wt")
    env_list = ALL_ENVS

    print(f"\n{'='*60}")
    print(f"  {method}: {'loss-proportional' if use_loss_weight else 'uniform'} weighting")
    print(f"{'='*60}")

    model, _ = load_model_and_tokenizer(device)
    global_state = get_lora_state(model)
    all_metrics = []

    for rnd in range(num_rounds):
        t0 = time.time()
        round_data = {"round": rnd, "method": method}
        apply_lora_state(model, global_state, device)

        # Evaluate first to get per-env losses for weighting
        eval_losses = {}
        for env in env_list:
            eval_losses[env] = evaluate(model, eval_loaders[env], device)

        # Compute aggregation weights
        if use_loss_weight:
            total_loss = sum(eval_losses.values())
            weights = {env: eval_losses[env] / total_loss for env in env_list}
        else:
            weights = {env: 1.0 / len(env_list) for env in env_list}

        # Log weights
        for env in env_list:
            round_data[f"{env}_eval_loss_before"] = eval_losses[env]
            round_data[f"{env}_agg_weight"] = weights[env]

        # Client training
        deltas = []
        for env in env_list:
            apply_lora_state(model, global_state, device)
            dl = make_dataloader(train_tensors[env])
            if dl is not None:
                train_one_epoch(model, dl, device)
            updated_state = get_lora_state(model)
            deltas.append(compute_lora_delta(global_state, updated_state))

        # Aggregate with weights
        weight_list = [weights[env] for env in env_list]
        global_state = OrderedDict({
            k: global_state[k] + delta_k
            for k, delta_k in _weighted_average_state_dicts(deltas, weight_list).items()
        })
        apply_lora_state(model, global_state, device)

        # Evaluate after aggregation
        for env in ALL_ENVS:
            round_data[f"{env}_eval_loss"] = evaluate(model, eval_loaders[env], device)

        easy_avg = sum(round_data[f"{e}_eval_loss"] for e in ["babyai", "maze"]) / 2
        hard_avg = sum(round_data[f"{e}_eval_loss"] for e in HARD_ENVS) / 3
        all_avg = sum(round_data[f"{e}_eval_loss"] for e in ALL_ENVS) / 5
        round_data["easy_avg"] = easy_avg
        round_data["hard_avg"] = hard_avg
        round_data["avg_eval_loss"] = all_avg
        round_data["time_sec"] = time.time() - t0
        all_metrics.append(round_data)

        wt_str = "  ".join(f"w_{e[:3]}={weights[e]:.3f}" for e in env_list)
        env_str = "  ".join(f"{e}={round_data[f'{e}_eval_loss']:.3f}" for e in ALL_ENVS)
        print(f"  R{rnd}: {env_str}  hard={hard_avg:.4f}  |  {wt_str}")

    del model
    torch.cuda.empty_cache()
    return all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--method", choices=["all", "fed_5env_uniform", "fed_5env_loss_wt"],
                        default="all")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    print("=" * 60)
    print(f"  E8: Loss-Weighted Aggregation")
    print(f"  Device: {device}  |  Rounds: {args.rounds}")
    print("=" * 60)

    _, tokenizer = load_model_and_tokenizer(device)
    torch.cuda.empty_cache()

    print("  Pre-tokenizing data...")
    train_tensors, eval_loaders = prepare_data(tokenizer)

    all_results = {}
    methods = (args.method.split(",") if args.method != "all"
               else ["fed_5env_uniform", "fed_5env_loss_wt"])

    for method in methods:
        all_results[method] = run_fed_method(
            device, train_tensors, eval_loaders, method, args.rounds)

        save_dir = Path(f"outputs/e8_loss_weighted/{method}")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(all_results[method], f, indent=2)

    # Final comparison
    print(f"\n{'='*100}")
    print("  E8 FINAL COMPARISON")
    print(f"{'='*100}")

    # Load E7 baselines for comparison
    e7_dir = Path("outputs/e7_cascading")
    e7_5env = json.load(open(e7_dir / "fed_5env_nat" / "metrics.json")) if (e7_dir / "fed_5env_nat" / "metrics.json").exists() else None
    e7_3hard = json.load(open(e7_dir / "fed_3hard_nat" / "metrics.json")) if (e7_dir / "fed_3hard_nat" / "metrics.json").exists() else None

    print(f"{'Method':>22} | {'Avg':>8} | {'Hard':>8} |", end="")
    for e in ALL_ENVS:
        print(f" {e[:5]:>7}", end="")
    print()
    print("-" * 100)

    comparisons = []
    if e7_5env:
        comparisons.append(("E7_fed_5env_uniform", e7_5env[-1]))
    if e7_3hard:
        comparisons.append(("E7_fed_3hard_uniform", e7_3hard[-1]))
    for method in methods:
        comparisons.append((method, all_results[method][-1]))

    for name, d in comparisons:
        hard_avg = sum(d[f"{e}_eval_loss"] for e in HARD_ENVS) / len(HARD_ENVS)
        all_avg = sum(d[f"{e}_eval_loss"] for e in ALL_ENVS) / len(ALL_ENVS)
        print(f"{name:>22} | {all_avg:>8.4f} | {hard_avg:>8.4f} |", end="")
        for e in ALL_ENVS:
            print(f" {d[f'{e}_eval_loss']:>7.4f}", end="")
        print()

    # Key comparison
    print(f"\n{'='*100}")
    print("  KEY: Does loss-weighting fix gradient dilution?")
    print(f"{'='*100}")
    if "fed_5env_uniform" in all_results and "fed_5env_loss_wt" in all_results:
        d_u = all_results["fed_5env_uniform"][-1]
        d_w = all_results["fed_5env_loss_wt"][-1]
        if e7_3hard:
            d_3h = e7_3hard[-1]
        for e in HARD_ENVS:
            lu = d_u[f"{e}_eval_loss"]
            lw = d_w[f"{e}_eval_loss"]
            delta = (lu - lw) / lu * 100
            print(f"  {e:>10}: uniform={lu:.4f}  loss_wt={lw:.4f}  improvement={delta:+.1f}%")
        hard_u = sum(d_u[f"{e}_eval_loss"] for e in HARD_ENVS) / 3
        hard_w = sum(d_w[f"{e}_eval_loss"] for e in HARD_ENVS) / 3
        hard_3h = sum(d_3h[f"{e}_eval_loss"] for e in HARD_ENVS) / 3 if e7_3hard else 0
        print(f"\n  Hard AVG: uniform={hard_u:.4f}  loss_wt={hard_w:.4f}  3hard={hard_3h:.4f}")
        print(f"  Loss-wt vs uniform: {(hard_u-hard_w)/hard_u*100:+.1f}%")
        if e7_3hard:
            print(f"  Loss-wt vs 3hard:   {(hard_3h-hard_w)/hard_3h*100:+.1f}% (target: match or beat 3hard)")


if __name__ == "__main__":
    main()
