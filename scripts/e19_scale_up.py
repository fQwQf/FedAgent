"""
E19: Scale-Up Analysis - Does Loss-Proportional Remain Optimal?

Tests whether loss-proportional aggregation remains optimal when scaling:
  (a) LoRA rank: 8 → 16
  (b) Training samples: 64 → 128
  (c) Combined: rank=16, samples=128

If complex methods (layer-adaptive, curriculum, etc.) start working at scale,
then loss-proportional's dominance is a "small-scale phenomenon" — itself novel.
If loss-proportional remains optimal across scales, it's robustly optimal.

Usage:
    python scripts/e19_scale_up.py --model_size 0.5B --device cuda:3 --lora_rank 16 --train_samples 128
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
NUM_ROUNDS = 20
SEED = 42
EVAL_SAMPLES = 32
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4
LR = 5e-5
LORA_ALPHA_SCALE = 2  # alpha = rank * 2

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

def load_model_and_tokenizer(device, model_name, lora_rank):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora_alpha = lora_rank * LORA_ALPHA_SCALE
    model = get_peft_model(model, LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM", bias="none"))
    return model, tokenizer

def get_lora_state(model):
    return OrderedDict({k: v.clone().cpu() for k, v in model.state_dict().items() if "lora_" in k})

def apply_lora_state(model, state, device):
    model.load_state_dict(OrderedDict({k: v.to(device) for k, v in state.items()}), strict=False)

def make_data(tokenizer, train_samples):
    train_dl = {}
    eval_dl = {}
    for env in ALL_ENVS:
        data = load_agentgym_data(env, max_samples=train_samples + EVAL_SAMPLES, seed=SEED)
        for split, start, end, store in [
            ("train", 0, train_samples, train_dl),
            ("eval", train_samples, train_samples + EVAL_SAMPLES, eval_dl)
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

def run_method(device, train_dl, eval_dl, method, num_rounds, model_name, lora_rank):
    use_loss_wt = (method == "loss_wt")
    print(f"\n{'='*70}")
    print(f"  {method}  ({'loss-proportional' if use_loss_wt else 'uniform'}, rank={lora_rank}, {num_rounds}R)")
    print(f"{'='*70}")
    model, _ = load_model_and_tokenizer(device, model_name, lora_rank)
    global_state = get_lora_state(model)
    metrics = []
    for rnd in range(num_rounds):
        t0 = time.time()
        rd = {"round": rnd, "method": method}
        apply_lora_state(model, global_state, device)
        eval_losses = {}
        for env in ALL_ENVS:
            eval_losses[env] = evaluate(model, eval_dl[env], device)
            rd[f"{env}_eval_loss"] = eval_losses[env]
        if use_loss_wt:
            total = sum(eval_losses[e] for e in ALL_ENVS)
            weights = {e: eval_losses[e] / total for e in ALL_ENVS}
        else:
            weights = {e: 1.0 / len(ALL_ENVS) for e in ALL_ENVS}
        deltas = {}
        for env in ALL_ENVS:
            apply_lora_state(model, global_state, device)
            train_client(model, train_dl[env], device)
            new_state = get_lora_state(model)
            deltas[env] = OrderedDict({k: new_state[k].float() - global_state[k].float() for k in global_state})
        weight_list = [weights[e] for e in ALL_ENVS]
        delta_list = [deltas[e] for e in ALL_ENVS]
        avg_delta = _weighted_average_state_dicts(delta_list, weight_list)
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
        wt_str = " ".join(f"w_{e[:3]}={weights[e]:.2f}" for e in ALL_ENVS) if use_loss_wt else ""
        print(f"  R{rnd:>2}: {loss_str}  hard={hard_avg:.3f}  |  {wt_str}")
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
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=64)
    parser.add_argument("--method", default="all", choices=["all", "uniform", "loss_wt"])
    args = parser.parse_args()
    device = args.device
    model_name = MODEL_MAP[args.model_size]
    set_seed(SEED)
    print("=" * 70)
    print(f"  E19: Scale-Up Analysis")
    print(f"  Model: {model_name}  Rank: {args.lora_rank}  Samples: {args.train_samples}")
    print(f"  Rounds: {args.rounds}  Device: {device}")
    print("=" * 70)
    _, tokenizer = load_model_and_tokenizer(device, model_name, args.lora_rank)
    torch.cuda.empty_cache()
    print("  Tokenizing data...")
    train_dl, eval_dl = make_data(tokenizer, args.train_samples)
    del tokenizer
    torch.cuda.empty_cache()
    methods = (args.method.split(",") if args.method != "all" else ["uniform", "loss_wt"])
    all_results = {}
    for m in methods:
        all_results[m] = run_method(device, train_dl, eval_dl, m, args.rounds, model_name, args.lora_rank)
        save_dir = Path(f"outputs/e19_scale_up/{args.model_size}/R{args.lora_rank}_S{args.train_samples}/{m}")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(all_results[m], f, indent=2)
    print(f"\n{'='*70}")
    print(f"  E19 RESULTS (rank={args.lora_rank}, samples={args.train_samples})")
    print(f"{'='*70}")
    for m in methods:
        final = all_results[m][-1]
        print(f"\n  {m}:")
        print(f"    hard_avg = {final['hard_avg']:.4f}")
        for e in ALL_ENVS:
            print(f"    {e:10s} = {final[f'{e}_eval_after']:.4f}")
    if len(methods) == 2:
        lw_hard = all_results["loss_wt"][-1]["hard_avg"]
        uf_hard = all_results["uniform"][-1]["hard_avg"]
        delta = lw_hard - uf_hard
        pct = delta / uf_hard * 100 if uf_hard != 0 else 0
        print(f"\n  Loss-Wt vs Uniform: {delta:+.4f} ({pct:+.1f}%) on hard_avg")
    print(f"\n  Results saved to outputs/e19_scale_up/{args.model_size}/R{args.lora_rank}_S{args.train_samples}/")

if __name__ == "__main__":
    main()
