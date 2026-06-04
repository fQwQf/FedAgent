"""
E17: Sequential Compositional LoRA (SeqComp-LoRA)

Core paradigm shift: Instead of aggregating gradients from multiple tasks,
train tasks sequentially. Each task gets its own LoRA module. After training,
merge the LoRA into the base model. The next task trains on the merged model
with a fresh LoRA.

At inference: Model = Base + LoRA_1 + LoRA_2 + ... + LoRA_K

Key insight: Zero gradient conflict (tasks never compete), zero catastrophic
forgetting (previous LoRAs are merged and frozen).

Task order (easy → hard): babyai → maze → wordle → textcraft → webshop

Usage:
    python scripts/e17_seqcomp_lora.py --model_size 0.5B --device cuda:3
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

# Task order: easy → hard
TASK_ORDER = ["babyai", "maze", "wordle", "textcraft", "webshop"]
ROUNDS_PER_TASK = 4
NUM_ROUNDS = len(TASK_ORDER) * ROUNDS_PER_TASK  # 20 total

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


def load_base_model(device, model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model, tokenizer


def add_lora_to_model(model):
    from peft import LoraConfig, get_peft_model
    return get_peft_model(model, LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM", bias="none"))


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


def train_task(model, dataloader, device, num_rounds, task_name, global_round_offset):
    """Train model on a single task for num_rounds."""
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    
    metrics = []
    for rnd in range(num_rounds):
        t0 = time.time()
        rd = {
            "round": global_round_offset + rnd,
            "task_round": rnd,
            "task": task_name,
            "method": "seqcomp"
        }
        
        # Train one epoch
        for batch in dataloader:
            ids, mask, labels = [b.to(device) for b in batch]
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        rd["time_sec"] = time.time() - t0
        metrics.append(rd)
    
    return metrics


def evaluate_model(model, eval_dl, device):
    """Evaluate model on all tasks. Returns dict of losses."""
    model.eval()
    results = {}
    with torch.no_grad():
        for env in ALL_ENVS:
            total, n = 0.0, 0
            for batch in eval_dl[env]:
                ids, mask, labels = [b.to(device) for b in batch]
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    total += model(input_ids=ids, attention_mask=mask, labels=labels).loss.item() * ids.size(0)
                n += ids.size(0)
            results[env] = total / max(n, 1)
    return results


def run_seqcomp(device, train_dl, eval_dl, model_name):
    print(f"\n{'='*70}")
    print(f"  SeqComp-LoRA  (Sequential Compositional)")
    print(f"  Order: {' → '.join(TASK_ORDER)}")
    print(f"  {ROUNDS_PER_TASK} rounds per task, {NUM_ROUNDS} total")
    print(f"  Model: {model_name}")
    print(f"{'='*70}")
    
    # Load base model
    cumulative_model, _ = load_base_model(device, model_name)
    torch.cuda.empty_cache()
    
    all_metrics = []
    global_round = 0
    
    for task_idx, task_name in enumerate(TASK_ORDER):
        print(f"\n{'='*70}")
        print(f"  Task {task_idx+1}/{len(TASK_ORDER)}: {task_name}")
        print(f"{'='*70}")
        
        # Add fresh LoRA to cumulative model
        model = add_lora_to_model(cumulative_model)
        model.print_trainable_parameters()
        
        # Train
        task_metrics = train_task(
            model, train_dl[task_name], device, 
            ROUNDS_PER_TASK, task_name, global_round
        )
        
        # Evaluate after each round
        for i, rd in enumerate(task_metrics):
            eval_losses = evaluate_model(model, eval_dl, device)
            for env, loss in eval_losses.items():
                rd[f"{env}_eval_after"] = loss
            
            hard_avg = sum(eval_losses[e] for e in HARD_ENVS) / len(HARD_ENVS)
            all_avg = sum(eval_losses.values()) / len(eval_losses)
            rd["hard_avg"] = hard_avg
            rd["avg_eval_loss"] = all_avg
            
            loss_str = " ".join(f"{e[:3]}={eval_losses[e]:.3f}" for e in ALL_ENVS)
            print(f"  R{rd['round']:>2} [{task_name}]: {loss_str}  hard={hard_avg:.3f}")
        
        all_metrics.extend(task_metrics)
        global_round += ROUNDS_PER_TASK
        
        # Merge LoRA into base model
        print(f"  Merging LoRA for {task_name} into base model...")
        cumulative_model = model.merge_and_unload()
        del model
        torch.cuda.empty_cache()
        
        # Evaluate merged model
        print(f"  Evaluating merged model (after {task_name}):")
        merged_eval = evaluate_model(cumulative_model, eval_dl, device)
        merged_hard = sum(merged_eval[e] for e in HARD_ENVS) / len(HARD_ENVS)
        loss_str = " ".join(f"{e[:3]}={merged_eval[e]:.3f}" for e in ALL_ENVS)
        print(f"    {loss_str}  hard={merged_hard:.3f}")
        
        # Save checkpoint
        save_dir = Path(f"outputs/e17_seqcomp/{model_name.split('/')[-1]}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(cumulative_model.state_dict(), save_dir / f"merged_after_{task_name}.pt")
    
    del cumulative_model
    torch.cuda.empty_cache()
    return all_metrics


def run_baseline(device, train_dl, eval_dl, model_name):
    """Baseline: standard FedAvg with all tasks simultaneously."""
    print(f"\n{'='*70}")
    print(f"  Baseline  (Simultaneous FedAvg, {NUM_ROUNDS}R)")
    print(f"  Model: {model_name}")
    print(f"{'='*70}")
    
    from peft import LoraConfig, get_peft_model
    
    model, _ = load_base_model(device, model_name)
    model = get_peft_model(model, LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM", bias="none"))
    
    # Get initial LoRA state
    global_state = OrderedDict({
        k: v.clone().cpu() for k, v in model.state_dict().items() if "lora_" in k
    })
    
    metrics = []
    
    for rnd in range(NUM_ROUNDS):
        t0 = time.time()
        rd = {"round": rnd, "method": "baseline"}
        
        # Eval
        apply_state(model, global_state, device)
        eval_losses = evaluate_model(model, eval_dl, device)
        for env, loss in eval_losses.items():
            rd[f"{env}_eval_loss"] = loss
        
        # Client updates
        deltas = {}
        for env in ALL_ENVS:
            apply_state(model, global_state, device)
            model.train()
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
            for batch in train_dl[env]:
                ids, mask, labels = [b.to(device) for b in batch]
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            new_state = OrderedDict({
                k: v.clone().cpu() for k, v in model.state_dict().items() if "lora_" in k
            })
            deltas[env] = OrderedDict({
                k: new_state[k].float() - global_state[k].float() for k in global_state
            })
        
        # Uniform aggregate
        avg_delta = OrderedDict()
        for key in global_state.keys():
            tensors = [deltas[env][key].float() for env in ALL_ENVS]
            avg_delta[key] = torch.stack(tensors).mean(dim=0)
        
        global_state = OrderedDict({k: global_state[k] + avg_delta[k] for k in global_state})
        apply_state(model, global_state, device)
        
        # Final eval
        eval_losses = evaluate_model(model, eval_dl, device)
        for env, loss in eval_losses.items():
            rd[f"{env}_eval_after"] = loss
        
        hard_avg = sum(eval_losses[e] for e in HARD_ENVS) / len(HARD_ENVS)
        all_avg = sum(eval_losses.values()) / len(eval_losses)
        rd["hard_avg"] = hard_avg
        rd["avg_eval_loss"] = all_avg
        rd["time_sec"] = time.time() - t0
        
        metrics.append(rd)
        
        loss_str = " ".join(f"{e[:3]}={eval_losses[e]:.3f}" for e in ALL_ENVS)
        print(f"  R{rnd:>2}: {loss_str}  hard={hard_avg:.3f}")
        
        if rnd % 5 == 4:
            torch.cuda.empty_cache()
    
    del model
    torch.cuda.empty_cache()
    return metrics


def apply_state(model, state, device):
    model.load_state_dict(
        OrderedDict({k: v.to(device) for k, v in state.items()}), strict=False
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--model_size", default="0.5B", choices=["0.5B", "1.5B"])
    parser.add_argument("--method", default="all", choices=["all", "seqcomp", "baseline"])
    args = parser.parse_args()

    device = args.device
    model_name = MODEL_MAP[args.model_size]
    set_seed(SEED)

    print("=" * 70)
    print(f"  E17: Sequential Compositional LoRA (SeqComp-LoRA)")
    print(f"  Model: {model_name}  Total Rounds: {NUM_ROUNDS}  Device: {device}")
    print("=" * 70)

    # Load tokenizer and prepare data
    _, tokenizer = load_base_model(device, model_name)
    torch.cuda.empty_cache()
    print("  Tokenizing data...")
    train_dl, eval_dl = make_data(tokenizer)
    del tokenizer
    torch.cuda.empty_cache()

    methods = (args.method.split(",") if args.method != "all"
               else ["seqcomp", "baseline"])

    all_results = {}
    for m in methods:
        if m == "seqcomp":
            all_results[m] = run_seqcomp(device, train_dl, eval_dl, model_name)
        else:
            all_results[m] = run_baseline(device, train_dl, eval_dl, model_name)
        
        save_dir = Path(f"outputs/e17_seqcomp/{args.model_size}/{m}")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(all_results[m], f, indent=2)

    # ================================================================
    # ANALYSIS
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  E17 FINAL COMPARISON ({args.model_size})")
    print(f"{'='*70}")

    for m in methods:
        final = all_results[m][-1]
        print(f"\n  {m}:")
        print(f"    hard_avg = {final['hard_avg']:.4f}")
        print(f"    avg_loss = {final['avg_eval_loss']:.4f}")
        for e in ALL_ENVS:
            print(f"    {e:10s} = {final[f'{e}_eval_after']:.4f}")

    if "seqcomp" in all_results and "baseline" in all_results:
        seq_hard = all_results["seqcomp"][-1]["hard_avg"]
        base_hard = all_results["baseline"][-1]["hard_avg"]
        delta = seq_hard - base_hard
        pct = delta / base_hard * 100 if base_hard != 0 else 0
        print(f"\n  SeqComp vs Baseline: {delta:+.4f} ({pct:+.1f}%) on hard_avg")
        
        # Show progression
        print(f"\n  SeqComp progression (after each task merge):")
        for i, task in enumerate(TASK_ORDER):
            idx = (i + 1) * ROUNDS_PER_TASK - 1
            if idx < len(all_results["seqcomp"]):
                rd = all_results["seqcomp"][idx]
                print(f"    After {task:10s}: hard={rd['hard_avg']:.3f}")

    print(f"\n  Results saved to outputs/e17_seqcomp/{args.model_size}/")


if __name__ == "__main__":
    main()
