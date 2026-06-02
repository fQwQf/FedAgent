"""
E16: FedAgentProto - Prototypical Trajectory Alignment for Multi-Task FL

Direction C: Instead of manipulating parameter aggregation, align trajectory
representations in a shared embedding space. Each task has a prototype
(mean representation of its trajectories). During training, pull trajectory
representations toward their task prototype while pushing prototypes of
dissimilar tasks apart.

Core idea: bypass gradient conflict by defining objectives in representation
space rather than parameter space.

Loss: L = L_SFT + λ_align * L_align
  where L_align = ||h(x) - p_k||^2 / ||p_k||^2
  (pull trajectory embedding toward task prototype, normalized)

Usage:
    python scripts/e16_fedagent_proto.py --model_size 0.5B --device cuda:3 --lambda_align 0.5
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
        device_map={"": device}, trust_remote_code=True,
        output_hidden_states=True)
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


def extract_prototypes(model, dataloader, device, max_batches=8):
    """Extract mean hidden state as task prototype."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            ids, mask, labels = [b.to(device) for b in batch]
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=ids, attention_mask=mask, labels=labels,
                                output_hidden_states=True)
                # Use last hidden state, mean over sequence
                last_hidden = outputs.hidden_states[-1]  # [batch, seq, hidden]
                # Mask out padding tokens
                mask_expanded = mask.unsqueeze(-1).expand(last_hidden.size()).float()
                masked_hidden = last_hidden * mask_expanded
                sum_hidden = masked_hidden.sum(dim=1)  # [batch, hidden]
                mean_hidden = sum_hidden / mask_expanded.sum(dim=1).clamp(min=1)
                embeddings.append(mean_hidden.float().cpu())
    
    if embeddings:
        all_emb = torch.cat(embeddings, dim=0)
        prototype = all_emb.mean(dim=0)  # [hidden]
        return prototype
    return None


def train_client_with_proto(model, dataloader, prototype, device, lambda_align):
    """Train with SFT loss + prototype alignment loss."""
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    
    total_sft_loss = 0.0
    total_align_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        ids, mask, labels = [b.to(device) for b in batch]
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=ids, attention_mask=mask, labels=labels,
                           output_hidden_states=True)
            sft_loss = outputs.loss
            
            # Prototype alignment loss
            if lambda_align > 0 and prototype is not None:
                last_hidden = outputs.hidden_states[-1]
                mask_expanded = mask.unsqueeze(-1).expand(last_hidden.size()).float()
                masked_hidden = last_hidden * mask_expanded
                sum_hidden = masked_hidden.sum(dim=1)
                mean_hidden = sum_hidden / mask_expanded.sum(dim=1).clamp(min=1)
                
                proto_device = prototype.to(device)
                # Normalize
                mean_hidden_norm = mean_hidden / (mean_hidden.norm(dim=-1, keepdim=True) + 1e-8)
                proto_norm = proto_device / (proto_device.norm() + 1e-8)
                
                # Cosine distance to prototype
                align_loss = (1 - (mean_hidden_norm * proto_norm).sum(dim=-1)).mean()
            else:
                align_loss = 0.0
            
            total_loss = sft_loss + lambda_align * align_loss
        
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        
        total_sft_loss += sft_loss.item()
        if isinstance(align_loss, torch.Tensor):
            total_align_loss += align_loss.item()
        n_batches += 1
    
    return total_sft_loss / max(n_batches, 1), total_align_loss / max(n_batches, 1)


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


def run_method(device, train_dl, eval_dl, method, num_rounds, model_name, lambda_align=0.0):
    use_proto = (method == "fedagent_proto")

    print(f"\n{'='*70}")
    print(f"  {method}  (λ_align={lambda_align}, {num_rounds}R, {model_name})")
    print(f"{'='*70}")

    model, _ = load_model_and_tokenizer(device, model_name)
    global_state = get_lora_state(model)
    metrics = []
    
    # Compute initial prototypes (from global model before any training)
    prototypes = {}
    if use_proto:
        print("  Computing initial prototypes...")
        apply_lora_state(model, global_state, device)
        for env in ALL_ENVS:
            proto = extract_prototypes(model, eval_dl[env], device, max_batches=8)
            if proto is not None:
                prototypes[env] = proto
                print(f"    {env}: proto_norm={proto.norm().item():.3f}")

    for rnd in range(num_rounds):
        t0 = time.time()
        rd = {"round": rnd, "method": method, "lambda_align": lambda_align}

        # Eval
        apply_lora_state(model, global_state, device)
        eval_losses = {}
        for env in ALL_ENVS:
            eval_losses[env] = evaluate(model, eval_dl[env], device)
            rd[f"{env}_eval_loss"] = eval_losses[env]

        # Update prototypes every 3 rounds (optional)
        if use_proto and rnd > 0 and rnd % 3 == 0:
            for env in ALL_ENVS:
                proto = extract_prototypes(model, eval_dl[env], device, max_batches=4)
                if proto is not None:
                    prototypes[env] = proto

        # Client updates
        deltas = {}
        align_losses = {}
        for env in ALL_ENVS:
            apply_lora_state(model, global_state, device)
            if use_proto:
                sft_l, align_l = train_client_with_proto(
                    model, train_dl[env], prototypes.get(env), device, lambda_align)
                align_losses[env] = align_l
            else:
                model.train()
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
                for batch in train_dl[env]:
                    ids, mask, labels = [b.to(device) for b in batch]
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            
            new_state = get_lora_state(model)
            deltas[env] = OrderedDict({
                k: new_state[k].float() - global_state[k].float() for k in global_state})

        # Aggregate (uniform for now - the novelty is in the local training)
        delta_list = [deltas[env] for env in ALL_ENVS]
        avg_delta = OrderedDict()
        for key in global_state.keys():
            tensors = [d[key].float() for d in delta_list]
            avg_delta[key] = torch.stack(tensors).mean(dim=0)

        # Update global state
        global_state = OrderedDict({k: global_state[k] + avg_delta[k] for k in global_state})
        apply_lora_state(model, global_state, device)

        # Final eval
        for env in ALL_ENVS:
            rd[f"{env}_eval_after"] = evaluate(model, eval_dl[env], device)

        hard_avg = sum(rd[f"{e}_eval_after"] for e in HARD_ENVS) / len(HARD_ENVS)
        all_avg = sum(rd[f"{e}_eval_after"] for e in ALL_ENVS) / len(ALL_ENVS)
        rd["hard_avg"] = hard_avg
        rd["avg_eval_loss"] = all_avg
        rd["time_sec"] = time.time() - t0
        
        if use_proto:
            rd["align_losses"] = {k: float(v) for k, v in align_losses.items()}

        metrics.append(rd)

        loss_str = " ".join(f"{e[:3]}={rd[f'{e}_eval_after']:.3f}" for e in ALL_ENVS)
        extra = f"align={np.mean(list(align_losses.values())):.3f}" if use_proto else ""
        print(f"  R{rnd:>2}: {loss_str}  hard={hard_avg:.3f}  {extra}")

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
    parser.add_argument("--lambda_align", type=float, default=0.5)
    parser.add_argument("--method", default="all", choices=["all", "baseline", "fedagent_proto"])
    args = parser.parse_args()

    device = args.device
    model_name = MODEL_MAP[args.model_size]
    set_seed(SEED)

    print("=" * 70)
    print(f"  E16: FedAgentProto - Trajectory Prototype Alignment")
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
               else ["baseline", "fedagent_proto"])

    all_results = {}
    for m in methods:
        la = args.lambda_align if m == "fedagent_proto" else 0.0
        all_results[m] = run_method(device, train_dl, eval_dl, m, args.rounds, model_name, la)
        save_dir = Path(f"outputs/e16_fedagent_proto/{args.model_size}/{m}")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(all_results[m], f, indent=2)

    # ================================================================
    # ANALYSIS
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  E16 RESULTS COMPARISON ({args.model_size}, {args.rounds}R)")
    print(f"{'='*70}")

    for m in methods:
        final = all_results[m][-1]
        print(f"\n  {m}:")
        print(f"    hard_avg = {final['hard_avg']:.4f}")
        print(f"    avg_loss = {final['avg_eval_loss']:.4f}")
        for e in ALL_ENVS:
            print(f"    {e:10s} = {final[f'{e}_eval_after']:.4f}")

    if "fedagent_proto" in all_results and "baseline" in all_results:
        fp_hard = all_results["fedagent_proto"][-1]["hard_avg"]
        base_hard = all_results["baseline"][-1]["hard_avg"]
        delta = fp_hard - base_hard
        pct = delta / base_hard * 100 if base_hard != 0 else 0
        print(f"\n  FedAgentProto vs Baseline: {delta:+.4f} ({pct:+.1f}%) on hard_avg")

    print(f"\n  Results saved to outputs/e16_fedagent_proto/{args.model_size}/")


if __name__ == "__main__":
    main()
