"""
E9: Per-Layer Gradient Analysis for LLM Agent FL

Diagnoses WHERE gradient dilution happens by decomposing gradients per layer.

Analyses:
  1. Per-layer gradient norms by environment
  2. Per-layer cosine similarity between all environment pairs
  3. Shared vs task-specific gradient decomposition per layer
  4. FedAvg signal retention: how much of each env's gradient survives averaging
  5. Gradient norm ratio: easy vs hard environments

Run at two training stages: R0 (initial) and after 1 round (post-adaptation).

Usage:
    python scripts/e9_grad_analysis.py --device cuda:4
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict, defaultdict
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import load_agentgym_data, get_trainable_messages

ALL_ENVS = ["babyai", "webshop", "textcraft", "maze", "wordle"]
EASY_ENVS = ["babyai", "maze"]
HARD_ENVS = ["webshop", "textcraft", "wordle"]
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_RANK = 8
LORA_ALPHA = 16
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4
LR = 5e-5
SEED = 42


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


def tokenize_data(tokenizer):
    tensors = {}
    for env in ALL_ENVS:
        data = load_agentgym_data(env, max_samples=64, seed=SEED)
        pts = []
        for traj in data:
            messages, _ = get_trainable_messages(traj)
            if not messages:
                continue
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            if not text.strip():
                continue
            tok = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH,
                            padding="max_length", return_tensors="pt")
            ids, masks = tok["input_ids"], tok["attention_mask"]
            labels = ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            pts.append((ids[0], masks[0], labels[0]))
        ids = torch.stack([p[0] for p in pts])
        masks = torch.stack([p[1] for p in pts])
        labels = torch.stack([p[2] for p in pts])
        tensors[env] = DataLoader(TensorDataset(ids, masks, labels),
                                   batch_size=BATCH_SIZE, shuffle=True)
        print(f"  {env}: {len(pts)} samples")
    return tensors


def extract_per_layer_grads(model, dataloader, device):
    """Extract gradient for each LoRA layer after one forward+backward pass."""
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    optimizer.zero_grad()

    batch = next(iter(dataloader))
    input_ids, attention_mask, labels = [b.to(device) for b in batch]
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()

    layer_grads = OrderedDict()
    for name, param in model.named_parameters():
        if "lora_" in name and param.grad is not None:
            layer_grads[name] = param.grad.detach().float().clone()

    optimizer.zero_grad()
    return layer_grads, outputs.loss.item()


def get_layer_group(name):
    """Classify a LoRA parameter name into layer group."""
    parts = name.split(".")
    layer_idx = None
    for p in parts:
        if p.isdigit():
            layer_idx = int(p)
            break

    if "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name:
        proj = name.split(".")[-2]
        return f"attn_{proj}", layer_idx
    elif "gate_proj" in name:
        return "ffn_gate", layer_idx
    elif "up_proj" in name:
        return "ffn_up", layer_idx
    elif "down_proj" in name:
        return "ffn_down", layer_idx
    return "other", layer_idx


def cosine_sim(v1, v2):
    return torch.dot(v1.flatten(), v2.flatten()) / (v1.norm() * v2.norm() + 1e-8)


def train_one_round(model, dataloaders, device):
    """Simulate one FedAvg round and return updated state."""
    from src.aggregation import _weighted_average_state_dicts

    initial_state = OrderedDict({
        k: v.clone().cpu() for k, v in model.state_dict().items() if "lora_" in k
    })

    deltas = []
    for env in ALL_ENVS:
        for k, v in model.state_dict().items():
            if "lora_" in k:
                v.copy_(initial_state[k].to(device))

        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=LR
        )
        for batch in dataloaders[env]:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                labels=labels)
            outputs.loss.backward()
            optimizer.step()

        after_state = OrderedDict({
            k: v.clone().cpu() for k, v in model.state_dict().items() if "lora_" in k
        })
        deltas.append(OrderedDict({
            k: after_state[k].float() - initial_state[k].float() for k in initial_state
        }))

    avg_delta = _weighted_average_state_dicts(deltas, [1.0] * len(deltas))
    new_state = OrderedDict({
        k: initial_state[k] + avg_delta[k] for k in initial_state
    })

    model.load_state_dict(
        OrderedDict({k: v.to(device) for k, v in new_state.items()}), strict=False
    )
    return new_state


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:4")
    args = parser.parse_args()
    device = args.device

    print("=" * 70)
    print("  E9: Per-Layer Gradient Analysis")
    print("=" * 70)

    print("\n  Loading model and data...")
    model, tokenizer = load_model_and_tokenizer(device)
    dataloaders = tokenize_data(tokenizer)

    results = {"stages": []}

    for stage_name, stage_idx in [("R0_initial", 0), ("R1_after_1round", 1)]:
        print(f"\n{'='*70}")
        print(f"  Stage: {stage_name}")
        print(f"{'='*70}")

        if stage_idx == 1:
            print("  Training one FedAvg round...")
            train_one_round(model, dataloaders, device)

        # Extract per-env gradients
        env_grads = {}
        env_losses = {}
        for env in ALL_ENVS:
            grads, loss = extract_per_layer_grads(model, dataloaders[env], device)
            env_grads[env] = grads
            env_losses[env] = loss
            print(f"  {env}: loss={loss:.4f}, layers={len(grads)}")

        # ============================================================
        # Analysis 1: Per-layer gradient norms
        # ============================================================
        print(f"\n  --- Analysis 1: Per-layer gradient norms ---")

        layer_groups = defaultdict(list)
        for env in ALL_ENVS:
            for name, grad in env_grads[env].items():
                group, idx = get_layer_group(name)
                layer_groups[group].append({
                    "env": env, "name": name, "idx": idx,
                    "norm": grad.norm().item(),
                    "grad": grad
                })

        # Summarize by group
        group_norms = {}
        for group, entries in layer_groups.items():
            norm_by_env = {}
            for env in ALL_ENVS:
                env_entries = [e for e in entries if e["env"] == env]
                if env_entries:
                    total_norm = sum(e["norm"]**2 for e in env_entries)**0.5
                    norm_by_env[env] = total_norm
            group_norms[group] = norm_by_env

            print(f"\n  {group}:")
            for env in ALL_ENVS:
                if env in norm_by_env:
                    print(f"    {env:>10}: {norm_by_env[env]:.6f}")

        # ============================================================
        # Analysis 2: Per-layer cosine similarity
        # ============================================================
        print(f"\n  --- Analysis 2: Per-layer cosine similarity ---")

        all_layer_names = list(env_grads[ALL_ENVS[0]].keys())

        cos_results = {}
        for layer_name in all_layer_names:
            group, idx = get_layer_group(layer_name)
            cos_matrix = {}
            for i, e1 in enumerate(ALL_ENVS):
                for j, e2 in enumerate(ALL_ENVS):
                    if j > i:
                        cos = cosine_sim(env_grads[e1][layer_name],
                                         env_grads[e2][layer_name]).item()
                        cos_matrix[f"{e1}_vs_{e2}"] = cos
            cos_results[layer_name] = {
                "group": group, "idx": idx,
                "cosines": cos_matrix
            }

        # Print summary by layer group
        for group in ["attn_q_proj", "attn_v_proj", "attn_o_proj",
                       "ffn_gate", "ffn_up", "ffn_down"]:
            group_entries = {k: v for k, v in cos_results.items() if v["group"] == group}
            if not group_entries:
                continue

            print(f"\n  {group} ({len(group_entries)} layers):")
            for pair_name in ["babyai_vs_webshop", "babyai_vs_textcraft",
                              "maze_vs_webshop", "maze_vs_textcraft",
                              "webshop_vs_textcraft", "babyai_vs_maze"]:
                pair_cosines = [v["cosines"][pair_name]
                                for v in group_entries.values()
                                if pair_name in v["cosines"]]
                if pair_cosines:
                    mean_cos = np.mean(pair_cosines)
                    min_cos = np.min(pair_cosines)
                    max_cos = np.max(pair_cosines)
                    print(f"    {pair_name:>25}: mean={mean_cos:.4f} "
                          f"min={min_cos:.4f} max={max_cos:.4f}")

        # ============================================================
        # Analysis 3: Depth-wise cosine similarity (easy vs hard)
        # ============================================================
        print(f"\n  --- Analysis 3: Depth-wise gradient alignment ---")

        depth_data = defaultdict(lambda: {"easy_hard_cos": [], "hard_hard_cos": [],
                                           "easy_norm": [], "hard_norm": []})
        for layer_name, info in cos_results.items():
            group, idx = info["group"], info["idx"]
            if idx is None:
                continue
            depth_data[idx]["group"] = group

            for pair, cos in info["cosines"].items():
                e1, e2 = pair.split("_vs_")
                if (e1 in EASY_ENVS and e2 in HARD_ENVS) or \
                   (e1 in HARD_ENVS and e2 in EASY_ENVS):
                    depth_data[idx]["easy_hard_cos"].append(cos)
                elif e1 in HARD_ENVS and e2 in HARD_ENVS:
                    depth_data[idx]["hard_hard_cos"].append(cos)

            for env in EASY_ENVS:
                depth_data[idx]["easy_norm"].append(
                    env_grads[env][layer_name].norm().item())
            for env in HARD_ENVS:
                depth_data[idx]["hard_norm"].append(
                    env_grads[env][layer_name].norm().item())

        print(f"\n  {'Layer':>6} | {'Group':>12} | {'E→H cos':>8} | {'H→H cos':>8} | "
              f"{'Easy ||g||':>10} | {'Hard ||g||':>10} | {'E/H norm':>8}")
        print("  " + "-" * 80)
        for idx in sorted(depth_data.keys()):
            d = depth_data[idx]
            eh_cos = np.mean(d["easy_hard_cos"]) if d["easy_hard_cos"] else 0
            hh_cos = np.mean(d["hard_hard_cos"]) if d["hard_hard_cos"] else 0
            e_norm = np.mean(d["easy_norm"]) if d["easy_norm"] else 0
            h_norm = np.mean(d["hard_norm"]) if d["hard_norm"] else 0
            ratio = e_norm / h_norm if h_norm > 0 else float('inf')
            print(f"  {idx:>6} | {d.get('group','?'):>12} | {eh_cos:>8.4f} | {hh_cos:>8.4f} | "
                  f"{e_norm:>10.6f} | {h_norm:>10.6f} | {ratio:>8.2f}")

        # ============================================================
        # Analysis 4: FedAvg signal retention
        # ============================================================
        print(f"\n  --- Analysis 4: FedAvg signal retention ---")

        for layer_name in all_layer_names[:5]:
            grads = {env: env_grads[env][layer_name] for env in ALL_ENVS}
            avg_grad = sum(grads.values()) / len(grads)

            for env in ALL_ENVS:
                retention = cosine_sim(grads[env], avg_grad).item()
                norm_ratio = avg_grad.norm().item() / max(grads[env].norm().item(), 1e-8)

            group, idx = get_layer_group(layer_name)
            retentions = {}
            for env in ALL_ENVS:
                retentions[env] = cosine_sim(grads[env], avg_grad).item()

            if idx is not None and idx <= 2:
                ret_str = "  ".join(f"{e[:3]}={retentions[e]:.3f}" for e in ALL_ENVS)
                print(f"  L{idx:>2} {group:>12}: {ret_str}")

        # ============================================================
        # Analysis 5: Gradient norm decomposition
        # ============================================================
        print(f"\n  --- Analysis 5: Gradient norm by environment group ---")

        for group in ["attn_q_proj", "attn_v_proj", "ffn_gate", "ffn_down"]:
            if group not in group_norms:
                continue
            norms = group_norms[group]
            easy_norm = sum(norms.get(e, 0)**2 for e in EASY_ENVS)**0.5
            hard_norm = sum(norms.get(e, 0)**2 for e in HARD_ENVS)**0.5
            ratio = easy_norm / hard_norm if hard_norm > 0 else float('inf')
            print(f"  {group:>15}: easy={easy_norm:.6f}  hard={hard_norm:.6f}  "
                  f"easy/hard={ratio:.2f}")

        # Save results
        stage_results = {
            "stage": stage_name,
            "env_losses": env_losses,
            "group_norms": {g: {e: float(v) for e, v in norms.items()}
                           for g, norms in group_norms.items()},
            "per_layer_cosine_summary": {}
        }

        for group in ["attn_q_proj", "attn_v_proj", "attn_o_proj",
                       "ffn_gate", "ffn_up", "ffn_down"]:
            group_entries = {k: v for k, v in cos_results.items()
                            if v["group"] == group}
            pair_summary = defaultdict(list)
            for entry in group_entries.values():
                for pair, cos in entry["cosines"].items():
                    pair_summary[pair].append(cos)
            stage_results["per_layer_cosine_summary"][group] = {
                pair: {"mean": float(np.mean(vals)),
                       "min": float(np.min(vals)),
                       "max": float(np.max(vals))}
                for pair, vals in pair_summary.items()
            }

        results["stages"].append(stage_results)

    save_dir = Path("outputs/e9_grad_analysis")
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {save_dir / 'results.json'}")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
