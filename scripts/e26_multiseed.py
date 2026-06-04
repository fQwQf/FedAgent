#!/usr/bin/env python3
"""
E26: Multi-Seed Validation
Run loss-proportional vs uniform on 0.5B model with 5 different seeds.
Verify robustness of the advantage.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aggregation import _weighted_average_state_dicts as aggregate_lora
from data_loader import load_agentgym_data, get_trainable_messages

print("="*70)
print("E26: Multi-Seed Validation - Qwen2.5-0.5B-Instruct")
print("="*70)

output_base = Path("outputs/e26_multiseed")
output_base.mkdir(parents=True, exist_ok=True)
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
cache_dir = "/data1/tongjizhou/.cache/huggingface/hub"

seeds = [42, 123, 456, 789, 2024]
tasks = ["babyai", "webshop", "textcraft"]
n_samples = 64
max_seq_length = 512

print(f"\nModel: {model_name}")
print(f"Seeds: {seeds}")
print(f"Tasks: {tasks}")
print(f"GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True, cache_dir=cache_dir, local_files_only=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM"
)

print("\nLoading 0.5B model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto",
    cache_dir=cache_dir, trust_remote_code=True, local_files_only=True
)
print(f"Model loaded. Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

class AgentGymDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.examples = []
        for item in data:
            messages, _ = get_trainable_messages(item)
            if not messages:
                continue
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            self.examples.append(text)
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        encoding = self.tokenizer(
            text, truncation=True, max_length=max_seq_length, padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

all_results = {}

for seed in seeds:
    print(f"\n{'='*70}")
    print(f"SEED {seed}")
    print(f"{'='*70}")
    
    seed_dir = output_base / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    results = {}
    
    for method in ["uniform", "loss_proportional"]:
        print(f"\n  Method: {method.upper()}")
        
        model = get_peft_model(base_model, lora_config)
        task_models = {}
        task_losses = {}
        
        for task in tasks:
            data = load_agentgym_data(task, max_samples=n_samples)
            dataset = AgentGymDataset(data, tokenizer)
            
            training_args = TrainingArguments(
                output_dir=str(seed_dir / method / task),
                num_train_epochs=3,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=2,
                learning_rate=1e-4,
                logging_steps=5,
                save_strategy="no",
                report_to="none",
                fp16=True,
                seed=seed,
            )
            
            trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
            train_result = trainer.train()
            final_loss = train_result.training_loss
            task_losses[task] = final_loss
            
            task_models[task] = {k: v.cpu().clone() for k, v in model.state_dict().items() if "lora" in k}
            torch.cuda.empty_cache()
        
        if method == "uniform":
            weights = {t: 1.0/len(tasks) for t in tasks}
        else:
            total_loss = sum(task_losses.values())
            weights = {t: task_losses[t]/total_loss for t in tasks}
        
        task_list = [task_models[t] for t in tasks]
        weight_list = [weights[t] for t in tasks]
        aggregated = aggregate_lora(task_list, weight_list)
        model.load_state_dict(aggregated, strict=False)
        
        eval_results = {}
        for task in tasks:
            data = load_agentgym_data(task, max_samples=n_samples)
            dataset = AgentGymDataset(data, tokenizer)
            
            eval_args = TrainingArguments(
                output_dir=str(seed_dir / method / f"eval_{task}"),
                num_train_epochs=1, per_device_train_batch_size=4,
                logging_steps=1, save_strategy="no", report_to="none", fp16=True,
            )
            
            trainer = Trainer(model=model, args=eval_args, eval_dataset=dataset)
            eval_metrics = trainer.evaluate()
            eval_loss = eval_metrics.get('eval_loss', 0.0)
            eval_results[task] = eval_loss
        
        results[method] = {
            'task_losses': task_losses,
            'eval_results': eval_results,
            'hard_avg': np.mean([eval_results[t] for t in tasks]),
        }
        
        del model
        torch.cuda.empty_cache()
    
    all_results[seed] = results
    
    with open(seed_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    uf = results['uniform']
    lp = results['loss_proportional']
    improvement = (uf['hard_avg'] - lp['hard_avg']) / uf['hard_avg'] * 100
    print(f"\n  Seed {seed}: Loss-proportional improves hard avg by {improvement:.1f}%")


print("\n" + "="*70)
print("MULTI-SEED SUMMARY")
print("="*70)

print(f"\n{'Seed':<10} {'Uniform Hard Avg':<20} {'Loss-Prop Hard Avg':<20} {'Improvement':<15}")
print("-" * 65)

improvements = []
for seed in seeds:
    uf = all_results[seed]['uniform']['hard_avg']
    lp = all_results[seed]['loss_proportional']['hard_avg']
    imp = (uf - lp) / uf * 100
    improvements.append(imp)
    print(f"{seed:<10} {uf:<20.4f} {lp:<20.4f} {imp:<15.1f}%")

print("-" * 65)
print(f"{'Mean':<10} {'':<20} {'':<20} {np.mean(improvements):<15.1f}%")
print(f"{'Std':<10} {'':<20} {'':<20} {np.std(improvements):<15.1f}%")
print(f"{'Min':<10} {'':<20} {'':<20} {np.min(improvements):<15.1f}%")
print(f"{'Max':<10} {'':<20} {'':<20} {np.max(improvements):<15.1f}%")

with open(output_base / 'summary.json', 'w') as f:
    json.dump({
        'seeds': seeds,
        'per_seed_results': {str(s): all_results[s] for s in seeds},
        'statistics': {
            'mean_improvement': float(np.mean(improvements)),
            'std_improvement': float(np.std(improvements)),
            'min_improvement': float(np.min(improvements)),
            'max_improvement': float(np.max(improvements)),
            'all_positive': bool(all(i > 0 for i in improvements)),
        }
    }, f, indent=2)

print(f"\nSummary saved to {output_base / 'summary.json'}")

if all(i > 0 for i in improvements):
    print(f"\n✓ Loss-proportional consistently outperforms uniform across all {len(seeds)} seeds")
else:
    print(f"\n✗ Loss-proportional does NOT consistently outperform uniform")

print("\n" + "="*70)
print("E26 COMPLETE")
print("="*70)
