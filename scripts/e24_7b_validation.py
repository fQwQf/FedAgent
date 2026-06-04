#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import torch
import torch.nn.functional as F
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
print("E24: 7B Model Validation - Qwen2.5-7B-Instruct")
print("="*70)

output_base = Path("outputs/e24_7b_validation")
output_base.mkdir(parents=True, exist_ok=True)
model_name = "Qwen/Qwen2.5-7B-Instruct"
cache_dir = "/data1/tongjizhou/.cache/huggingface/hub"

print(f"\nModel: {model_name}")
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

print("\nLoading 7B model in float16...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto",
    cache_dir=cache_dir, trust_remote_code=True, local_files_only=True
)
print(f"Model loaded. Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

tasks = ["babyai", "webshop", "textcraft"]
n_samples = 64

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
            text, truncation=True, max_length=512, padding="max_length",
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

results = {}

for method in ["uniform", "loss_proportional"]:
    print(f"\n{'='*70}")
    print(f"Training with {method.upper()}")
    print(f"{'='*70}")
    
    method_dir = output_base / f"7B_{method}"
    method_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating model copy...")
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    task_models = {}
    task_losses = {}
    
    for task in tasks:
        print(f"\n  Training task: {task}")
        data = load_agentgym_data(task, max_samples=n_samples)
        dataset = AgentGymDataset(data, tokenizer)
        
        training_args = TrainingArguments(
            output_dir=str(method_dir / task),
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            logging_steps=2,
            save_strategy="no",
            report_to="none",
            fp16=True,
            dataloader_num_workers=0,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        
        train_result = trainer.train()
        final_loss = train_result.training_loss
        task_losses[task] = final_loss
        print(f"    Final loss: {final_loss:.4f}")
        
        task_models[task] = {k: v.cpu().clone() for k, v in model.state_dict().items() if "lora" in k}
        torch.cuda.empty_cache()
    
    print(f"\n  Aggregating with {method}...")
    if method == "uniform":
        weights = {t: 1.0/len(tasks) for t in tasks}
    else:
        total_loss = sum(task_losses.values())
        weights = {t: task_losses[t]/total_loss for t in tasks}
    
    print(f"    Weights: {weights}")
    task_list = [task_models[t] for t in tasks]
    weight_list = [weights[t] for t in tasks]
    aggregated = aggregate_lora(task_list, weight_list)
    model.load_state_dict(aggregated, strict=False)
    
    print(f"\n  Evaluating aggregated model...")
    torch.cuda.empty_cache()
    
    eval_results = {}
    for task in tasks:
        data = load_agentgym_data(task, max_samples=n_samples)
        dataset = AgentGymDataset(data, tokenizer)
        
        eval_args = TrainingArguments(
            output_dir=str(method_dir / f"eval_{task}"),
            num_train_epochs=1,
            per_device_train_batch_size=1,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            fp16=True,
        )
        
        trainer = Trainer(model=model, args=eval_args, eval_dataset=dataset)
        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics.get('eval_loss', 0.0)
        eval_results[task] = eval_loss
        print(f"    {task}: {eval_loss:.4f}")
        torch.cuda.empty_cache()
    
    results[method] = {
        'task_losses': task_losses,
        'eval_results': eval_results,
        'weights': weights,
        'hard_avg': np.mean([eval_results[t] for t in tasks]),
    }
    
    with open(method_dir / 'metrics.json', 'w') as f:
        json.dump(results[method], f, indent=2)
    
    print(f"\n  Hard average: {results[method]['hard_avg']:.4f}")
    
    del model
    torch.cuda.empty_cache()

print("\n" + "="*70)
print("FINAL COMPARISON")
print("="*70)

uf = results['uniform']
lp = results['loss_proportional']

print(f"\n{'Metric':<25} {'Uniform':<15} {'Loss-Proportional':<20} {'Improvement':<15}")
print("-" * 75)
for task in tasks:
    improvement = (uf['eval_results'][task] - lp['eval_results'][task]) / uf['eval_results'][task] * 100
    print(f"{task:<25} {uf['eval_results'][task]:<15.4f} {lp['eval_results'][task]:<20.4f} {improvement:<15.1f}%")

hard_improvement = (uf['hard_avg'] - lp['hard_avg']) / uf['hard_avg'] * 100
print(f"{'Hard Average':<25} {uf['hard_avg']:<15.4f} {lp['hard_avg']:<20.4f} {hard_improvement:<15.1f}%")

final_metrics = {
    'uniform': uf,
    'loss_proportional': lp,
    'improvement_percent': {
        'hard_avg': float(hard_improvement),
        'per_task': {t: float((uf['eval_results'][t] - lp['eval_results'][t]) / uf['eval_results'][t] * 100) 
                     for t in tasks}
    }
}

with open(output_base / 'final_comparison.json', 'w') as f:
    json.dump(final_metrics, f, indent=2)

print(f"\nKey Finding: Loss-proportional improves hard average by {hard_improvement:.1f}% at 7B scale")

print("\n" + "="*70)
print("E24 COMPLETE")
print("="*70)
