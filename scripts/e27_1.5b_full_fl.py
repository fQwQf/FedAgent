#!/usr/bin/env python3
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from multiprocessing import Process, Queue

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_agentgym_data, get_trainable_messages

def train_task_worker(gpu_id, task_name, global_lora_state, data_cache, result_queue, model_name, cache_dir, epoch=1):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none", task_type="CAUSAL_LM")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", cache_dir=cache_dir, trust_remote_code=True, local_files_only=True)
    model = get_peft_model(model, lora_config)
    if global_lora_state is not None:
        model.load_state_dict(global_lora_state, strict=False)
    class TaskDataset(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer, max_len=512):
            self.examples = []
            for item in data:
                messages, _ = get_trainable_messages(item)
                if not messages:
                    continue
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                self.examples.append(text)
            self.tokenizer = tokenizer
            self.max_len = max_len
        def __len__(self):
            return len(self.examples)
        def __getitem__(self, idx):
            text = self.examples[idx]
            encoding = self.tokenizer(text, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    dataset = TaskDataset(data_cache, tokenizer)
    training_args = TrainingArguments(output_dir=f"/tmp/e27_{task_name}_gpu{gpu_id}", num_train_epochs=epoch, per_device_train_batch_size=4, gradient_accumulation_steps=2, learning_rate=1e-4, logging_steps=5, save_strategy="no", report_to="none", fp16=True, seed=42)
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    train_result = trainer.train()
    final_loss = train_result.training_loss
    lora_state = {k: v.cpu().clone() for k, v in model.state_dict().items() if "lora" in k}
    result_queue.put({"task": task_name, "loss": final_loss, "lora_state": lora_state, "gpu": gpu_id})
    del model, trainer
    torch.cuda.empty_cache()

def evaluate_task_worker(gpu_id, task_name, global_lora_state, data_cache, result_queue, model_name, cache_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none", task_type="CAUSAL_LM")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", cache_dir=cache_dir, trust_remote_code=True, local_files_only=True)
    model = get_peft_model(model, lora_config)
    model.load_state_dict(global_lora_state, strict=False)
    class TaskDataset(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer, max_len=512):
            self.examples = []
            for item in data:
                messages, _ = get_trainable_messages(item)
                if not messages:
                    continue
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                self.examples.append(text)
            self.tokenizer = tokenizer
            self.max_len = max_len
        def __len__(self):
            return len(self.examples)
        def __getitem__(self, idx):
            text = self.examples[idx]
            encoding = self.tokenizer(text, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    dataset = TaskDataset(data_cache, tokenizer)
    eval_args = TrainingArguments(output_dir=f"/tmp/e27_eval_{task_name}", per_device_eval_batch_size=4, report_to="none", fp16=True)
    trainer = Trainer(model=model, args=eval_args, eval_dataset=dataset)
    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics.get("eval_loss", 0.0)
    result_queue.put({"task": task_name, "eval_loss": eval_loss, "gpu": gpu_id})
    del model, trainer
    torch.cuda.empty_cache()

def aggregate_lora_weights(task_states, weights):
    aggregated = {}
    keys = task_states[0].keys()
    for key in keys:
        tensors = [state[key].float() for state in task_states]
        stacked = torch.stack(tensors)
        w = torch.tensor(weights, dtype=torch.float32)
        w = w / w.sum()
        w_shape = [-1] + [1] * (stacked.dim() - 1)
        aggregated[key] = (stacked * w.view(w_shape)).sum(dim=0)
    return aggregated

def run_fl_round(gpu_ids, tasks, task_data, global_state, method, model_name, cache_dir, round_num):
    print(f"\n{'='*70}")
    print(f"Round {round_num} - {method.upper()}")
    print(f"{'='*70}")
    print("  Training tasks in parallel...")
    result_queue = Queue()
    processes = []
    for i, task in enumerate(tasks):
        p = Process(target=train_task_worker, args=(gpu_ids[i], task, global_state, task_data[task], result_queue, model_name, cache_dir))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    task_losses = {r["task"]: r["loss"] for r in results}
    task_states = [r["lora_state"] for r in results]
    print(f"  Training losses: {task_losses}")
    if method == "uniform":
        weights = [1.0 / len(tasks)] * len(tasks)
    else:
        total_loss = sum(task_losses[t] for t in tasks)
        weights = [task_losses[t] / total_loss for t in tasks]
    print(f"  Aggregation weights: {dict(zip(tasks, [f'{w:.3f}' for w in weights]))}")
    aggregated_state = aggregate_lora_weights(task_states, weights)
    print("  Evaluating aggregated model...")
    eval_queue = Queue()
    eval_processes = []
    for i, task in enumerate(tasks):
        p = Process(target=evaluate_task_worker, args=(gpu_ids[i], task, aggregated_state, task_data[task], eval_queue, model_name, cache_dir))
        p.start()
        eval_processes.append(p)
    for p in eval_processes:
        p.join()
    eval_results = []
    while not eval_queue.empty():
        eval_results.append(eval_queue.get())
    eval_losses = {r["task"]: r["eval_loss"] for r in eval_results}
    print(f"  Eval losses: {eval_losses}")
    return aggregated_state, task_losses, eval_losses

def main():
    print("="*70)
    print("E27: Full Multi-Round FL - Qwen2.5-1.5B-Instruct")
    print("="*70)
    output_base = Path("outputs/e27_1.5b_full_fl")
    output_base.mkdir(parents=True, exist_ok=True)
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    cache_dir = "/data1/tongjizhou/.cache/huggingface/hub"
    tasks = ["babyai", "webshop", "textcraft"]
    gpu_ids = [0, 1, 2]
    n_samples = 64
    n_rounds = 5
    print(f"\nModel: {model_name}")
    print(f"Tasks: {tasks}")
    print(f"GPUs: {gpu_ids}")
    print(f"Rounds: {n_rounds}")
    print(f"Samples per task: {n_samples}")
    print("\nLoading task data...")
    task_data = {}
    for task in tasks:
        task_data[task] = load_agentgym_data(task, max_samples=n_samples)
        print(f"  {task}: {len(task_data[task])} samples")
    all_results = {}
    for method in ["uniform", "loss_proportional"]:
        print(f"\n{'#'*70}")
        print(f"# METHOD: {method.upper()}")
        print(f"{'#'*70}")
        method_results = []
        global_state = None
        for round_num in range(n_rounds):
            global_state, task_losses, eval_losses = run_fl_round(gpu_ids, tasks, task_data, global_state, method, model_name, cache_dir, round_num)
            easy_tasks = ["babyai"]
            hard_tasks = ["webshop", "textcraft"]
            result_record = {"round": round_num, "method": method, "task_losses": task_losses, "eval_losses": eval_losses, "easy_avg": np.mean([eval_losses[t] for t in easy_tasks]), "hard_avg": np.mean([eval_losses[t] for t in hard_tasks]), "avg_eval_loss": np.mean([eval_losses[t] for t in tasks])}
            method_results.append(result_record)
            print(f"\n  Round {round_num} Summary:")
            print(f"    Easy avg: {result_record['easy_avg']:.4f}")
            print(f"    Hard avg: {result_record['hard_avg']:.4f}")
            print(f"    Overall avg: {result_record['avg_eval_loss']:.4f}")
        all_results[method] = method_results
        with open(output_base / f"{method}_metrics.json", "w") as f:
            json.dump(method_results, f, indent=2)
    print("\n" + "="*70)
    print("FINAL COMPARISON (Round 5)")
    print("="*70)
    uf_final = all_results["uniform"][-1]
    lp_final = all_results["loss_proportional"][-1]
    print(f"\n{'Metric':<25} {'Uniform':<15} {'Loss-Proportional':<20} {'Improvement':<15}")
    print("-" * 75)
    for task in tasks:
        improvement = (uf_final['eval_losses'][task] - lp_final['eval_losses'][task]) / uf_final['eval_losses'][task] * 100
        print(f"{task:<25} {uf_final['eval_losses'][task]:<15.4f} {lp_final['eval_losses'][task]:<20.4f} {improvement:<15.1f}%")
    hard_improvement = (uf_final['hard_avg'] - lp_final['hard_avg']) / uf_final['hard_avg'] * 100
    print(f"{'Hard Average':<25} {uf_final['hard_avg']:<15.4f} {lp_final['hard_avg']:<20.4f} {hard_improvement:<15.1f}%")
    print(f"\nKey Finding: Loss-proportional improves hard average by {hard_improvement:.1f}% at 1.5B scale")
    with open(output_base / "final_comparison.json", "w") as f:
        json.dump({"uniform": uf_final, "loss_proportional": lp_final, "improvement_percent": {"hard_avg": float(hard_improvement), "per_task": {t: float((uf_final['eval_losses'][t] - lp_final['eval_losses'][t]) / uf_final['eval_losses'][t] * 100) for t in tasks}}}, f, indent=2)
    print(f"\nAll results saved to {output_base}")
    print("\n" + "="*70)
    print("E27 COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
