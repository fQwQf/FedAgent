"""Federated client: local SFT training with LoRA gradient collection."""

from collections import OrderedDict
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import default_data_collator

from src.data import SFTDataset, Trajectory, compute_success_rate
from src.model import get_lora_gradients, flatten_gradients


class Client:
    def __init__(self, client_id: int, env_name: str, cfg: DictConfig):
        self.client_id = client_id
        self.env_name = env_name
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.local_lora_state = None

    def set_model(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def load_global_model(self, global_lora_state: OrderedDict) -> None:
        if self.model is None:
            raise RuntimeError("Client model not set. Call set_model() first.")
        self.model.load_state_dict(global_lora_state, strict=False)
        self.local_lora_state = OrderedDict({
            k: v.clone() for k, v in global_lora_state.items()
        })

    def local_train(self, trajectories: list[Trajectory]) -> dict:
        dataset = SFTDataset(
            trajectories,
            max_length=self.cfg.training.max_seq_length,
        )
        if len(dataset) == 0:
            return self._empty_result(trajectories)

        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

        self.model.train()
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.training.learning_rate,
        )

        total_loss = 0.0
        num_steps = 0

        for epoch in range(self.cfg.training.local_epochs):
            for batch in dataloader:
                self.optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / self.cfg.training.gradient_accumulation_steps
                loss.backward()

                if (num_steps + 1) % self.cfg.training.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        1.0,
                    )
                    self.optimizer.step()

                total_loss += loss.item() * self.cfg.training.gradient_accumulation_steps
                num_steps += 1

        avg_loss = total_loss / max(num_steps, 1)

        attn_grads, mlp_grads = get_lora_gradients(self.model, self.cfg)
        attn_grad_flat = flatten_gradients(attn_grads) if attn_grads else torch.tensor(0.0)
        mlp_grad_flat = flatten_gradients(mlp_grads) if mlp_grads else torch.tensor(0.0)

        updated_state = OrderedDict({
            k: v.clone() for k, v in self.model.state_dict().items()
            if "lora_" in k
        })

        success_rate = compute_success_rate(trajectories)

        return {
            "loss": avg_loss,
            "num_trajectories": len(trajectories),
            "success_rate": success_rate,
            "updated_lora_state": updated_state,
            "local_lora_state": self.local_lora_state,
            "attn_grad": attn_grad_flat,
            "mlp_grad": mlp_grad_flat,
            "attn_grad_dict": attn_grads,
            "mlp_grad_dict": mlp_grads,
        }

    def evaluate(self, eval_trajectories: list[Trajectory]) -> dict:
        dataset = SFTDataset(eval_trajectories, max_length=self.cfg.training.max_seq_length)
        if len(dataset) == 0:
            return {"score": 0.0}

        self.model.eval()
        total_loss = 0.0
        count = 0

        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            collate_fn=self._collate_fn,
        )

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()
                count += 1

        return {
            "score": 1.0 / (1.0 + total_loss / max(count, 1)),
            "eval_loss": total_loss / max(count, 1),
        }

    def get_delta_state(self, train_result: dict) -> OrderedDict:
        updated = train_result["updated_lora_state"]
        original = train_result["local_lora_state"]
        delta = OrderedDict()
        for key in updated:
            if key in original:
                delta[key] = updated[key].float() - original[key].float()
        return delta

    def _collate_fn(self, batch: list[dict]) -> dict:
        texts = []
        for item in batch:
            text = self.tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.training.max_seq_length,
            return_tensors="pt",
        )
        return tokenized

    def _empty_result(self, trajectories: list[Trajectory]) -> dict:
        return {
            "loss": 0.0,
            "num_trajectories": len(trajectories),
            "success_rate": compute_success_rate(trajectories),
            "updated_lora_state": OrderedDict(),
            "local_lora_state": self.local_lora_state,
            "attn_grad": torch.tensor(0.0),
            "mlp_grad": torch.tensor(0.0),
            "attn_grad_dict": OrderedDict(),
            "mlp_grad_dict": OrderedDict(),
        }
