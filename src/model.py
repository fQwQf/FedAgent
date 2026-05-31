"""Model loading, LoRA application, and Attention/MLP parameter splitting."""

import re
from collections import OrderedDict
from pathlib import Path

import torch
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base_model(cfg: DictConfig):
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=getattr(torch, cfg.model.dtype),
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def apply_lora(model, cfg: DictConfig) -> PeftModel:
    lora_cfg = LoraConfig(
        r=cfg.model.lora.rank,
        lora_alpha=cfg.model.lora.alpha,
        lora_dropout=cfg.model.lora.dropout,
        target_modules=list(cfg.model.lora.target_modules),
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def split_lora_state_dict(state_dict: dict, cfg: DictConfig) -> tuple[dict, dict]:
    attn_modules = set(cfg.model.lora.attn_modules)
    mlp_modules = set(cfg.model.lora.mlp_modules)

    attn_params = OrderedDict()
    mlp_params = OrderedDict()

    for name, param in state_dict.items():
        matched_attn = any(f".{m}." in name for m in attn_modules)
        matched_mlp = any(f".{m}." in name for m in mlp_modules)

        if matched_attn:
            attn_params[name] = param
        elif matched_mlp:
            mlp_params[name] = param

    return attn_params, mlp_params


def filter_state_dict(state_dict: dict, modules: list[str]) -> dict:
    return OrderedDict(
        (name, param)
        for name, param in state_dict.items()
        if any(f".{m}." in name for m in modules)
    )


def remove_from_state_dict(state_dict: dict, modules: list[str]) -> dict:
    return OrderedDict(
        (name, param)
        for name, param in state_dict.items()
        if not any(f".{m}." in name for m in modules)
    )


def count_parameters(state_dict: dict) -> int:
    return sum(p.numel() for p in state_dict.values())


def get_lora_gradients(model, cfg: DictConfig) -> tuple[dict, dict]:
    attn_modules = set(cfg.model.lora.attn_modules)
    mlp_modules = set(cfg.model.lora.mlp_modules)

    attn_grads = OrderedDict()
    mlp_grads = OrderedDict()

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not is_lora_param(name):
            continue

        matched_attn = any(f".{m}." in name for m in attn_modules)
        matched_mlp = any(f".{m}." in name for m in mlp_modules)

        if matched_attn:
            attn_grads[name] = param.grad.detach().clone()
        elif matched_mlp:
            mlp_grads[name] = param.grad.detach().clone()

    return attn_grads, mlp_grads


def is_lora_param(name: str) -> bool:
    return "lora_" in name


def flatten_gradients(grad_dict: dict) -> torch.Tensor:
    return torch.cat([g.flatten() for g in grad_dict.values()])
