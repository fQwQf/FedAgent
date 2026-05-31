"""Federated aggregation strategies for LoRA parameters.

Supports: FedAvg, FedDebias (p_k-weighted), FedRNK (attention-only), Local, Centralized.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from omegaconf import DictConfig

from src.model import (
    split_lora_state_dict,
    filter_state_dict,
    remove_from_state_dict,
    count_parameters,
)


class AggregationStrategy(ABC):
    @abstractmethod
    def aggregate(self, client_states: list[OrderedDict], cfg: DictConfig,
                  client_weights: list[float] | None = None) -> OrderedDict:
        pass

    @abstractmethod
    def apply_update(self, client_model_state: OrderedDict,
                     global_update: OrderedDict,
                     local_update: OrderedDict,
                     cfg: DictConfig) -> OrderedDict:
        pass


class FedAvgStrategy(AggregationStrategy):
    def aggregate(self, client_states, cfg, client_weights=None):
        return _average_state_dicts(client_states)

    def apply_update(self, client_model_state, global_update, local_update, cfg):
        return global_update


class FedDebiasStrategy(AggregationStrategy):
    """Success-rate debiased aggregation.

    Weights each client's update by p_k (success rate) instead of uniform.
    This corrects the implicit 1/p_k bias in FedAvg+SFT:
        FedAvg aggregates: (1/K) sum g_k^SFT = (1/K) sum (1/p_k) nabla J_k  (biased)
        FedDebias aggregates: sum(p_k * g_k^SFT) / sum(p_k) = sum(nabla J_k) / K  (unbiased)
    """

    def aggregate(self, client_states, cfg, client_weights=None):
        if client_weights is None:
            client_weights = [1.0] * len(client_states)
        return _weighted_average_state_dicts(client_states, client_weights)

    def apply_update(self, client_model_state, global_update, local_update, cfg):
        return global_update


class FedRNKStrategy(AggregationStrategy):
    def aggregate(self, client_states, cfg, client_weights=None):
        attn_states = []
        for state in client_states:
            attn_params, _ = split_lora_state_dict(state, cfg)
            attn_states.append(attn_params)
        return _average_state_dicts(attn_states)

    def apply_update(self, client_model_state, global_update, local_update, cfg):
        attn_params, mlp_params = split_lora_state_dict(client_model_state, cfg)
        merged = OrderedDict()
        merged.update(global_update)
        merged.update(local_update if local_update else mlp_params)
        return merged


class FedRNKInvStrategy(AggregationStrategy):
    def aggregate(self, client_states, cfg, client_weights=None):
        mlp_states = []
        for state in client_states:
            _, mlp_params = split_lora_state_dict(state, cfg)
            mlp_states.append(mlp_params)
        return _average_state_dicts(mlp_states)

    def apply_update(self, client_model_state, global_update, local_update, cfg):
        attn_params, mlp_params = split_lora_state_dict(client_model_state, cfg)
        merged = OrderedDict()
        merged.update(local_update if local_update else attn_params)
        merged.update(global_update)
        return merged


class LocalStrategy(AggregationStrategy):
    def aggregate(self, client_states, cfg, client_weights=None):
        return OrderedDict()

    def apply_update(self, client_model_state, global_update, local_update, cfg):
        return client_model_state


class CentralizedStrategy(AggregationStrategy):
    def aggregate(self, client_states, cfg, client_weights=None):
        return _average_state_dicts(client_states)

    def apply_update(self, client_model_state, global_update, local_update, cfg):
        return global_update


STRATEGIES = {
    "fedavg": FedAvgStrategy,
    "feddebias": FedDebiasStrategy,
    "fedrnk": FedRNKStrategy,
    "fedrnk_inv": FedRNKInvStrategy,
    "local": LocalStrategy,
    "centralized": CentralizedStrategy,
}


def get_strategy(method: str) -> AggregationStrategy:
    if method not in STRATEGIES:
        raise ValueError(f"Unknown method: {method}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[method]()


def _average_state_dicts(state_dicts: list[OrderedDict]) -> OrderedDict:
    if not state_dicts:
        return OrderedDict()

    result = OrderedDict()
    keys = state_dicts[0].keys()

    for key in keys:
        tensors = [sd[key].float() for sd in state_dicts]
        result[key] = torch.stack(tensors).mean(dim=0)

    return result


def _weighted_average_state_dicts(state_dicts: list[OrderedDict],
                                   weights: list[float]) -> OrderedDict:
    if not state_dicts:
        return OrderedDict()

    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum()

    result = OrderedDict()
    keys = state_dicts[0].keys()

    for key in keys:
        tensors = torch.stack([sd[key].float() for sd in state_dicts])
        w_shape = [-1] + [1] * (tensors.dim() - 1)
        result[key] = (tensors * w.view(w_shape)).sum(dim=0)

    return result
