"""Main federated training loop for FedRNK experiments."""

import argparse
import copy
from collections import OrderedDict
from pathlib import Path

import torch
from omegaconf import DictConfig

from src.config import load_config, save_config, get_aggregate_modules
from src.logging_utils import ExperimentLogger
from src.model import load_base_model, apply_lora, split_lora_state_dict, count_parameters
from src.client import Client
from src.aggregation import get_strategy
from src.data import create_mock_trajectories


def get_env_assignments(cfg: DictConfig) -> list[str]:
    return ["babyai", "webshop", "textcraft", "maze", "wordle"][:cfg.federation.num_clients]


def federated_train(cfg: DictConfig) -> None:
    logger = ExperimentLogger(cfg)
    save_config(cfg, Path(cfg.logging.log_dir) / cfg.logging.experiment_name / "config.yaml")

    logger.log("experiment_start",
               method=cfg.federation.method,
               num_clients=cfg.federation.num_clients,
               num_rounds=cfg.training.num_rounds,
               model=cfg.model.name)

    model, tokenizer = load_base_model(cfg)
    model = apply_lora(model, cfg)

    strategy = get_strategy(cfg.federation.method)
    env_names = get_env_assignments(cfg)

    clients = [
        Client(client_id=k, env_name=env_names[k], cfg=cfg)
        for k in range(cfg.federation.num_clients)
    ]

    all_gradient_data = []

    for round_idx in range(cfg.training.num_rounds):
        logger.log_round_start(round_idx)

        client_updates = []
        round_gradient_data = {}

        for client in clients:
            client.set_model(copy.deepcopy(model), tokenizer)

            trajectories = create_mock_trajectories(
                env_name=client.env_name,
                num_traj=50,
                success_rate={"babyai": 0.83, "webshop": 0.73, "textcraft": 0.67,
                              "maze": 0.28, "wordle": 0.05}.get(client.env_name, 0.5),
                seed=cfg.training.seed + round_idx * 100 + client.client_id,
            )

            result = client.local_train(trajectories)

            logger.log_client_update(
                round_idx=round_idx,
                client_id=client.client_id,
                env=client.env_name,
                loss=result["loss"],
                num_traj=result["num_trajectories"],
                success_rate=result["success_rate"],
            )

            delta = client.get_delta_state(result)
            client_updates.append(delta)

            if cfg.logging.log_gradients:
                attn_norm = result["attn_grad"].norm().item() if result["attn_grad"].numel() > 1 else 0.0
                mlp_norm = result["mlp_grad"].norm().item() if result["mlp_grad"].numel() > 1 else 0.0

                logger.log_gradient_stats(
                    round_idx=round_idx,
                    client_id=client.client_id,
                    env=client.env_name,
                    attn_grad_norm=attn_norm,
                    mlp_grad_norm=mlp_norm,
                )

                round_gradient_data[client.client_id] = {
                    "env": client.env_name,
                    "attn_grad": result["attn_grad"],
                    "mlp_grad": result["mlp_grad"],
                    "attn_grad_dict": result["attn_grad_dict"],
                    "mlp_grad_dict": result["mlp_grad_dict"],
                }

        global_update = strategy.aggregate(client_updates, cfg)

        agg_modules = get_aggregate_modules(cfg)
        all_modules = list(cfg.model.lora.attn_modules) + list(cfg.model.lora.mlp_modules)
        iso_modules = [m for m in all_modules if m not in agg_modules]

        logger.log_aggregation(
            round_idx=round_idx,
            method=cfg.federation.method,
            num_shared=count_parameters(global_update) if global_update else 0,
            num_isolated=sum(count_parameters(s) for s in client_updates) -
                         (count_parameters(global_update) if global_update else 0),
        )

        for client in clients:
            pass

        if round_gradient_data:
            all_gradient_data.append({
                "round": round_idx,
                "clients": round_gradient_data,
            })

        logger.log_round_end(round_idx)

    if all_gradient_data:
        _save_gradient_data(all_gradient_data, cfg)

    logger.log("experiment_end")
    logger.close()


def _save_gradient_data(gradient_data: list, cfg: DictConfig) -> None:
    save_dir = Path(cfg.logging.log_dir) / cfg.logging.experiment_name / "analysis"
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(gradient_data, save_dir / "gradient_data.pt")
    print(f"\nGradient data saved to {save_dir / 'gradient_data.pt'}")


def main():
    parser = argparse.ArgumentParser(description="FedRNK Federated Training")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("overrides", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(
        experiment=args.experiment,
        method=args.method,
        env=args.env,
        overrides=args.overrides,
    )

    federated_train(cfg)


if __name__ == "__main__":
    main()
