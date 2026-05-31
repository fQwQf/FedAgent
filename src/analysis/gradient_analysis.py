"""E0 experiment: Collect and analyze gradient consistency across environments.

Measures P1: ||Δ_k^{Attn}||^2 vs ||Δ_k^{MLP}||^2
- Δ_k = ∇J_k - ḡ (deviation from global gradient)
- If ||Δ^{Attn}||^2 / ||Δ^{MLP}||^2 < 0.5, H1 is supported
"""

import argparse
from collections import OrderedDict
from pathlib import Path

import torch
from omegaconf import DictConfig

from src.config import load_config, save_config
from src.logging_utils import ExperimentLogger
from src.model import load_base_model, apply_lora, get_lora_gradients, flatten_gradients
from src.data import create_mock_trajectories
from src.client import Client


def run_e0(cfg: DictConfig) -> dict:
    logger = ExperimentLogger(cfg)
    save_config(cfg, Path(cfg.logging.log_dir) / cfg.logging.experiment_name / "config.yaml")

    model, tokenizer = load_base_model(cfg)
    model = apply_lora(model, cfg)

    env_configs = {
        "babyai":    {"success_rate": 0.83},
        "webshop":   {"success_rate": 0.73},
        "textcraft": {"success_rate": 0.67},
        "maze":      {"success_rate": 0.28},
        "wordle":    {"success_rate": 0.05},
    }

    env_gradients = {}

    for env_name, env_cfg in env_configs.items():
        logger.log("env_start", env=env_name)

        client = Client(client_id=0, env_name=env_name, cfg=cfg)
        client.set_model(model, tokenizer)

        trajectories = create_mock_trajectories(
            env_name=env_name,
            num_traj=30,
            success_rate=env_cfg["success_rate"],
            seed=cfg.training.seed,
        )

        result = client.local_train(trajectories)

        env_gradients[env_name] = {
            "attn_grad": result["attn_grad"],
            "mlp_grad": result["mlp_grad"],
            "attn_grad_dict": result["attn_grad_dict"],
            "mlp_grad_dict": result["mlp_grad_dict"],
            "success_rate": env_cfg["success_rate"],
        }

        logger.log("env_done",
                    env=env_name,
                    loss=result["loss"],
                    attn_grad_norm=result["attn_grad"].norm().item(),
                    mlp_grad_norm=result["mlp_grad"].norm().item(),
                    success_rate=env_cfg["success_rate"])

    results = analyze_gradient_consistency(env_gradients, cfg)

    logger.log("e0_results", **results)

    save_dir = Path(cfg.logging.log_dir) / cfg.logging.experiment_name / "analysis"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"env_gradients": env_gradients, "results": results},
               save_dir / "e0_results.pt")

    logger.close()
    return results


def analyze_gradient_consistency(env_gradients: dict, cfg: DictConfig) -> dict:
    env_names = list(env_gradients.keys())
    n = len(env_names)

    attn_grads = {k: v["attn_grad"] for k, v in env_gradients.items()}
    mlp_grads = {k: v["mlp_grad"] for k, v in env_gradients.items()}

    global_attn = torch.stack([attn_grads[k] for k in env_names]).mean(dim=0)
    global_mlp = torch.stack([mlp_grads[k] for k in env_names]).mean(dim=0)

    attn_deviations = {}
    mlp_deviations = {}
    for env in env_names:
        attn_deviations[env] = attn_grads[env] - global_attn
        mlp_deviations[env] = mlp_grads[env] - global_mlp

    attn_dev_norm_sq = sum(d.norm().item() ** 2 for d in attn_deviations.values()) / n
    mlp_dev_norm_sq = sum(d.norm().item() ** 2 for d in mlp_deviations.values()) / n

    attn_cosine_matrix = torch.zeros(n, n)
    mlp_cosine_matrix = torch.zeros(n, n)
    for i, ei in enumerate(env_names):
        for j, ej in enumerate(env_names):
            if i == j:
                attn_cosine_matrix[i, j] = 1.0
                mlp_cosine_matrix[i, j] = 1.0
            else:
                attn_cosine_matrix[i, j] = torch.nn.functional.cosine_similarity(
                    attn_grads[ei].unsqueeze(0), attn_grads[ej].unsqueeze(0)
                ).item()
                mlp_cosine_matrix[i, j] = torch.nn.functional.cosine_similarity(
                    mlp_grads[ei].unsqueeze(0), mlp_grads[ej].unsqueeze(0)
                ).item()

    attn_avg_cosine = attn_cosine_matrix[numpy_triu_indices(n)].mean().item()
    mlp_avg_cosine = mlp_cosine_matrix[numpy_triu_indices(n)].mean().item()

    beta_sq = [(1.0 - env_gradients[e]["success_rate"]) ** 2 / env_gradients[e]["success_rate"] ** 2
               for e in env_names]
    mlp_dev_norms = [mlp_deviations[e].norm().item() for e in env_names]
    variance_corr = _pearson_corr(beta_sq, mlp_dev_norms)

    ratio = attn_dev_norm_sq / mlp_dev_norm_sq if mlp_dev_norm_sq > 0 else float("inf")

    return {
        "attn_deviation_norm_sq": attn_dev_norm_sq,
        "mlp_deviation_norm_sq": mlp_dev_norm_sq,
        "ratio": ratio,
        "h1_supported": ratio < 0.5,
        "attn_avg_cosine": attn_avg_cosine,
        "mlp_avg_cosine": mlp_avg_cosine,
        "attn_cosine_matrix": attn_cosine_matrix.tolist(),
        "mlp_cosine_matrix": mlp_cosine_matrix.tolist(),
        "variance_beta_corr": variance_corr,
        "env_names": env_names,
    }


def numpy_triu_indices(n: int, k: int = 1):
    indices = []
    for i in range(n):
        for j in range(i + k, n):
            indices.append((i, j))
    return indices


def _pearson_corr(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    sx = sum((xi - mx) ** 2 for xi in x)
    sy = sum((yi - my) ** 2 for yi in y)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return cov / (sx * sy) ** 0.5


def main():
    parser = argparse.ArgumentParser(description="FedRNK E0 Gradient Consistency Probe")
    parser.add_argument("--experiment", type=str, default="e0")
    parser.add_argument("overrides", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(experiment=args.experiment, overrides=args.overrides)
    results = run_e0(cfg)

    print("\n" + "=" * 60)
    print("  E0 RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Attn deviation ||Δ^attn||² = {results['attn_deviation_norm_sq']:.6f}")
    print(f"  MLP  deviation ||Δ^mlp||²  = {results['mlp_deviation_norm_sq']:.6f}")
    print(f"  Ratio (attn/mlp)           = {results['ratio']:.4f}")
    print(f"  H1 supported (ratio < 0.5) = {results['h1_supported']}")
    print(f"  Attn avg cosine sim         = {results['attn_avg_cosine']:.4f}")
    print(f"  MLP  avg cosine sim         = {results['mlp_avg_cosine']:.4f}")
    print(f"  β²-MLP deviation corr       = {results['variance_beta_corr']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
