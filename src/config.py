"""Configuration loading and merging for FedRNK experiments."""

import copy
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def load_config(
    experiment: str | None = None,
    method: str | None = None,
    env: str | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    base = OmegaConf.load(CONFIGS_DIR / "base.yaml")

    configs = [base]

    if experiment:
        p = CONFIGS_DIR / f"experiment_{experiment}.yaml"
        if p.exists():
            configs.append(OmegaConf.load(p))

    if method:
        p = CONFIGS_DIR / "methods" / f"{method}.yaml"
        if p.exists():
            configs.append(OmegaConf.load(p))

    if env:
        p = CONFIGS_DIR / "envs" / f"{env}.yaml"
        if p.exists():
            configs.append(OmegaConf.load(p))

    merged = OmegaConf.merge(*configs)

    if overrides:
        cli = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(merged, cli)

    OmegaConf.resolve(merged)
    return merged


def save_config(cfg: DictConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def get_attn_modules(cfg: DictConfig) -> list[str]:
    return list(cfg.model.lora.attn_modules)


def get_mlp_modules(cfg: DictConfig) -> list[str]:
    return list(cfg.model.lora.mlp_modules)


def get_aggregate_modules(cfg: DictConfig) -> list[str]:
    method = cfg.federation.method
    if method == "fedrnk":
        return get_attn_modules(cfg)
    elif method == "fedrnk_inv":
        return get_mlp_modules(cfg)
    elif method == "fedavg":
        return get_attn_modules(cfg) + get_mlp_modules(cfg)
    else:
        return []
