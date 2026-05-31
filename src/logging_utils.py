"""Structured JSONL logging with console output for FedRNK experiments."""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig


class ExperimentLogger:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        exp_name = cfg.logging.experiment_name
        self.log_dir = Path(cfg.logging.log_dir) / exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.events_path = self.log_dir / "events.jsonl"
        self.metrics_path = self.log_dir / "metrics.jsonl"

        self._events_file = open(self.events_path, "a")
        self._metrics_file = open(self.metrics_path, "a")

        self._start_time = time.time()
        self._round_start_time = None

    def log(self, event_type: str, **kwargs) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": round(time.time() - self._start_time, 1),
            "type": event_type,
            **kwargs,
        }

        line = json.dumps(entry, ensure_ascii=False, default=str)
        self._events_file.write(line + "\n")
        self._events_file.flush()

        if "loss" in kwargs or "score" in kwargs or "reward" in kwargs or "grad_norm" in kwargs:
            self._metrics_file.write(line + "\n")
            self._metrics_file.flush()

        self._console_print(event_type, kwargs)

    def log_round_start(self, round_idx: int) -> None:
        self._round_start_time = time.time()
        self.log("round_start", round=round_idx)

    def log_round_end(self, round_idx: int) -> None:
        elapsed = time.time() - self._round_start_time if self._round_start_time else 0
        self.log("round_end", round=round_idx, round_time_sec=round(elapsed, 1))

    def log_client_update(self, round_idx: int, client_id: int, env: str,
                          loss: float, num_traj: int, success_rate: float, **extra) -> None:
        self.log(
            "client_update",
            round=round_idx,
            client=client_id,
            env=env,
            loss=round(loss, 4),
            num_trajectories=num_traj,
            success_rate=round(success_rate, 4),
            **extra,
        )

    def log_aggregation(self, round_idx: int, method: str,
                        num_shared: int, num_isolated: int) -> None:
        self.log(
            "aggregation",
            round=round_idx,
            method=method,
            num_params_shared=num_shared,
            num_params_isolated=num_isolated,
        )

    def log_gradient_stats(self, round_idx: int, client_id: int, env: str,
                           attn_grad_norm: float, mlp_grad_norm: float,
                           attn_cosine_with_global: float | None = None,
                           mlp_cosine_with_global: float | None = None,
                           attn_deviation_norm: float | None = None,
                           mlp_deviation_norm: float | None = None,
                           **extra) -> None:
        self.log(
            "gradient_stats",
            round=round_idx,
            client=client_id,
            env=env,
            attn_grad_norm=round(attn_grad_norm, 6),
            mlp_grad_norm=round(mlp_grad_norm, 6),
            **({"attn_cosine_with_global": round(attn_cosine_with_global, 4)} if attn_cosine_with_global is not None else {}),
            **({"mlp_cosine_with_global": round(mlp_cosine_with_global, 4)} if mlp_cosine_with_global is not None else {}),
            **({"attn_deviation_norm": round(attn_deviation_norm, 6)} if attn_deviation_norm is not None else {}),
            **({"mlp_deviation_norm": round(mlp_deviation_norm, 6)} if mlp_deviation_norm is not None else {}),
            **extra,
        )

    def log_eval(self, round_idx: int, client_id: int, env: str, score: float, **extra) -> None:
        self.log("eval", round=round_idx, client=client_id, env=env, score=round(score, 4), **extra)

    def close(self) -> None:
        self._events_file.close()
        self._metrics_file.close()

    def _console_print(self, event_type: str, data: dict) -> None:
        round_str = f"R{data['round']}" if "round" in data else "  "
        client_str = f"C{data['client']}" if "client" in data else "  "

        if event_type == "round_start":
            print(f"\n{'='*60}\n  Round {data.get('round', '?')} START\n{'='*60}")
        elif event_type == "round_end":
            print(f"  Round {data.get('round', '?')} END  [{data.get('round_time_sec', '?')}s]")
        elif event_type == "client_update":
            print(f"  [{round_str}|{client_str}|{data.get('env',''):>10}] "
                  f"loss={data.get('loss','?'):8.4f}  "
                  f"sr={data.get('success_rate','?'):.3f}  "
                  f"traj={data.get('num_trajectories','?')}")
        elif event_type == "aggregation":
            print(f"  [{round_str}| AGG|{data.get('method',''):>10}] "
                  f"shared={data.get('num_params_shared',0)}  "
                  f"isolated={data.get('num_params_isolated',0)}")
        elif event_type == "gradient_stats":
            print(f"  [{round_str}|{client_str}|{data.get('env',''):>10}] "
                  f"attn_norm={data.get('attn_grad_norm',0):.4f}  "
                  f"mlp_norm={data.get('mlp_grad_norm',0):.4f}")
        elif event_type == "eval":
            print(f"  [{round_str}|{client_str}|{data.get('env',''):>10}] "
                  f"score={data.get('score',0):.4f}")
        sys.stdout.flush()
