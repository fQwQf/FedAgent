"""Convert AgentGym trajectory format to FedRNK internal format.

AgentGym format:
  {"conversations": [{"from": "human"|"gpt", "loss": null|true|false, "value": "..."}]}

FedRNK format (for SFT training):
  We directly load conversations and use loss=true turns as training targets.
"""

import json
import random
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

ENV_FILES = {
    "babyai": "babyai_train.json",
    "webshop": "webshop_train.json",
    "textcraft": "textcraft_train.json",
    "maze": "lmrlgym_maze_train.json",
    "wordle": "lmrlgym_wordle_train.json",
}


def load_agentgym_data(env_name: str, max_samples: int | None = None,
                       seed: int = 42) -> list[dict]:
    fname = ENV_FILES.get(env_name)
    if fname is None:
        raise ValueError(f"Unknown env: {env_name}. Available: {list(ENV_FILES.keys())}")

    path = DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    if max_samples and len(data) > max_samples:
        rng = random.Random(seed)
        data = rng.sample(data, max_samples)

    return data


def convert_to_messages(trajectory: dict) -> list[dict]:
    """Convert AgentGym conversation to standard chat format.

    Returns list of {"role": "user"|"assistant", "content": "..."} messages.
    Only includes turns that are part of the actual interaction
    (skips system prompt and initial confirmation).
    """
    conversations = trajectory["conversations"]
    messages = []

    for turn in conversations:
        role = "user" if turn["from"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["value"]})

    return messages


def get_trainable_messages(trajectory: dict) -> list[dict]:
    """Get only the messages that should be trained on (loss=true).

    Returns full conversation context but marks which turns are training targets.
    For SFT, we typically train on the full conversation with a mask on non-loss turns.
    """
    conversations = trajectory["conversations"]
    messages = []
    labels = []

    for turn in conversations:
        role = "user" if turn["from"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["value"]})
        labels.append(turn.get("loss", False) is True)

    return messages, labels


def get_env_stats() -> dict:
    """Get statistics for all environments."""
    stats = {}
    for env_name, fname in ENV_FILES.items():
        path = DATA_DIR / fname
        if not path.exists():
            stats[env_name] = {"error": "file not found"}
            continue

        with open(path) as f:
            data = json.load(f)

        total_turns = sum(len(t["conversations"]) for t in data)
        action_turns = sum(
            1 for t in data for turn in t["conversations"]
            if turn.get("loss") is True
        )

        stats[env_name] = {
            "num_trajectories": len(data),
            "total_turns": total_turns,
            "action_turns": action_turns,
            "avg_trajectory_length": total_turns / len(data) if data else 0,
            "avg_actions": action_turns / len(data) if data else 0,
            "file_size_mb": path.stat().st_size / 1024 / 1024,
        }

    return stats


if __name__ == "__main__":
    import sys
    stats = get_env_stats()
    print(f"{'Environment':>12} | {'Trajectories':>12} | {'Avg Actions':>11} | {'Size (MB)':>9}")
    print("-" * 60)
    for env, s in stats.items():
        if "error" in s:
            print(f"{env:>12} | ERROR: {s['error']}")
        else:
            print(f"{env:>12} | {s['num_trajectories']:>12d} | {s['avg_actions']:>11.1f} | {s['file_size_mb']:>9.1f}")
