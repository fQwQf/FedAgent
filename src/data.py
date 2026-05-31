"""Trajectory data loading and SFT dataset construction.

Each environment produces trajectories (state, action, reward) tuples.
Successful trajectories (reward=1) are used for SFT training.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class Trajectory:
    env_name: str
    messages: list[dict]
    reward: float

    @property
    def is_success(self) -> bool:
        return self.reward > 0


class SFTDataset(Dataset):
    def __init__(self, trajectories: list[Trajectory], max_length: int = 2048):
        self.samples = [t for t in trajectories if t.is_success]
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        traj = self.samples[idx]
        return {
            "messages": traj.messages,
            "env_name": traj.env_name,
        }


def load_trajectories(data_dir: Path, env_name: str) -> list[Trajectory]:
    trajectory_file = data_dir / env_name / "trajectories.jsonl"
    if not trajectory_file.exists():
        raise FileNotFoundError(f"No trajectory data at {trajectory_file}")

    trajectories = []
    with open(trajectory_file) as f:
        for line in f:
            entry = json.loads(line)
            trajectories.append(Trajectory(
                env_name=entry["env_name"],
                messages=entry["messages"],
                reward=entry.get("reward", 0.0),
            ))
    return trajectories


def filter_successful(trajectories: list[Trajectory]) -> list[Trajectory]:
    return [t for t in trajectories if t.is_success]


def compute_success_rate(trajectories: list[Trajectory]) -> float:
    if not trajectories:
        return 0.0
    return sum(1 for t in trajectories if t.is_success) / len(trajectories)


def create_mock_trajectories(env_name: str, num_traj: int = 50,
                             success_rate: float = 0.5,
                             seed: int = 42) -> list[Trajectory]:
    rng = random.Random(seed)
    trajectories = []
    for i in range(num_traj):
        is_success = rng.random() < success_rate
        messages = [
            {"role": "system", "content": f"You are an agent in {env_name}."},
            {"role": "user", "content": f"Complete task {i}."},
            {"role": "assistant", "content": f"Step-by-step reasoning for task {i} in {env_name}."
                                             f"{' Task completed successfully.' if is_success else ' Task failed.'}"},
        ]
        trajectories.append(Trajectory(
            env_name=env_name,
            messages=messages,
            reward=1.0 if is_success else 0.0,
        ))
    return trajectories
