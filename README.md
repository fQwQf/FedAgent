# FedAgent

> **Loss-Proportional Aggregation for Heterogeneous Task Federated Learning**
> 
> Proving that simple loss-proportional aggregation is theoretically optimal for heterogeneous task FL, with systematic validation across model scales (0.5B → 7B+).

---

## Quick Overview

**One-sentence summary:** Loss-proportional aggregation ($w_k = L_k / \sum_j L_j$) is the unique optimal linear strategy for heterogeneous task FL — it implicitly minimizes worst-case loss via softmax, automatically implements curriculum learning, and eliminates noise floors from converged tasks.

**Key result:** 43.6% improvement on hard tasks at 7B scale (5-round FL), exceeding 20.8% at 0.5B scale.

**Theory:** 8 theorems with proofs (softmax identification, minimax convergence, rate equalization, linear collapse, noise floor bounds, structural conflict, dominance, convergence rate gap).

**Experiments:** 28 systematic experiments validating every theoretical claim.

---

## Repository Structure

```
FedAgent/
├── docs/                          # Research documentation (primary output)
│   ├── theory.md                  # 8 theorems + proofs + Three Regimes framework
│   ├── experiments.md             # Complete experiment log (E5-E28)
│   ├── research_plan.md           # Roadmap: completed + planned work
│   ├── failures.md                # Failed experiments and why (14 cases)
│   ├── novelty_assessment.md      # Insight analysis + novelty scoring
│   ├── related_work_and_references.md  # Literature library
│   └── archive/                   # Early exploration (E0-E5)
│       ├── research_archive.md
│       └── analysis_comprehensive.md
│
├── src/                           # Core library
│   ├── aggregation.py             # Weighted LoRA aggregation
│   ├── data_loader.py             # AgentGym data loading
│   ├── model.py                   # Model loading + LoRA config
│   ├── client.py                  # Single-task training
│   ├── train.py                   # FL training orchestrator
│   └── analysis/                  # Visualization tools
│
├── scripts/                       # Experiment scripts (one per experiment)
│   ├── e8_loss_weighted.py        # Core: loss-proportional vs uniform
│   ├── e14_lafa.py                # Theorem 4: linear collapse
│   ├── e23_theory_validation.py   # Theorem 6-7 synthetic validation
│   ├── e28_7b_full_fl.py          # 7B model, 5-round FL (43.6% improvement)
│   └── ... (28 total)
│
├── outputs/                       # Experiment results (gitignored)
├── data/raw/                      # AgentGym trajectories
└── README.md                      # This file
```

---

## Core Results

### Three Regimes Framework

| Regime | Condition | Loss-Proportional Effect | Key Experiment |
|--------|-----------|-------------------------|----------------|
| **A** | Small model + Multi-round FL | **20.8%** improvement | E8 (0.5B, 5 rounds) |
| **B** | Large model + Multi-round FL | **43.6%** improvement | E28 (7B, 5 rounds) |
| **C** | Single-round training | **~0%** (no advantage) | E24/25/26 |

**Counter-intuitive:** Advantage *increases* with model size because larger models have more complex loss landscapes with deeper suboptimal basins for hard tasks.

### Failed Methods (All Linear Alternatives Collapse)

- **LAFA** (layer-adaptive weights): Collapses to global uniform within 5 rounds
- **Curriculum learning**: Catastrophic forgetting (+27.9% worse)
- **Prototype alignment**: ~0% improvement (SFT already aligns)
- **SeqComp-LoRA**: +64% worse (LoRA merge is not compositional)

**Why:** Theorem 4 proves any linear parameter-space operation is equivalent to global reweighting when averaged over training.

---

## Quick Start

### Prerequisites

```bash
# Conda environment (pre-configured)
conda activate realm

# Required packages (already installed)
# torch, transformers, peft, datasets, accelerate
```

### Run Core Experiment (E8)

```bash
python scripts/e8_loss_weighted.py --device cuda:0
```

### Run 7B Validation (E28)

```bash
# Requires 3 GPUs (5,6,7) with ~24GB each
python scripts/e28_7b_full_fl.py
```

### View Results

```bash
# All experiments save results to outputs/{experiment_name}/metrics.json
cat outputs/e28_7b_full_fl/final_comparison.json
```

---

## Architecture

### Design Principles

1. **Theory-Driven:** Every experiment validates a theorem or tests a prediction
2. **Minimal Dependencies:** Only conda env `realm` (torch, transformers, peft, datasets)
3. **Explicit Over Config:** Python scripts with inline parameters — research code needs transparency
4. **Full Traceability:** Every experiment outputs `metrics.json` with complete hyperparameters

### Key Components

#### `src/aggregation.py`

The only aggregation primitive used across all experiments:

```python
def _weighted_average_state_dicts(state_dicts, weights):
    """Weighted average of LoRA parameters."""
```

**No strategy pattern.** Theorem 4 proves any linear parameter-space aggregation collapses to weighted averaging, so we only need this one function.

#### `src/data_loader.py`

```python
def load_agentgym_data(env_name, max_samples=None) -> list[dict]:
    """Load raw trajectories from data/raw/{env}_train.json"""

def get_trainable_messages(trajectory) -> tuple[list[dict], float]:
    """Extract trainable turns and success indicator"""
```

#### Experiment Pattern

Each script is self-contained with inline parameters:

```python
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TASKS = ["babyai", "webshop", "textcraft"]
N_SAMPLES = 64
N_ROUNDS = 5

for method in ["uniform", "loss_proportional"]:
    for round in range(N_ROUNDS):
        # Train → Aggregate → Evaluate
        pass

# Save results
with open(output_dir / "metrics.json", "w") as f:
    json.dump(results, f, indent=2)
```

**Why explicit parameters?** Research code changes frequently. Each experiment is unique. Inline parameters make scripts self-documenting.

### Data Flow

```
AgentGym Trajectories
        ↓
data/raw/{env}_train.json
        ↓
load_agentgym_data() → get_trainable_messages()
        ↓
SFTTrainer (per-task training)
        ↓
LoRA state dict extraction
        ↓
_weighted_average_state_dicts() (uniform or loss-proportional)
        ↓
Global model update → Evaluate → metrics.json
```

---

## Documentation

This project uses **documentation as the primary research output**.

**Reading order:**
1. `docs/theory.md` — Mathematical foundation (8 theorems)
2. `docs/experiments.md` — Experimental validation
3. `docs/failures.md` — What didn't work and why
4. `docs/novelty_assessment.md` — Insight depth analysis
5. `docs/research_plan.md` — Roadmap and next steps

**Workflow:**
1. **Hypothesis** → Write in `docs/theory.md`
2. **Proof** → Derive in `docs/theory.md`
3. **Experiment** → Implement in `scripts/e{XX}_*.py`
4. **Results** → Auto-save to `outputs/e{XX}/metrics.json`
5. **Analysis** → Write in `docs/experiments.md`

---

## Target

**AAAI 2027** — "AdaTem-FL: Adaptive Temperature Aggregation with Minimax Guarantees for Heterogeneous Task Federated Learning"

**Current status:** ★★★★☆ (8 theorems + 28 experiments)
**To reach ★★★★★:** Theorem 9 (non-convex) + 14B validation

---

## Citation

If you use this work, please cite:

```bibtex
@article{adatem2026,
  title={AdaTem-FL: Adaptive Temperature Aggregation with Minimax Guarantees for Heterogeneous Task Federated Learning},
  year={2026},
  note={In preparation for AAAI 2027}
}
```

---

*Last updated: E28 complete, Three Regimes framework corrected, 8 theorems established*
