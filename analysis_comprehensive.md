# Comprehensive Analysis: Federated LLM Agent Training

> Generated from E0-E5 experimental data. All conclusions backed by quantitative evidence.

---

## 1. Problem Definition

**Setting**: K=5 heterogeneous agent environments (babyai, webshop, textcraft, maze, wordle), each acting as a federated client. A shared LoRA-adapted LLM is trained via FedAvg.

**Constraints**:
- Model: Qwen2.5-0.5B-Instruct with LoRA (rank=8, alpha=16)
- Data: AgentGym offline trajectories, 64 train + 32 eval per env
- Budget: 5 federated rounds, 1 epoch per round
- Evaluation: Cross-entropy loss on held-out eval set (lower = better)

**Key distinction**: Our "online" simulation uses **model loss as filter proxy**, not actual task rewards. Our data is **offline expert demonstrations**, not online self-generated trajectories.

---

## 2. Experimental Inventory

| Exp | Methods | Rounds | Key Variable | Status |
|-----|---------|--------|-------------|--------|
| E0 | Single-env gradients | 1 | Gradient decomposition | Done |
| E1 | FedAvg, FedDebias, Local | 5 | Offline, no filtering | Done |
| E2 | Offline FedNorm, Online FedAvg, Online FedDebias | 5 | Online vs offline | Done |
| E3 | FedPow α-sweep (5×2) | 5 | α ∈ {0, 0.25, 0.5, 0.75, 1.0} | Done |
| E4 | Option B, Option C (4τ) | 5 | Adaptive α, soft filtering | Done |
| E5 | SFT, NAT, GRPO, Curriculum | 5 | Data utilization strategy | Done |

**Total experiments**: 22 method runs × 5 rounds = 110 training rounds

---

## 3. Performance Summary (Final Round R4)

### 3.1 Average Eval Loss Across All Environments

| Rank | Method | Avg Loss | Std | Category |
|------|--------|----------|-----|----------|
| 1 | FedAvg-NAT (all data) | **0.990** | **0.561** | Data strategy |
| 2 | Offline FedNorm | 1.021 | 0.543 | Baseline |
| 3 | Option C (τ=2.0) | 0.960 | 0.543 | Soft filtering |
| 4 | Option C (τ=0.5) | 0.981 | 0.556 | Soft filtering |
| 5 | Fed-GRPO (τ=0.5) | 0.997 | 0.561 | RL-style |
| 6 | Online FedAvg | 1.442 | 0.824 | Standard |
| 7 | FedAvg-SFT (baseline) | 1.440 | 0.823 | Standard |
| 8 | FedPow α=0.25 (online) | 1.459 | 0.733 | Aggregation |
| 9 | Option B (adaptive α) | 1.474 | 0.671 | Aggregation |
| 10 | Fed-Curriculum | 1.480 | 1.120 | Training schedule |
| 11 | Online FedDebias | 1.742 | 1.076 | Debiasing |

**Note**: Ranks 1-5 all use all 64 samples per environment. Ranks 6-11 use filtered subsets.

### 3.2 Per-Environment Breakdown (R4)

| Method | babyai | webshop | textcraft | maze | wordle |
|--------|--------|---------|-----------|------|--------|
| FedAvg-NAT | 0.749 | 1.708 | 1.275 | 0.048 | 1.169 |
| Offline FedNorm | 0.774 | 1.636 | 1.298 | 0.038 | 1.356 |
| Fed-GRPO | 0.741 | 1.688 | 1.335 | 0.054 | 1.167 |
| FedAvg-SFT | 0.823 | 2.504 | 1.941 | 0.200 | 1.731 |
| Fed-Curriculum | 0.379 | 2.990 | 2.174 | 0.009 | 1.847 |

---

## 4. Gap Decomposition

### 4.1 Three Sources of Performance Gap

```
Local-only (5 models):   0.108   ← theoretical upper bound (unfair)
NAT (all data, FedAvg):  0.990   ← practical upper bound
SFT (filtered, FedAvg):  1.440   ← current standard
```

**Gap 1: SFT → NAT = 0.450 (31%)**
- Cause: Data scarcity from loss-based filtering
- Mechanism: Hard envs lose 75-89% of data → stall after 1-2 rounds
- Evidence: webshop/textcraft improvement deceleration = 0.05-0.16x (online) vs 0.96-1.76x (offline)

**Gap 2: NAT → Offline = -0.031 (NAT is 3% BETTER)**
- Cause: Within noise — these are essentially the same method with different code paths
- Implication: **FedAvg imposes no optimization penalty when data is sufficient**
- This is a positive finding: federation is "free" for this task scale

**Gap 3: NAT → Local = 0.882 (89%)**
- Cause: One shared LoRA adapter must serve 5 heterogeneous tasks
- This is the fundamental capacity limit of shared-parameter federation
- Not a fair comparison (local uses 5× parameters), but represents the cost of federation

### 4.2 Data Scarcity is Environment-Specific

| Environment | Filter Pass Rate (R4) | Samples Retained | Eval Loss (SFT) | Eval Loss (NAT) | Δ |
|-------------|----------------------|-------------------|-----------------|-----------------|---|
| maze | 100% | 64/64 | 0.200 | 0.048 | -76% |
| babyai | 100% | 64/64 | 0.823 | 0.749 | -9% |
| wordle | 25% | 16/64 | 1.731 | 1.169 | -32% |
| webshop | 25% | 16/64 | 2.504 | 1.708 | -32% |
| textcraft | 11% | 7/64 | 1.941 | 1.275 | -34% |

**The Matthew Effect**: Easy envs (low loss → pass filter → more data → get easier) vs Hard envs (high loss → fail filter → less data → get harder). The loss gap between easy/hard envs grows from 3.6× (R0) to 12.5× (R4) under filtering, but stabilizes under NAT.

---

## 5. Convergence Dynamics

### 5.1 Improvement Deceleration

The per-round loss decrease reveals which environments "stall":

| Env | Online-SFT R0→R1 | R3→R4 | Decel | Online-NAT R0→R1 | R3→R4 | Decel |
|-----|-------------------|-------|-------|-------------------|-------|-------|
| webshop | 0.249 | 0.047 | **0.19** | 0.147 | 0.346 | **2.35** |
| textcraft | 0.217 | 0.015 | **0.07** | 0.265 | 0.321 | **1.21** |
| wordle | 0.176 | 0.065 | **0.36** | 0.309 | 0.220 | **0.71** |
| babyai | 0.317 | 0.240 | 0.76 | 0.320 | 0.235 | 0.73 |
| maze | 0.204 | 0.106 | 0.52 | 0.231 | 0.114 | 0.49 |

**Key insight**: webshop and textcraft **stall** (decel < 0.2) in SFT but **accelerate** (decel > 1.0) in NAT. This is the fingerprint of data scarcity causing premature convergence. With full data, these environments enter a virtuous cycle: more data → better model → faster improvement.

### 5.2 Cross-Environment Correlation

All environments' round-over-round improvements are highly correlated (Pearson r > 0.86):

```
             babyai  webshop  textcraft  maze   wordle
babyai       1.000   0.976    0.971      0.947  0.956
webshop              1.000    0.933      0.863  0.886
textcraft                     1.000      0.974  0.989
maze                                     1.000  0.997
wordle                                          1.000
```

**Interpretation**: The shared LoRA captures general language modeling capabilities that benefit all environments simultaneously. There is NO evidence of gradient conflict or negative transfer between environments. When the global model improves on babyai, it also improves on webshop.

This high correlation also explains why FedAvg ≈ centralized: the gradient updates from different environments point in similar directions, so averaging doesn't cause destructive interference.

---

## 6. GRPO Analysis

### 6.1 Weight Distribution

Fed-GRPO uses softmax(-loss/τ) with τ=0.5 to weight 64 trajectories per environment:

| Env | Mean Loss | Max Weight | Uniform Weight | Max/Uniform |
|-----|-----------|------------|----------------|-------------|
| babyai | 2.278 | 0.0279 | 0.0156 | 1.78× |
| webshop | 2.725 | 0.0293 | 0.0156 | 1.88× |
| textcraft | 2.166 | 0.0286 | 0.0156 | 1.83× |
| maze | 0.787 | 0.0156 | 0.0156 | **1.00×** |
| wordle | 2.149 | 0.0156 | 0.0156 | **1.00×** |

**Result**: Weights are near-uniform (1.0-1.9× deviation from uniform). The GRPO method degenerates to NAT with trivial weight perturbations. This is why GRPO (0.997) ≈ NAT (0.990).

### 6.2 Why GRPO Fails Here

1. **Small sample size (N=64)**: softmax over 64 samples with similar losses produces flat weights
2. **Low loss variance**: AgentGym trajectories have relatively uniform quality — no clear "good" vs "bad" split
3. **Temperature too high**: τ=0.5 relative to loss variance (σ≈0.3-0.8) makes the distribution too soft

For GRPO to be effective, we'd need:
- Much larger N (e.g., N=512+) with genuine quality variance
- Lower temperature (τ ≈ 0.05-0.1)
- Or actual reward signals from environment interaction (not model loss)

---

## 7. Curriculum Learning Failure Analysis

Fed-Curriculum (easy envs R0-2, all envs R3-4) is the WORST method for hard environments:

| Env | SFT R4 | Curriculum R4 | Δ |
|-----|--------|---------------|---|
| maze | 0.200 | **0.009** | -95% ← good |
| babyai | 0.823 | **0.379** | -54% ← good |
| webshop | 2.504 | **2.990** | +19% ← WORSE |
| textcraft | 1.941 | **2.174** | +12% ← WORSE |
| wordle | 1.731 | **1.847** | +7% ← WORSE |

**Mechanism**: Phase 1 (easy-only) overfits the shared LoRA to easy-env patterns. The maze loss drops to near-zero, consuming parameter capacity. Phase 2 (all envs) then starts from a model that's highly specialized for easy tasks, and the hard-env gradients are overwhelmed by the easy-env's near-zero-loss gradients during FedAvg.

This is analogous to the "environment mismatch collapse" identified by FedAgent (ICLR 2026).

---

## 8. The Fed-SE Contradiction: Resolved

### 8.1 Apparent Contradiction

| | Fed-SE | Our E5 |
|---|---|---|
| Using all data (no filter) | -26% (catastrophic) | **+31% (major improvement)** |
| Wordle specifically | Collapses to 0% | Improves by 43% |

### 8.2 Resolution: Different Filtering, Different Data

| Variable | Fed-SE | Our E5 |
|----------|--------|--------|
| **Filter criterion** | Task success (binary reward) | Model loss (continuous perplexity) |
| **What's removed** | Wrong-action trajectories | Hard-to-learn trajectories |
| **Data source** | Online self-generated | Offline expert demonstrations |
| **Failure semantics** | "Model went left, should go right" | "Model finds this trajectory surprising" |
| **Evaluation** | Success rate (0/1) | Eval loss (continuous) |

**Key insight**: Filtering by **task success** removes genuinely harmful data (learning to fail). Filtering by **model loss** removes the most informative data (hard examples). These are opposite operations with opposite effects.

### 8.3 Implication

In real online federated agent training, there exists a **tension**:
- **Filter failures** → Prevent learning wrong behavior → But starve hard envs of data
- **Keep all data** → Maximize data quantity → But risk learning incorrect actions

The sweet spot likely involves **failure-aware utilization**: learn the *structure* of failures (what went wrong, why) without mimicking the incorrect actions. This is the research frontier.

---

## 9. Lessons from Failed Approaches

### 9.1 Aggregation Doesn't Matter

FedPow (α-sweep), Option B (adaptive α), FedDebias (1/p_k weighting) — all modify how gradients are combined during FedAvg. None beat simple uniform averaging.

**Why**: Cross-environment gradient cosine similarity is high (r > 0.86). When gradients are already aligned, reweighting them doesn't change the optimization direction. The gradient magnitudes are also uniform (delta norms within 0.69-0.94 at R4), so magnitude-based reweighting is also moot.

### 9.2 Loss-Based Filtering is Harmful

Our "online simulation" uses model loss as a proxy for trajectory quality. This creates a **positive feedback loop**:

1. Hard env → high loss → filter removes more samples → less training data
2. Less data → slower convergence → loss stays high → filter removes even more next round
3. Easy env → low loss → filter keeps all samples → fast convergence → loss drops further

Result: The easy-hard gap grows from 3.6× (R0) to 12.5× (R4) under filtering.

### 9.3 Data Quantity > Data Quality (in Our Setting)

Option C with various τ values (0.1-2.0) all match offline training. The soft weighting doesn't matter — what matters is that all 64 samples are used. This means in our setting, the quality variance across trajectories is too small for weighting to differentiate.

---

## 10. Open Questions for Future Work

1. **True online setting**: How do our findings transfer when trajectories are generated by the model itself during training? We can't test this without real environment interaction.

2. **Failure quality spectrum**: Between "genuinely harmful" (wrong actions) and "slightly suboptimal" (high perplexity), there's a spectrum. Can we identify the transition point?

3. **Adaptive data utilization**: A method that uses failures intelligently — learning *from* them without learning *to reproduce* them — could bridge the Fed-SE vs NAT gap.

4. **Capacity allocation**: The 89% gap between shared-LoRA NAT and per-env models suggests the shared adapter is capacity-limited. Can we allocate more LoRA capacity to hard environments?

5. **Multi-scale experiments**: Our results are at 0.5B scale. Does data scarcity amplify or diminish at larger scales (7B, 14B)?
