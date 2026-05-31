# 研究笔记归档

> 汇总所有经分析仍有效的数学事实、已推翻的结论、和关键数据。

---

## 经验证的数学事实

### 1. SFT梯度与策略梯度的精确关系

Binary reward $R_k(\tau) \in \{0,1\}$，on-policy设定下：

$$g_k^{SFT} = \frac{1}{p_k}\nabla J_k$$

证明（3行）：
$\nabla J_k = \mathbb{E}_\tau[R_k \psi] = p_k \cdot \mathbb{E}_{\pi^+}[\psi] + (1-p_k) \cdot 0 = p_k \cdot g_k^{SFT}$

**前提**：on-policy（每轮内行为策略 = 被优化的策略）。在联邦Agent SFT中，每轮client收到全局模型后交互并更新同一个模型，此前提成立。

### 2. FedAvg+SFT的均值-偏差分解

定义：$\bar{g} = \frac{1}{K}\sum_k \nabla J_k$，$\Delta_k = \nabla J_k - \bar{g}$，$\beta_k = (1-p_k)/p_k$。

$$\hat{g}_{\text{FedAvg}} = (1 + \bar{\beta})\bar{g} + \frac{1}{K}\sum_k \beta_k \Delta_k$$

其中 $\sum_k \Delta_k = 0$。

- 全局分量 $(1+\bar{\beta})\bar{g}$：良性，只是学习率缩放。需注意 $\bar{\beta}=4.53$ 导致5.53倍步长放大。
- 偏差混合 $\frac{1}{K}\sum_k \beta_k \Delta_k$：仅因 $p_k$ 异质性而存在。

**FedDebias修正**：$\hat{g}_{\text{FedDebias}} = \frac{1}{K}\sum_k p_k g_k^{SFT} = \bar{g}$

修正后：偏差混合=0，步长放大=0。等价于均匀策略梯度聚合。

### 3. 偏差混合的量级

$$\frac{\|\text{偏差混合}\|}{\|\text{全局分量}\|} \leq \frac{\sqrt{\overline{\beta^2}}}{1+\bar{\beta}} \cdot \frac{\sqrt{\overline{\|\Delta\|^2}}}{\|\bar{g}\|}$$

- 因素1（$\sqrt{\overline{\beta^2}}/(1+\bar{\beta}) = 1.55$）：纯由 $p_k$ 异质性决定
- 因素2（$\sqrt{\overline{\|\Delta\|^2}}/\|\bar{g}\|$）：环境梯度分散度

### 4. SFT梯度方差

$$\text{Var}(\hat{g}_k^{SFT}) \approx \frac{1}{p_k |B_k|}\text{Var}_{\pi^+}(\psi)$$

低 $p_k$ 环境既有更大偏差又有更大方差——双重打击。

### 5. Fed-SE的 $p_k$ 数值

| 环境 | $p_k$ | $\beta_k$ | $\beta_k^2$ | 隐含权重 $1/p_k$ |
|------|-------|-----------|------------|-----------------|
| BabyAI | 0.83 | 0.20 | 0.04 | 1.20 |
| WebShop | 0.73 | 0.37 | 0.14 | 1.37 |
| TextCraft | 0.67 | 0.49 | 0.24 | 1.49 |
| Maze | 0.28 | 2.57 | 6.61 | 3.57 |
| Wordle | 0.05 | 19.0 | 361.0 | 20.0 |

$\bar{\beta} = 4.53$，$\sqrt{\overline{\beta^2}} = 8.58$

---

## E0实验数据（Qwen2.5-1.5B, LoRA rank=8, 16 acc steps）

### SFT梯度范数
| 环境 | SFT grad norm | Policy grad norm | Ratio ($1/p_k$ effect) |
|------|--------------|-----------------|----------------------|
| BabyAI | 3.48 | 2.89 | 1.2x |
| WebShop | 2.55 | 1.86 | 1.4x |
| TextCraft | 2.39 | 1.60 | 1.5x |
| Maze | 1.14 | 0.32 | 3.6x |
| Wordle | 1.15 | 0.06 | 19.2x |

### 跨环境梯度余弦相似度（全梯度）
| | babyai | webshop | textcraft | maze | wordle |
|---|---|---|---|---|---|
| babyai | 1.00 | 0.50 | 0.49 | 0.36 | 0.36 |
| webshop | | 1.00 | 0.54 | 0.44 | 0.46 |
| textcraft | | | 1.00 | 0.43 | 0.47 |
| maze | | | | 1.00 | 0.59 |
| wordle | | | | | 1.00 |

### H1测试结果
- Attn deviation² = 1.416, MLP deviation² = 1.339
- Ratio = 1.057（≈1.0）
- **H1不成立**：Attention和MLP偏差幅度几乎相同

---

## 已推翻的结论（历史教训）

1. **H1（$\|\Delta^{Attn}\| \ll \|\Delta^{MLP}\|$）被推翻**：E0实验证明两者偏差相当（ratio≈1.06）
2. **v3收益条件 $\mathcal{G}=G_{var}-C_{dir}-L_{bias}$ 是同义反复**
3. **v3的phase transition依赖人为参数化**
4. **v4偏差公式错误**
5. **"$1/p$只放大MLP"是错的**：$1/p$放大全部梯度
6. **旧版P1（余弦相似度）≠ H1**：方向一致不代表偏差幅度小

---

## FedRNK → FedDebias 方向变更记录

**2026-05-31**: H1被E0实验推翻后，从"按模块分离"（FedRNK）转向"成功率去偏"（FedDebias）。

**变更原因**：
1. E0实验显示 Attn/MLP 偏差相当（ratio=1.06），无法按模块分离
2. 但 E0 同时验证了 $1/p_k$ 放大效应（19x for wordle），理论核心仍然成立
3. FedDebias 直接修正 $1/p_k$ 偏差，不需要 H1 假设

**保留的核心**：
- §1.1 的 $g^{SFT} = \frac{1}{p}\nabla J$ 证明
- §1.3 的均值-偏差分解
- §1.4 的量级分析
- E0 的梯度异质性数据

**放弃的核心**：
- H1 假设（Attn偏差<MLP偏差）
- FedRNK 算法（只聚合 Attention）
- 按模块分离的整体思路

---

## FedDebias → FedPow 方向变更记录

**2026-05-31**: E1+E2实验证明FedDebias有害后，转向FedPow（梯度范数幂次归一化）。

**变更原因**：
1. E1 offline: FedDebias(fixed p_k) avg=0.979 vs FedAvg avg=0.625 → 去偏反而变差
2. E2 online: FedDebias(dynamic p_k) avg=1.742 vs FedAvg avg=1.442 → 动态去偏更差
3. 关键发现：$1/p_k$不是偏差而是隐式自适应学习率

**保留的核心**：
- §1.1 的 $g^{SFT} = \frac{1}{p}\nabla J$ 证明（仍成立）
- FedAvg隐含importance sampling的分析（仍成立）
- E0 的梯度异质性数据

**放弃的核心**：
- FedDebias 算法（p_k加权修正）
- "$1/p_k$是有害偏差"的假设
- "修正偏差→更好性能"的直觉

**E3 FedPow alpha-sweep实验结果**：

Online设定（有$1/p_k$效应）：
| α | Avg Loss | Std |
|---|---|---|
| 0.00 (FedAvg) | **1.4392** | 0.8183 |
| 0.25 | 1.4589 | 0.7328 |
| 0.50 | 1.4598 | 0.6858 |
| 0.75 | 1.4845 | 0.6535 |
| 1.00 (FedNorm) | 1.5109 | **0.6105** |

关键发现：单调tradeoff — α↑ avg_loss↑ std↓，无内部最优点。

Offline设定（无$1/p_k$效应）：所有α性能几乎相同。
