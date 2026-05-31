# Normalize, Don't Debias: Gradient Power-Law Normalization for Federated Agent SFT

## 一句话概括

> FedAvg on agent SFT applies implicit $1/p_k$ importance weights — this is an adaptive learning rate, not a bias; the optimal aggregation uses gradient power-law normalization $\hat{g}(\alpha) = \frac{\sum_k \|g_k\|^{-\alpha} g_k}{\sum_k \|g_k\|^{-\alpha}}$ to balance bias and variance.

---

## 1 核心理论

### 1.1 事实：$g^{SFT} = \frac{1}{p}\nabla J$（3行证明，无假设）

$$\nabla J_k = \mathbb{E}_\tau[R_k \psi] = p_k \cdot \mathbb{E}_{\pi^+}[\psi] = p_k \cdot g_k^{SFT}$$

### 1.2 FedAvg = Importance Sampling

$$\hat{g}_{\text{FedAvg}} = \frac{1}{K}\sum_k g_k^{SFT} = \frac{1}{K}\sum_k \frac{1}{p_k}\nabla J_k$$

隐含权重 $w_k = 1/p_k$：低成功率环境的策略梯度被放大。

### 1.3 关键洞察：$1/p_k$是自适应学习率，不是偏差

- 低$p_k$ → 更少成功轨迹（数据稀缺）但每个轨迹梯度被$1/p_k$放大（梯度增强）
- 数据稀缺 × 梯度增强 ≈ 互相补偿 → FedAvg意外实现了平衡
- **移除放大（FedDebias）= 去掉自适应学习率 → 难环境双重惩罚（无数据+无梯度）**

### 1.4 Bias-Variance分解

$$\text{MSE}(\hat{g}, \bar{g}) = \underbrace{\|\text{Bias}\|^2}_{\propto (1-p_k)^2/p_k^2} + \underbrace{\text{Var}}_{\propto 1/(p_k |B_k|)}$$

- FedAvg ($\alpha=0$): 保留完整自适应率，高bias低variance
- FedDebias ($\alpha=1$): 零bias但高variance，实验证明有害
- **最优 $\alpha^*$**: 最小化MSE

---

## 2 方法：FedPow

### 核心公式

已知$p_k$: $\hat{g}(\alpha) = \frac{\sum_k p_k^\alpha \cdot g_k^{SFT}}{\sum_k p_k^\alpha}$

未知$p_k$时（$p_k \propto 1/\|g_k\|$）: $\hat{g}(\alpha) = \frac{\sum_k \|g_k\|^{-\alpha} \cdot g_k}{\sum_k \|g_k\|^{-\alpha}}$

### $\alpha$的含义

| $\alpha$ | 等价于 | 性质 |
|---|---|---|
| 0 | FedAvg | 保留完整$1/p_k$自适应学习率 |
| 0.5 | FedPow-0.5 | **理论预测的最优平衡点** |
| 1.0 | FedNorm | 完全归一化 |

### 算法

```
FedPow
for t = 1 to T:
    每个 client k: 训练 → delta_k, 上传 (delta_k, ||delta_k||)
    Server: w_k = ||delta_k||^{-alpha}, agg = sum(w_k * delta_k) / sum(w_k)
    广播 agg
```

---

## 3 已完成实验

| 实验 | 方法 | 结果 |
|---|---|---|
| E0 | 梯度探针 | 梯度3x异质, H1推翻 |
| E1 offline | FedAvg/FedDebias/Local | FedDebias(fixed $p_k$)过修正, avg=0.98 vs FedAvg=0.62 |
| E2 online | Online FedAvg/FedDebias | FedDebias灾难: webshop 2.90→3.46↑ |
| E2 offline | FedNorm | **最优联邦方法**: avg=1.02, std=0.57 |

---

## 4 E3: $\alpha$-Sweep实验（执行中）

### 假设

存在最优 $\alpha^* \in (0, 1)$ 使得 FedPow($\alpha^*$) > FedAvg ($\alpha=0$) 且 FedPow($\alpha^*$) > FedNorm ($\alpha=1$)

### 设置

- 模型：Qwen2.5-0.5B + LoRA rank=8
- $\alpha \in \{0, 0.25, 0.5, 0.75, 1.0\}$
- Offline设定（全部数据，无成功过滤）
- 5 rounds, 64 train + 32 eval per env
- 指标：avg eval loss, std, per-env loss

---

## 5 创新点

1. **理论发现**：$g_k^{SFT} = \frac{1}{p_k}\nabla J_k$ → FedAvg隐含importance sampling
2. **反直觉发现**：$1/p_k$不是偏差而是自适应学习率（FedDebias失败证明）
3. **框架**：Bias-Variance tradeoff统一解释FedAvg/Debias/Norm
4. **方法**：FedPow — 一个参数$\alpha$遍历整个tradeoff
