# FedRNK 参考文献

> 与当前方案相关的文献，按类别整理。旧版FITAL框架的对比分析已移除。

---

## 1 联邦LLM Agent训练（最直接相关）

[1] **Fed-SE** — Z. Zhang et al., "Fed-SE: Federated Self-Evolution for Cross-Environment Knowledge Transfer in Privacy-Constrained LLM Agents," arXiv:2512.08870, Dec. 2025. [Code](https://github.com/Soever/Federated-Agents-Evolution)
- 5环境 (BabyAI, WebShop, TextCraft, Maze, Wordle) × 5模型
- 成功过滤 + LoRA(rank=8) SFT + 无权重平均
- 关键消融：移除成功过滤→崩溃(-26%)；无权重平均优于加权平均；Maze受益最大

[2] **FedAgent** — C. Chen et al., "Federated Agent Reinforcement Learning," ICLR 2026 under review. [Project](https://fed-agent.github.io/)
- GRPO + FedAvg聚合
- σ²-Dominance理论

[3] **FedIT** — Zhang et al., "Building Federated GPT: Federated Instruction Tuning," 2024.
- 标准FedAvg + LoRA基线

## 2 Transformer机制可解释性（H1的物理基础）

[4] **Geva et al.** — "Transformer Feed-Forward Layers Are Key-Value Memories," EMNLP 2021.
- MLP层 = 键值记忆，存储事实知识
- 每个MLP神经元编码特定的语义关系

[5] **arXiv:2403.19521** — "How do Language Models Bind Entities in Context?"
- Concept Depth: 浅层编码简单概念，深层编码复杂推理

[6] **arXiv:2410.20008** — "Language Models Resist Alignment: Evidence From Data Compression"
- Layer-by-Layer: 浅层共享语义/中层转换/深层精化

[7] **OpenReview 2025** — "Recall vs Reasoning"
- 浅层负责recall，深层负责reasoning
- Attention在不同层有不同功能

[8] **arXiv:2404.07066** — "Attention Heads of Large Language Models: A Survey"
- Attention层 = 信息路由和上下文整合

## 3 联邦LoRA方法（对比基线）

[9] **FedTreeLoRA** — 按层深度分层共享（浅共享、深隔离）
- 与我们正交：他们按层分，我们按模块（Attn/MLP）分

[10] **FedSA-LoRA** — 只共享A矩阵
- 区分维度：LoRA的A vs B，缺乏语义基础

[11] **ALoRA** — 共享B矩阵
- 同上

[12] **LoRA-PAR** — 按训练阶段分区（SFT vs RL）
- 我们按模型组件分区，不依赖训练阶段

[13] **FlexLoRA** — 动态rank调整
[14] **FedLEASE** — 稀疏更新聚合

## 4 SFT与RL的理论连接

[15] **iw-SFT** — T. Qin & J. Springenberg, arXiv:2507.12856, 2025.
- 证明SFT优化RL目标的下界
- 重要性加权SFT

[16] **DFT** — Y. Wu et al., arXiv:2508.05629, 2025.
- SFT梯度 = 带逆概率加权的策略梯度

[17] **ASFT** — arXiv:2509.23753, 2025.
- 锚定参考分布解决分布漂移

## 5 联邦RL理论（背景）

[18] **FedSARSA** — Z. Zhang et al., arXiv:2401.15273, 2024.
- 首个on-policy异构FRL有限时间分析

[19] **QAvg/PAvg** — C. Jin et al., AISTATS 2022.
- Q值/策略平均，收敛到次优解

[20] **PFedRL-Rep** — G. Xiong et al., ICLR 2025.
- 共享表征 + 个性化头，线性加速

[21] **FedPG-BR** — Z. Fan et al., NeurIPS 2021.
- 方差缩减 + Byzantine鲁棒

[22] **AFedPG** — Y. Lan et al., arXiv:2404.08003, 2024.
- 异步联邦策略梯度

[23] **FedSVRPG-M** — arXiv:2405.19499, 2024.
- 动量消除异质性影响

[24] **b-RS-FedPG** — Labbi et al., HAL-05155198, 2025.
- 异构环境下的全局收敛

[25] **FedHPD** — W. Jiang et al., AAMAS 2025.
- KL散度对齐全局共识

[26] **FedNPG** — Yang et al., NeurIPS 2023.
- 自然策略梯度的联邦扩展

## 6 联邦学习基础

[27] **FedAvg** — B. McMahan et al., AISTATS 2017.
[28] **FedProx** — T. Li et al., MLSys 2020.
[29] **SCAFFOLD** — Karimireddy et al., ICML 2020.
[30] **FedNova** — J. Wang et al., NeurIPS 2020.
[31] **FedLAW** — Z. Li et al., ICML 2023.
[32] **Aggregation-Heterogeneity Trade-off** — R. Zhao et al., COLT 2023.

## 7 LLM Agent自主学习

[33] **RAGEN/StarPO** — 2025. Star Policy Optimization
[34] **EvolveR** — arXiv:2510.16079, 2025. 自我蒸馏 + 经验驱动
