# 研究计划：AdaTem-FL

## 1. 核心问题

**一句话概括：** 在异构任务联邦学习（Heterogeneous Task FL）中，为什么简单的loss-proportional aggregation比复杂的参数空间操作更有效？

### 1.1 背景

联邦学习（FL）中，不同任务（如翻译、问答、摘要）共享同一模型。由于任务难度差异，简单平均聚合（FedAvg）导致**梯度稀释**——易任务梯度淹没难任务信号。

### 1.2 我们的发现

Loss-proportional aggregation（按损失比例加权）：
$$w_k = \frac{L_k}{\sum_j L_j}$$

在连续15+个实验中，它是**唯一**持续有效的方法。所有复杂改进（层自适应、课程学习、原型对齐、顺序组合LoRA）均失败。

## 2. 已完成工作

### 2.1 理论框架（7个定理）

| 定理 | 内容 | 状态 |
|------|------|------|
| **Theorem 1** | Loss-proportional ≈ Softmax目标函数的梯度下降 | 证明完成 |
| **Theorem 2** | Softmax → Minimax收敛 | 证明完成 |
| **Theorem 3** | Rate Equalization：所有任务按1/t速率收敛 | 证明完成 |
| **Theorem 4** | 线性方法必然坍塌（LAFA数学解释） | 证明完成 |
| **Theorem 5** | 固定权重噪声地板下界 | 证明完成 |
| **Theorem 6** | 结构性冲突下界（固定权重极限） | 证明完成 |
| **Theorem 7** | Loss-proportional支配固定权重 | 证明完成 |

### 2.2 实验验证（28个实验）

#### 核心验证实验

| 实验 | 配置 | 关键发现 |
|------|------|----------|
| **E8** | 0.5B, 5轮FL, 5任务 | Loss-proportional提升**20.8%** |
| **E19** | 规模扩大 (rank×2, samples×2) | 优势从13%→**20.2%**，随规模增大 |
| **E20** | 自适应温度λ | λ≈2最优，adaptive自动收敛至此 |
| **E21** | 泛化测试 | Loss-wt绝对test loss更低(0.383 vs 0.449) |
| **E23a** | Theorem 6合成验证 | 固定权重hit structural conflict floor (1.995) |
| **E23b** | Theorem 7合成验证 | Loss-proportional消除noise floor，**66.2%**提升 |
| **E28** | 7B, 5轮FL, 3任务 | Loss-proportional提升**43.6%**（重大发现！） |

#### 失败实验（明确排除的方向）

| 实验 | 方法 | 结果 | 失败原因 |
|------|------|------|----------|
| **E14** | LAFA（层自适应α） | α→1（坍塌） | 线性操作等价于全局重加权（Theorem 4） |
| **E15** | Curriculum（课程学习） | +27.9%更差 | 共享参数导致灾难性遗忘 |
| **E16** | Prototype Alignment | ~0%效果 | SFT已隐式对齐，无额外信号 |
| **E17** | SeqComp-LoRA（顺序组合） | +64%更差 | LoRA merge非组合性，破坏性干扰 |
| **E18** | Adaptive Epochs | +2%更差 | 难任务多epoch→更多噪声积累 |
| **E24/25** | 单轮7B验证 | -0.5%~-2.3% | **无迭代=无优势**（关键发现） |
| **E26** | 单轮0.5B验证 | -0.4% | 确认单次训练无优势 |

## 3. 核心Insight

### 3.1 三阶段框架（Three Regimes）

$$\text{Advantage} \propto \underbrace{(1 - \text{Capacity}/\text{Task Complexity})}_{\text{结构性冲突}} \times \underbrace{\text{Number of Rounds}}_{\text{迭代适应}}$$

| 阶段 | 条件 | Loss-Proportional效果 | 代表实验 |
|------|------|----------------------|----------|
| **阶段1** | 小模型 + 多轮FL | **强优势** (20.8%) | E8 |
| **阶段2** | 大模型 + 多轮FL | **更强优势** (43.6%) | E28 |
| **阶段3** | 单次训练（无迭代） | **无差异** | E24/25/26 |

**关键发现：** 
- Loss-proportional的优势**随模型规模增大而增大**（0.5B→20.8%, 7B→43.6%）
- **迭代是必要的**：单轮训练无论模型大小均无优势
- 这不是容量问题，而是**迭代优化问题**

### 3.2 为什么简单方法最优

1. **Softmax-Minimax连接**（Theorem 1-2）：Loss-proportional近似最小化worst-task loss，而非average loss
2. **自动课程**（Theorem 7 Corollary）：易任务先收敛，权重自动转移至难任务
3. **噪声地板消除**（Theorem 5）：收敛任务的权重→0，其噪声不再干扰
4. **线性坍塌**（Theorem 4）：任何线性参数空间操作等价于全局重加权

## 4. 研究计划

### 4.1 短期目标（1-2周）

1. **E29: 1.5B模型完整5轮FL**
   - 使用E28的单模型串行方式（避免多进程问题）
   - 验证优势是否介于0.5B(20%)和7B(43%)之间
   - 预期：~30%提升

2. **E30: 14B模型验证**
   - 使用4-bit量化（bitsandbytes）加载14B模型
   - 验证优势是否继续增大
   - 预期：~50%+提升

3. **Theorem 8: 收敛率下界**
   - 证明在多轮FL中，loss-proportional的worst-case收敛率优于任何固定权重
   - 目标：给出显式收敛率并证明最优性

### 4.2 中期目标（2-4周）

4. **E31: 真实FL场景模拟**
   - 非IID数据分布（不同客户端有不同任务组合）
   - 通信压缩（仅传输LoRA参数）
   - 部分参与（每轮随机选择客户端）

5. **E32: 跨领域验证**
   - 从AgentGym扩展到其他领域（代码生成、数学推理）
   - 验证generalizability

6. **Theorem 9: 非凸扩展**
   - 将定理从强凸假设扩展到非凸NN损失
   - 使用PL条件或误差界条件

### 4.3 长期目标（投稿前）

7. **论文撰写**：目标AAAI 2027
   - 标题："AdaTem-FL: Adaptive Temperature Aggregation with Minimax Guarantees for Heterogeneous Task Federated Learning"
   - 核心贡献：7个定理 + Three Regimes框架 + 28个实验验证

8. **代码开源**：整理实验代码，确保可复现

## 5. 风险评估

| 风险 | 概率 | 缓解措施 |
|------|------|----------|
| 14B模型OOM | 中 | 使用4-bit量化或双卡并行 |
| 审稿人质疑novelty | 中 | 强调Theorem 1的softmax识别是首次，Three Regimes是框架性贡献 |
| 实验无法复现 | 低 | 所有实验有详细日志和metrics.json |
| 理论假设过强 | 低 | Theorem 8-9将扩展到非凸情况 |

## 6. 当前状态评估

### Novelty: ★★★★☆

**已有：**
- ✓ 首次将loss-proportional识别为softmax梯度下降（Theorem 1）
- ✓ 首次证明结构性冲突下界（Theorem 6）
- ✓ 首次提出Three Regimes框架
- ✓ 28个实验的系统验证

**还需：**
- Theorem 8（收敛率下界）
- 14B模型验证
- 跨领域验证

**目标AAAI 2027：** 当前状态已有竞争力，加上Theorem 8和14B验证可达★★★★★

---

*Last updated: E28完成，Three Regimes框架确立*
