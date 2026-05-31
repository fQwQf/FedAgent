# FedRNK 实验架构设计

## 设计原则

1. **配置驱动**：所有超参、环境、方法选择通过 YAML 配置，零代码切换实验
2. **日志可追溯**：每个实验独立目录，结构化日志(JSONL)，支持 WandB
3. **模块解耦**：聚合策略、训练循环、数据处理独立可替换
4. **依赖最小**：只用 conda env `realm` 已有的库（torch, transformers, peft, trl, accelerate）
5. **单机模拟联邦**：8×3090 上用多进程模拟多 client，不用真实网络

---

## 目录结构

```
FedRNK/
├── configs/                    # 所有实验配置 (YAML)
│   ├── base.yaml               # 基础超参（模型、LoRA、学习率）
│   ├── envs/                   # 环境配置
│   │   ├── babyai.yaml
│   │   ├── webshop.yaml
│   │   ├── textcraft.yaml
│   │   ├── maze.yaml
│   │   └── wordle.yaml
│   └── methods/                # 聚合方法配置
│       ├── local.yaml          # 不共享
│       ├── fedavg.yaml         # 全部共享
│       ├── fedrnk.yaml         # 只共享Attention
│       ├── fedrnk_inv.yaml     # 只共享MLP
│       └── centralized.yaml    # 中心化
├── src/
│   ├── __init__.py
│   ├── config.py               # 配置加载与合并（OmegaConf）
│   ├── logging_utils.py        # 结构化日志 + 实验目录管理
│   ├── model.py                # 模型加载 + LoRA配置 + 分离工具
│   ├── data.py                 # 轨迹数据加载与SFT数据构建
│   ├── client.py               # 本地训练（SFT/GRPO）
│   ├── server.py               # 聚合服务器（多种策略）
│   ├── aggregation.py          # 聚合策略实现
│   ├── train.py                # 主训练循环
│   └── analysis/
│       ├── __init__.py
│       ├── gradient_collector.py  # E0: 收集每层梯度
│       ├── gradient_analysis.py   # E0: 偏差计算、余弦相似度
│       └── visualize.py           # 绘图工具
├── scripts/
│   ├── run_e0.sh               # E0预实验启动脚本
│   ├── run_main.sh             # 主实验启动脚本
│   └── analyze_e0.py           # E0结果分析
├── outputs/                    # 实验输出（gitignore）
│   └── {experiment_name}/
│       ├── config.yaml          # 本实验的完整配置快照
│       ├── events.jsonl         # 结构化事件日志
│       ├── metrics.jsonl        # 训练指标（每轮loss/reward/score）
│       ├── checkpoints/         # 模型检查点
│       └── analysis/            # 分析输出
└── pyproject.toml               # 项目元数据（可选）
```

---

## 核心设计决策

### 1. 配置系统：OmegaConf (Hydra风格)

```yaml
# configs/base.yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  lora:
    rank: 8
    alpha: 16
    dropout: 0.05
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj",   # Attention
                      "gate_proj", "up_proj", "down_proj"]       # MLP

training:
  num_rounds: 20
  local_epochs: 1
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 5e-5
  max_seq_length: 2048
  seed: 42

federation:
  num_clients: 5
  method: "fedrnk"   # local | fedavg | fedrnk | fedrnk_inv | centralized
  # FedRNK特有配置
  aggregate_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]  # 只聚合这些
  isolate_modules: ["gate_proj", "up_proj", "down_proj"]        # 这些不聚合

logging:
  experiment_name: "e0_baseline"
  log_dir: "outputs"
  use_wandb: false
  log_gradients: true   # E0需要
  log_interval: 1       # 每轮都记录
```

**合并规则**：`base.yaml` ← `envs/{env}.yaml` ← `methods/{method}.yaml` ← CLI覆盖

### 2. 日志系统：结构化 JSONL

每条日志一行 JSON，带时间戳和类型标签：

```jsonl
{"ts": "2025-06-01T10:00:00", "type": "round_start", "round": 1}
{"ts": "2025-06-01T10:05:00", "type": "client_update", "round": 1, "client": 0, "env": "babyai", "loss": 2.31, "num_trajectories": 50, "success_rate": 0.83}
{"ts": "2025-06-01T10:05:01", "type": "client_update", "round": 1, "client": 1, "env": "webshop", "loss": 2.45, "num_trajectories": 45, "success_rate": 0.73}
{"ts": "2025-06-01T10:10:00", "type": "aggregation", "round": 1, "method": "fedrnk", "num_params_shared": 123456, "num_params_isolated": 234567}
{"ts": "2025-06-01T10:10:00", "type": "gradient_stats", "round": 1, "client": 0, "attn_grad_norm": 1.23, "mlp_grad_norm": 4.56, "attn_cosine_with_global": 0.89, "mlp_cosine_with_global": 0.45}
{"ts": "2025-06-01T10:10:00", "type": "round_end", "round": 1, "total_time_sec": 600}
{"ts": "2025-06-01T10:30:00", "type": "eval", "round": 1, "client": 0, "env": "babyai", "score": 0.85}
```

**可读性**：同时输出到 console（彩色人类可读）+ JSONL（机器可解析）。

### 3. 模型管理：Attention/MLP分离

```python
# src/model.py 核心接口

def get_lora_config(cfg) -> LoraConfig:
    """创建LoRA配置，target_modules包含Attention和MLP"""

def split_lora_params(model) -> tuple[dict, dict]:
    """
    将LoRA参数分为Attention和MLP两组。
    返回 (attn_params, mlp_params)，每个是 {name: param} dict。
    
    基于模块名匹配：
    - Attention: */{q,k,v,o}_proj/lora_*
    - MLP: */{gate,up,down}_proj/lora_*
    """

def get_lora_state_dict(model, modules: str) -> dict:
    """
    只获取指定模块的LoRA state dict。
    modules="attn" → 只返回Attention LoRA
    modules="mlp" → 只返回MLP LoRA  
    modules="all" → 返回全部LoRA
    """
```

### 4. 聚合策略：Strategy模式

```python
# src/aggregation.py

class AggregationStrategy(ABC):
    @abstractmethod
    def aggregate(self, client_states: list[dict]) -> dict:
        """聚合多个client的LoRA更新，返回全局LoRA state dict"""
        pass

class FedAvg(AggregationStrategy):
    """聚合全部LoRA参数"""
    def aggregate(self, client_states):
        return average_state_dicts(client_states)

class FedRNK(AggregationStrategy):
    """只聚合Attention LoRA，MLP保持本地"""
    def aggregate(self, client_states):
        attn_states = [filter_attn(s) for s in client_states]
        global_attn = average_state_dicts(attn_states)
        return global_attn  # 只有Attention部分

class FedRNKInv(AggregationStrategy):
    """只聚合MLP LoRA，Attention保持本地"""
    def aggregate(self, client_states):
        mlp_states = [filter_mlp(s) for s in client_states]
        global_mlp = average_state_dicts(mlp_states)
        return global_mlp

class Local(AggregationStrategy):
    """不聚合"""
    def aggregate(self, client_states):
        return {}  # 空dict，不更新

class Centralized(AggregationStrategy):
    """中心化训练等价：所有数据合并"""
    # 特殊处理：不走联邦流程，直接全数据训练
```

### 5. 训练循环

```python
# src/train.py 核心伪代码

def federated_train(cfg):
    # 1. 初始化
    model, tokenizer = load_base_model(cfg)
    model = apply_lora(model, cfg)
    strategy = get_strategy(cfg.federation.method)
    clients = [Client(k, cfg) for k in range(cfg.federation.num_clients)]
    
    for round in range(cfg.training.num_rounds):
        log("round_start", round=round)
        
        # 2. Client本地训练（并行）
        client_updates = []
        for client in clients:
            # 每个 client 加载全局模型
            client.load_global_model(model)
            # 本地交互+训练
            update = client.local_train()
            client_updates.append(update)
            log("client_update", client=client.id, **update.metrics)
        
        # 3. 聚合
        global_update = strategy.aggregate(client_updates)
        log("aggregation", method=strategy.name, num_params=len(global_update))
        
        # 4. 广播+更新
        for client in clients:
            client.apply_update(global_update)
        
        # 5. 评估
        for client in clients:
            eval_result = client.evaluate()
            log("eval", client=client.id, **eval_result)
    
    # 6. 保存
    save_final_model(model, cfg)

def get_strategy(method: str) -> AggregationStrategy:
    return {
        "local": Local,
        "fedavg": FedAvg,
        "fedrnk": FedRNK,
        "fedrnk_inv": FedRNKInv,
        "centralized": Centralized,
    }[method]()
```

### 6. E0预实验：梯度分析

```python
# src/analysis/gradient_collector.py

def collect_gradients(model, dataloader, env_name: str) -> dict:
    """
    收集一个环境的LoRA梯度。
    返回:
    {
        "attn_grad": flattened_attention_gradient,
        "mlp_grad": flattened_mlp_gradient,
        "per_layer_attn_grad": {layer_idx: gradient},
        "per_layer_mlp_grad": {layer_idx: gradient},
    }
    """

# src/analysis/gradient_analysis.py

def compute_deviation(env_grads: dict[str, torch.Tensor]) -> dict:
    """
    计算P1的核心指标。
    输入: {env_name: {"attn_grad": ..., "mlp_grad": ...}}
    输出:
    {
        "attn_deviation_norm": mean(||Δ_k^attn||^2),     # P1的主要指标
        "mlp_deviation_norm": mean(||Δ_k^mlp||^2),        # P1的主要指标
        "attn_cosine_matrix": [[cos(g_i, g_j) for i,j]],  # 辅助
        "mlp_cosine_matrix": [[cos(g_i, g_j) for i,j]],    # 辅助
        "ratio": attn_deviation_norm / mlp_deviation_norm,   # <0.5 → H1成立
    }
    """
```

### 7. GPU分配策略

8×RTX 3090，5个client：
- E0（梯度收集）：串行，每次1个client用1-2张GPU
- 主实验（SFT训练）：每client 1-2张GPU，可2-3个client并行
- 中心化基线：全8张GPU

```python
# GPU分配逻辑
def get_gpu_assignment(num_clients: int, gpus_per_client: int = 1):
    """返回 {client_id: [gpu_ids]}"""
    total_gpus = torch.cuda.device_count()
    assignments = {}
    for i in range(num_clients):
        start = (i * gpus_per_client) % total_gpus
        assignments[i] = list(range(start, start + gpus_per_client))
    return assignments
```

---

## 依赖清单（全部已在 realm env 中）

| 包 | 版本 | 用途 |
|---|---|---|
| torch | 2.9.1 | 训练核心 |
| transformers | 4.57.6 | 模型加载 |
| peft | 0.18.1 | LoRA管理 |
| trl | 0.29.0 | SFTTrainer |
| accelerate | 1.12.0 | 多GPU |
| omegaconf | 需安装 | 配置系统 |
| wandb | 可选 | 实验追踪 |

**只需额外安装**: `omegaconf`（轻量配置库，pip install）
**可选安装**: `wandb`（实验追踪）

---

## 实验启动方式

```bash
# E0: 梯度一致性验证
bash scripts/run_e0.sh

# 等价于:
python -m src.train \
    +method=fedrnk \
    +experiment=e0 \
    training.num_rounds=3 \
    logging.experiment_name=e0_gradient_probe \
    logging.log_gradients=true

# 主实验: 5种方法 × 5个环境
bash scripts/run_main.sh

# 等价于:
for method in local fedavg fedrnk fedrnk_inv centralized; do
    python -m src.train \
        +method=$method \
        +experiment=main \
        logging.experiment_name=main_${method}
done
```
