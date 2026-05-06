# 三阶段课程学习训练设计

## 1. 总体思路

将频率域地震波场建模（Helmholtz 方程求解）拆分为三个递进阶段：低频 → 中频 → 高频。低频波场空间变化平缓，网络更容易学习；高频波场细节丰富、梯度更大，学习难度更高。通过课程学习（Curriculum Learning）策略，让网络逐步掌握不同频率尺度的特征，提升泛化能力和收敛稳定性。

## 2. 数据集构成

### 2.1 原始数据

来源路径：`/home/sharedata/zdg/multifreq/`

原始数据包含 20000 个速度模型，每个模型在 5 个震源位置下生成波场，按三个频率段分别存储：

| 阶段子目录 | 频率范围 | 速度场 shape | 背景场/波场 shape | 频率 shape |
|-----------|---------|-------------|------------------|-----------|
| `freq_3to11` | 3–11 Hz (离散值: 3, 5, 7, 9, 11) | (20000, 160, 180) | (100000, 2, 160, 180) | (20000,) |
| `freq_12to18` | 12–18 Hz (离散值: 12, 14, 15, 16, 18) | (20000, 160, 180) | (100000, 2, 160, 180) | (20000,) |
| `freq_18to25` | 18–25 Hz (离散值: 19, 21, 22, 23, 25) | (20000, 160, 180) | (100000, 2, 160, 180) | (20000,) |

说明：
- 背景场/波场的 100000 = 20000 × 5（5 个震源按顺序排列：`[src0 × N, src1 × N, ..., src4 × N]`）
- 通道数 2 代表复数波场的实部和虚部
- 三个阶段的**速度场完全相同**（同一组地质模型），仅波场/背景场因频率不同而不同
- 频率文件 `freq_used` 在三个阶段中文件名相同（不含 stage tag），但内容对应各自的频率值

### 2.2 裁切数据

通过 `prepare_multifreq_selected.py` 脚本，使用 `np.random.seed(1)` + `np.random.choice(20000, 2000, replace=False)` 从 20000 个模型中选取 2000 个，保存到 `/home/sharedata/zdg/multifreq_selected/`。

选取规则：
- 三个阶段使用**完全相同的 2000 个索引**，保证速度模型一致性
- 震源排列保持原始顺序：`[src0 × 2000, src1 × 2000, ..., src4 × 2000]`

裁切后数据规模：

| 阶段 | 速度场 | 背景场/波场 | 频率 |
|-----|--------|-----------|------|
| freq_3to11 | (2000, 160, 180) | (10000, 2, 160, 180) | (2000,) |
| freq_12to18 | (2000, 160, 180) | (10000, 2, 160, 180) | (2000,) |
| freq_18to25 | (2000, 160, 180) | (10000, 2, 160, 180) | (2000,) |

### 2.3 训练/验证划分

`dataloader.py` 中 `Training_data()` 使用 `np.random.seed(1)` 后 `np.random.choice` 选取 `nvel_train` 个模型作为训练集，剩余作为验证集。

当前配置：`nvel_train = 1200`，即从 2000 个中选 1200 个训练，800 个验证。每个震源单独处理后再按 batch 维度拼接，实际训练样本数为 `1200 × len(source_list)`。

### 2.4 PML 边界裁切

所有数据在加载时经过 PML (Perfectly Matched Layer) 边界裁切：

```
原始尺寸: 160 (z) × 180 (x)
边界类型: free_surface（顶部自由表面 + 其余三边 PML）
PML 裁切: pml_crop = 15
裁切后:   z_slice = [0, -15]  → 145
          x_slice = [15, -15] → 150
网络输入: 145 × 150
```

## 3. 阶段切换策略

### 3.1 阶段定义

```python
stages = [
    {   # Stage 0: 低频
        'name': 'low_freq',
        'freq_range': '3to11',
        'NIter': 501,
        'lr': 1e-4,
        'warmup_epochs': 100,
        'replay_stages': [],
        'data_dir': '/home/sharedata/zdg/multifreq_selected/freq_3to11',
    },
    {   # Stage 1: 中频
        'name': 'mid_freq',
        'freq_range': '12to18',
        'NIter': 501,
        'lr': 5e-5,
        'warmup_epochs': 50,
        'replay_stages': [0],
        'data_dir': '/home/sharedata/zdg/multifreq_selected/freq_12to18',
    },
    {   # Stage 2: 高频
        'name': 'high_freq',
        'freq_range': '18to25',
        'NIter': 501,
        'lr': 2e-5,
        'warmup_epochs': 50,
        'replay_stages': [0, 1],
        'data_dir': '/home/sharedata/zdg/multifreq_selected/freq_18to25',
    },
]
```

### 3.2 切换流程

每个阶段结束时，模型权重保存为 `{filename}_stage{i}_final_weights_{nz}.pth`。

下一阶段开始时：
1. **权重继承**：加载上一阶段的最终权重（`model.load_state_dict`），作为当前阶段的初始化
2. **数据切换**：通过 `data_dir` 指向新阶段的数据目录，文件名通过替换 `freq3to20` → `freq{range}` 得到
3. **优化器重置**：创建全新的 Adam 优化器和 ReduceLROnPlateau 调度器，使用当前阶段专属的学习率
4. **Replay 数据合并**：将前序阶段训练数据 cat 拼接到当前训练集（详见第 4 节）

### 3.3 数据文件名映射

vel/bg/wf 文件名包含频率范围标签，在阶段切换时替换：

```
基础文件名: freesurface_velocity_freq3to20_5sources_160_180_pml20_n1.npy
  → Stage 0: freesurface_velocity_freq3to11_5sources_160_180_pml20_n1.npy
  → Stage 1: freesurface_velocity_freq12to18_5sources_160_180_pml20_n1.npy
  → Stage 2: freesurface_velocity_freq18to25_5sources_160_180_pml20_n1.npy
```

freq 文件不含阶段标签，三个阶段统一为：
```
freesurface_freq_used_5sources_160_180_pml20_n1.npy
```

## 4. Replay 防遗忘机制

### 4.1 设计目的

课程学习中，后续阶段专注于高频数据时，模型可能遗忘低频段已学到的特征。Replay 机制通过在训练集中混入前序阶段的完整数据来缓解灾难性遗忘。

### 4.2 工作方式

| 阶段 | 当前数据 | Replay 来源 | replay_ratio | 合并后训练集大小 |
|-----|---------|-----------|-------------|----------------|
| Stage 0 | freq_3to11 (1200 models) | 无 | — | 1200 × 1 src = 1200 |
| Stage 1 | freq_12to18 (1200 models) | + Stage 0 × 20% | 0.2 | 1200 + 240 = 1440 |
| Stage 2 | freq_18to25 (1200 models) | + Stage 0 × 20% + Stage 1 × 20% | 0.2 | 1200 + 240 + 240 = 1680 |

（上表假设 `source_list = [0]`，若使用多震源则乘以震源数）

### 4.3 实现流程

1. 加载当前阶段数据，构建 DataLoader
2. 从当前 DataLoader 的 `dataset.tensors` 中提取训练集 Tensor
3. 遍历 `replay_stages` 列表，逐个加载前序阶段数据（修改 `load_path` 和文件名）
4. 按 `replay_ratio` 对 replay 数据随机采样（1.0 = 全部使用）
5. 将 replay 数据 cat 拼接到当前训练集
6. 用合并后的 Tensor 重建 DataLoader
7. 恢复当前阶段的文件名和 `load_path`

### 4.4 可配置参数

- `replay_stages`：列表，指定要 replay 的前序阶段编号（如 `[0]` 或 `[0, 1]`）
- `replay_ratio`：浮点数，replay 数据保留比例。1.0 = 全部，0.2 = 随机抽取 20%。当前 Stage 1/2 均设为 0.2，在防遗忘与训练效率间取平衡

## 5. 各阶段训练策略

### 5.1 通用配置

| 参数 | 值 | 说明 |
|-----|---|------|
| `batch_size` | 900 | Trunk Net 坐标采样批次 |
| `batch_size_v` | 40 | Branch Net 速度场批次 |
| `accumulation_steps` | 4 | 梯度累加（等效 batch_size_v = 160） |
| `weight_decay` | 1e-4 | L2 正则化 |
| 优化器 | Adam | — |
| 调度器 | ReduceLROnPlateau | factor=0.9, patience=30, min_lr=1e-5 |

### 5.2 Stage 0 — 低频 (3–11 Hz)

| 参数 | 值 | 说明 |
|-----|---|------|
| NIter | 501 | — |
| lr | 1e-4 | 最高学习率，从头训练 |
| warmup | 100 epochs | 线性从 1e-5 升到 1e-4 |
| Loss 权重 | a=1, b=1, c=0, d=1 | Data + PDE + Envelope |
| 权重初始化 | `_init_weights()` | Xavier/He 随机初始化 |
| Replay | 无 | — |

**设计考量**：低频波场平滑，梯度小，网络容易拟合。使用最高学习率加速收敛，较长的 warmup 保证初始稳定。此阶段建立网络对波场基本结构的认知。

### 5.3 Stage 1 — 中频 (12–18 Hz)

| 参数 | 值 | 说明 |
|-----|---|------|
| NIter | 501 | — |
| lr | 5e-5 | 降为 Stage 0 的 50% |
| warmup | 50 epochs | 较短 warmup（已有低频基础） |
| Loss 权重 | a=1, b=1, c=0, d=1 | 同 Stage 0 |
| 权重初始化 | 继承 Stage 0 最终权重 | — |
| Replay | Stage 0 × 20% | 防遗忘低频特征 |

**设计考量**：中频波场细节增加，降低学习率避免破坏已学到的低频特征。引入 Stage 0 的 20% 数据作为 replay（`replay_ratio=0.2`），在防遗忘与训练效率间取平衡。warmup 减半，因为模型已有合理的参数空间分布。

### 5.4 Stage 2 — 高频 (18–25 Hz)

| 参数 | 值 | 说明 |
|-----|---|------|
| NIter | 501 | — |
| lr | 2e-5 | 最低学习率 |
| warmup | 50 epochs | — |
| Loss 权重 | a=1, b=1, c=0, d=1 | 同前 |
| 权重初始化 | 继承 Stage 1 最终权重 | — |
| Replay | Stage 0 × 20% + Stage 1 × 20% | 防遗忘所有低中频特征 |

**设计考量**：高频波场空间变化剧烈，学习难度最大。使用最低学习率进行精细调整。同时以 20% 比例 replay 低频和中频数据，保持全频段能力。训练集扩充至约 1.4 倍基础数据量。

### 5.5 学习率调度对比

```
Stage 0:  1e-5 ──── warmup (100) ────→ 1e-4 ── ReduceLROnPlateau ──→ ≥1e-5
Stage 1:  5e-6 ─── warmup (50) ─────→ 5e-5 ── ReduceLROnPlateau ──→ ≥1e-5
Stage 2:  2e-6 ─── warmup (50) ─────→ 2e-5 ── ReduceLROnPlateau ──→ ≥1e-5
```

## 6. Epoch-Level 共享空间采样

每个 epoch 开始时，基于当前训练集的速度场计算全局梯度概率图，采样一批共享的空间坐标点（y_ran）用于 PDE 残差计算。

采样构成：
采样构成（总计 500 个点）：
- 60% 结构感知点（300 点，沿速度梯度大的区域采样）
- 20% 表层点（100 点，自由表面附近 5 个网格深度内）
- 20% 均匀随机点（100 点）

概率图按 `y_ran_prob_update_every` 的频率更新（当前为每个 epoch 更新）。

## 7. 损失函数

$$L = a \cdot L_{data} + b \cdot L_{PDE} + c \cdot L_{reg} + d \cdot L_{env}$$

| 损失项 | 权重 | 说明 |
|-------|-----|------|
| $L_{data}$ (a) | 1 | 数据拟合：预测 vs 标签的 MSE |
| $L_{PDE}$ (b) | 1 | 物理残差：Helmholtz 方程残差 |
| $L_{reg}$ (c) | 0 | 正则化（当前未启用） |
| $L_{env}$ (d) | 1 | 包络损失：复数波场包络的 MSE |

三个阶段使用相同的损失权重配置。动态权重调整（`if_adjust`）当前未启用。
