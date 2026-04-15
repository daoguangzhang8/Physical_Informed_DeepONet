# Physics-Informed DeepONet for Helmholtz Equation

基于物理信息神经算子 (Physics-Informed DeepONet) 的 Helmholtz 方程求解器，支持 PML 边界条件与多震源波场预测。

> 📖 **[查看项目改进记录](IMPROVEMENT.md)** - 按时间顺序记录的所有重要改进和修复

## 项目简介

本项目实现了一个**物理信息约束的 DeepONet** 架构，用于求解带 PML (Perfectly Matched Layer) 吸收边界条件的 **Helmholtz 方程**。模型结合了：

- **FNO (Fourier Neural Operator)**: 作为 Branch 网络，提取速度场和背景场的频域特征
- **FiLM (Feature-wise Linear Modulation)**: 作为 Trunk 网络，实现坐标与物理场特征的交叉调制
- **PDE Loss**: 基于 Helmholtz 方程的物理约束损失函数

### 数学背景

求解频域声波方程 (Helmholtz 方程)：

$$\nabla^2 u + k^2 \omega^2 u = f$$

其中：
- $u = u_r + i \cdot u_i$ 为复数波场
- $k = 1/c^2$ 为波数，$c$ 为速度场
- $\omega$ 为角频率
- PML 边界条件用于吸收边界反射

## 项目结构

```
Physical_Informed_DeepONet/
├── config.py              # 训练配置参数
├── main2.py               # 训练入口脚本
├── test.py               # 测试评估脚本
├── Labconfig.py          # 实验室通用配置
├── model/
│   ├── PI_DeepOnet.py    # PI-DeepONet 主模型定义
│   ├── train.py          # 单 GPU 训练循环逻辑
│   ├── train_distributed.py  # 多 GPU 分布式训练模块
│   ├── dataloader.py     # 数据加载与预处理
│   ├── net_module.py     # 网络组件 (FNO, Attention, FiLM等)
│   ├── plotting.py       # 可视化绘图工具
│   ├── utils.py          # 通用工具函数 (含分布式工具)
│   └── FNO.py            # FNO 基准模型
├── output*/              # 训练输出目录 (自动生成)
├── README.md             # 项目说明文档
├── IMPROVEMENT.md        # 改进记录文档
└── PARALLEL_USAGE.md     # 多 GPU 并行训练详细指南
```

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.0 (推荐 GPU 显存 >= 24GB)

### 依赖安装

```bash
conda create -n pytorch python=3.10
conda activate pytorch
pip install torch numpy matplotlib tqdm scipy
```

## 数据集构成

### 原始数据文件

将数据集放置于 `load_path` 指定的目录，需包含以下文件：

| 数据类型 | 文件名 | 形状 | 说明 |
|---------|--------|------|------|
| 速度场 | `freesurface_velocity_freq3to20_5sources_160_180_pml20_n1.npy` | `[N_vel, NZ_raw, NX_raw]` | N_vel 个速度模型，含 PML 边界 |
| 背景波场 | `freesurface_backgroundfield_...n1.npy` | `[N_vel*5, 2, NZ_raw, NX_raw]` | 5 个震源交织排列，通道 0/1 为实部/虚部 |
| 总波场 | `freesurface_wavefield_...n1.npy` | `[N_vel*5, 2, NZ_raw, NX_raw]` | 同上 |
| 频率 | `freesurface_freq_used_...n1.npy` | `[N_vel]` | 每个速度模型对应一个频率值 |

外部测试集（可选）：

```
data_root/
├── marmousi_velocity_data_70_70_n1.npy        # Marmousi 速度模型
├── marmousi_backgroundfield_data_...n1.npy     # Marmousi 背景场
├── marmousi_wavefield_data_...n1.npy           # Marmousi 波场
├── overthrust_velocity_data_70_70_n1.npy       # Overthrust 速度模型
└── ...
```

### PML 边界裁剪

原始数据包含 PML 吸收层，训练时根据 `boundary_type` 进行裁剪：

| 参数 | 值 | 含义 |
|------|-----|------|
| `pml_total` | 20 | PML 总厚度（网格数） |
| `pml_crop` | 15 | 裁剪掉的 PML 网格数 |
| `pml_active` | 5 | 保留参与训练的 PML 网格数 |

- **`free_surface`**：顶部自由表面不裁剪，底部裁剪 15 格；左右各裁剪 15 格 → `160×180 → 145×150`
- **`full_pml`**：四边各裁剪 15 格 → `160×180 → 130×150`

### 震源拆分

原始波场数据中 5 个震源按速度模型数 `N_vel` 交织排列，需按索引拆分：

```
UU_loc[i]  = UU[i*N_vel : (i+1)*N_vel, :, :, :]    # i = 0,1,2,3,4
UU0_loc[i] = UU0[i*N_vel : (i+1)*N_vel, :, :, :]
```

通过 `source_list` 控制训练中使用哪些震源，如 `[0,1,2,3,4]` 表示使用全部 5 个震源。

### 训练集 / 验证集划分

**速度模型划分：**

```
idx = np.random.choice(N_vel_total, nvel_train, replace=False)  # 随机选取训练集
remaining = 其余速度模型 → 验证集
valid_num = int(valid_rate * nvel_train) + 1                    # 默认 10%+1
```

**多震源拼接：** 对每个激活的震源，将速度模型复制、波场按震源索引提取，沿 batch 维拼接：

```
总样本数 = nvel_train × len(source_list)    # 如 1×5 = 5
```

**标签定义 — 散射场残差：**

```
labels = UU - UU0    # shape: [N_samples, 2, NZ, NX]  (通道0=实部, 通道1=虚部)
```

**速度场归一化：** `vel = vel / 1000.0`（将速度值从 m/s 量级归一化到 1 量级）

### 坐标点采样

| 模式 | 参数 | 采样量 | 说明 |
|------|------|--------|------|
| `full_grid` | — | NZ × NX | 全网格所有点 |
| `halton` | `halton_sample_ratio = 0.5` | NZ × NX × 0.5 | Halton 准随机采样，覆盖更均匀 |

坐标值 = 网格索引 × `dh`（物理坐标，单位：米，`dh = 20`m）。训练/验证共用同一份坐标点。

训练中每个 velocity batch 额外动态生成 **900 个结构感知 PDE 采样点**：50% 分布在速度梯度边界附近，50% 分布在自由表面附近。

### DataLoader 构建

训练阶段构建 6 个 DataLoader：

| 名称 | 数据内容 | batch_size | shuffle | 用途 |
|------|---------|-----------|---------|------|
| `train` | `(vel, UU0, labels[, freq])` | `batch_size_v` | True | 遍历速度模型 |
| `train_y` | `(y_coords,)` | `batch_size` | True | 遍历空间坐标 |
| `valid` | `(vel, UU0, labels[, freq])` | `valid_batch_size_v` | True | 验证集速度模型 |
| `valid_y` | `(y_coords,)` | `valid_batch_size` | True | 验证集坐标 |
| `pred` | `(y_full_grid,)` | `batch_size` | False | 验证绘图全网格 |
| `test` | `(y_full_grid,)` | `batch_size` | False | 训练集绘图 |

**嵌套迭代逻辑：**

```
for vel_batch, UU0_batch, labels_batch in train:       # 外层: 速度模型
    y_ran = model.generate_structure_aware_y_ran(...)   # 自适应PDE采样点
    for y_batch in train_y:                             # 内层: 空间坐标
        y_combined = cat([y_batch, y_ran])              # 拼接数据点+PDE点
        loss = model.loss(vel, y_combined, UU0, labels) # 单次前向 → Data+PDE Loss
        loss.backward()                                 # 梯度累积
```

### 数据流全链路

```
.npy 原始文件 → PML 裁剪 → 震源拆分(5 sources) → 训练/验证划分
    │
    ├─ 速度场: vel / 1000.0
    ├─ 标签: labels = UU - UU0 (散射场残差)
    └─ 坐标: Halton采样 或 全网格, 物理坐标 = index × dh
    │
    ▼
TensorDataset → DataLoader
    │
    ├─ train:   (vel, UU0, labels, freq)  batch_size_v=1
    ├─ train_y: (y_coords,)               batch_size=800
    ├─ valid:   (vel, UU0, labels, freq)  valid_batch_size_v=6
    └─ valid_y: (y_coords,)               valid_batch_size=350
    │
    ▼
嵌套循环: 外层遍历速度模型 → 内层遍历坐标 → 拼接自适应PDE采样点 → 单次前向 → 联合Loss
```

> 详细的数据集文档参见 [docs/dataset_and_dataloader.md](docs/dataset_and_dataloader.md)

## 快速开始

### 1. 配置修改

编辑 `config.py` 修改训练参数：

```python
class Args:
    load_path = '/path/to/your/data'
    device = 0                    # GPU 编号
    nvel_train = 1                # 训练速度模型数
    batch_size = 800              # 坐标采样批次
    batch_size_v = 1              # 速度场批次
    NIter = 5001                  # 训练轮数
```

### 2. 开始训练

#### 单 GPU 训练

```python
# config.py
use_parallel = False
device = 0                        # 指定 GPU 编号
```

```bash
python main2.py
```

#### 单机多 GPU 并行训练

```python
# config.py
use_parallel = True               # 启用多 GPU 并行
num_gpus = 2                      # 使用的 GPU 数量
min_gpu_memory = 10 * 1024        # GPU 最小可用内存 (MB)
```

```bash
python main2.py                   # 自动启动多进程，无需 torchrun
```

> 💡 **提示**: 多 GPU 训练时，程序会自动检测满足 `min_gpu_memory` 要求的 GPU，若可用 GPU 不足则自动回退到单 GPU 模式。

### 3. 模型测试

```bash
python test.py
```

## 配置参数说明

### 单机多卡并行配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_parallel` | False | 是否启用多 GPU 并行训练 |
| `num_gpus` | 2 | 使用的 GPU 数量 |
| `min_gpu_memory` | 23552 | GPU 最小可用内存 (MB)，低于此值的 GPU 不会被使用 |

### 数据与批次配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `nvel_train` | 1500 | 训练速度模型数量 | 
| `batch_size` | 800 | Trunk Net 坐标批次大小 |
| `batch_size_v` | 30 | Branch Net 速度场批次大小 |
| `accumulation_steps` | 4 | 梯度累加步数 (等效增大 batch size) |
| `valid_rate` | 0.1 | 验证集划分比例 |
| `sampling_mode` | `halton` | 采样模式 (`full_grid` / `halton`) |
| `halton_sample_ratio` | 0.5 | Halton 采样比例 |

### 物理网格与边界条件

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dh` | 20 | 空间网格间距 (m) |
| `nz × nx` | 145 × 150 (裁剪后) | 训练网格尺寸 |
| `pml` | True | 是否启用 PML 吸收边界 |
| `pml_total` | 20 | PML 总厚度 (网格数) |
| `pml_crop` | 15 | 训练时裁剪的 PML 网格数 |
| `boundary_type` | `free_surface` | 边界类型 (`free_surface` / `full_pml`) |

### 训练控制配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NIter` | 5001 | 总训练轮数 |
| `lr` | 1e-4 | 初始学习率 |
| `warmup_epochs` | 100 | 学习率预热轮数 |
| `weight_decay` | 1e-4 | L2 正则化系数 |
| `save_model_every` | 500 | 模型保存间隔 |
| `save_fig_every` | 50 | 可视化保存间隔 |

### 标签来源配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_fno_as_label` | False | True: 使用 FNO 预测作为软标签 / False: 使用真实标签 |
| `fno_weights_path` | '' | FNO 预训练权重路径 (当 `use_fno_as_label=True` 时需要指定) |

### 损失函数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `a` | 1 | 数据损失权重 |
| `b` | 1 | PDE 损失权重 |
| `c` | 0 | 正则化损失权重 |
| `if_adjust` | True | 是否动态调整 Loss 权重 |
| `adjust_from` | 2000 | 从第几个 epoch 开始动态调整 |

### Loss 构成说明

总损失函数：

$$L_{total} = a \cdot L_{data} + b \cdot L_{pde}$$

**Data Loss** (仅在标签点上计算):
$$L_{data} = MSE(u_{pred}[y], u_{label}[y])$$

- `use_fno_as_label=False`: $u_{label}$ 为真实波场标签
- `use_fno_as_label=True`: $u_{label}$ 为 FNO 预测值 (软标签)

**PDE Loss** (在所有采样点上计算):
$$L_{pde} = \|\nabla^2 u_{pred} + \omega^2 k^2 u_{pred}\|^2$$

采样点包括:
- 标签点 $y$: 有监督数据的位置
- 自由配点 $y_{ran}$: 结构感知自适应采样生成的物理约束点

### 网络架构配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `in_channels` | 2 | 波场输入通道数 (实部+虚部) |
| `in_channels_vel` | 1 | 速度场输入通道数 |

## 模型架构

### PI-DeepONet

```
Input:
  - vel: 速度场 [B, 1, Z, X]
  - UU0: 背景场 [B, 2, Z, X]
  - y: 查询坐标 [B, N_pts, 2]

Branch Networks:
  - branch1 (FNO): vel → features [B, feat_dim, Z, X]
  - branch2 (FNO): UU0 → features [B, feat_dim, Z, X]
  - Channel Attention + Gaussian Weighted Layer

Trunk Network:
  - Positional Encoding: y → encoded_coords [B, N_pts, 16]
  - FiLM Layers: modulated by branch features

Output:
  - u_pred: 预测波场残差 [B, N_pts, 2] (实部 + 虚部)
```

### 损失函数

```python
L_total = a * L_data + b * L_pde

# 数据损失
L_data = MSE(u_pred, u_true)

# PDE 损失 (Helmholtz 方程残差)
L_pde = mean(|∂²u/∂z² + ∂²u/∂x² + k²ω²u|²)
```

## 输出文件

训练过程会在 `save_doc` 目录下生成：

```
output/
├── PI_DeepONet_pde_PI_model_500epoch_weights_72.pth  # 模型权重
├── loss_log.npy                   # 总损失记录
├── loss_data_log.npy              # 数据损失记录
├── loss_pde_log.npy               # PDE 损失记录
└── *.png                           # 可视化图片
```

## 多 GPU 并行训练

本项目支持单机多卡分布式训练，使用 `torch.multiprocessing.spawn` 内部启动多进程。

### 使用方法

1. **配置参数** (`config.py`)：
```python
use_parallel = True               # 启用多 GPU 并行
num_gpus = 2                      # GPU 数量
min_gpu_memory = 10 * 1024        # 最小可用内存 (MB)
```

2. **启动训练**：
```bash
python main2.py                   # 无需 torchrun，自动启动多进程
```

### 工作原理

```
main2.py
    │
    ├── use_parallel = False ──→ model/train.py (单 GPU)
    │
    └── use_parallel = True  ──→ model/train_distributed.py
                                        │
                                        └── mp.spawn() 启动 N 个进程
```

### 注意事项

- **Batch Size**: 多 GPU 时，实际 batch size = `batch_size * num_gpus`
- **模型保存**: 只在主进程保存，访问原始模型需用 `model.module`
- **数据加载**: 使用 `DistributedSampler` 确保数据不重复

详细说明请参阅 [PARALLEL_USAGE.md](PARALLEL_USAGE.md)

## 外部数据集测试

支持在外部速度模型 (如 Marmousi, BP1994) 上进行泛化测试：

```python
# config.py
ext_val_datasets = {
    'Marmousi': {'prefix': 'marmousi_', 'loc_target': 2},
    'BP1994': {'prefix': '1994BP_', 'loc_target': [0,1,2,3,4]},
}
```

### 域适应微调

启用 `if_finetune=True` 可在外部数据集上进行微调评估：

```python
if_finetune = True
ft_NIter = 1000    # 微调迭代数
ft_lr = 2e-5       # 微调学习率
ft_a = 0.2          # 微调数据损失权重
ft_b = 1            # 微调 PDE 损失权重
```

## 显存优化

本项目已实现以下显存优化：

1. **梯度累加**: 通过 `accumulation_steps` 等效增大 batch size
2. **PDE 采样合并**: 将固定采样点与自适应采样点合并计算，减少前向传播次数
3. **选择性 detach**: PML 边界系数、速度场采样值等不参与梯度计算的部分已 detach
4. **DataLoader 多进程**: 使用 `num_workers=4` 加速数据加载

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{pi deeponet helmholtz,
  author = {Zhang, Daoguang},
  title = {Physics-Informed DeepONet for Helmholtz Equation with PML Boundaries},
  year = {2025},
  howpublished = {\url{https://github.com/xxx/Physical_Informed_DeepONet}}
}
```

## License

MIT License
