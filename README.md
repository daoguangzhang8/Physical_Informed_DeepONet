# Physics-Informed DeepONet for Helmholtz Equation

基于物理信息神经算子 (Physics-Informed DeepONet) 的 Helmholtz 方程求解器，支持 PML 边界条件与多震源波场预测。

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
│   ├── train.py          # 训练循环逻辑
│   ├── dataloader.py     # 数据加载与预处理
│   ├── net_module.py     # 网络组件 (FNO, Attention, FiLM等)
│   ├── plotting.py       # 可视化绘图工具
│   ├── utils.py          # 通用工具函数
│   └── FNO.py            # FNO 基准模型
├── output*/              # 训练输出目录 (自动生成)
└── README.md
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

## 快速开始

### 1. 数据准备

将数据集放置于 `load_path` 指定的目录，需包含：

```
data_root/
├── velocity_data_70_70_n1.npy           # 速度场数据 [N, Z, X]
├── backgroundfield_data_freq5_1source_70_70_n1.npy  # 背景场数据
├── wavefield_data_freq5_5sources_70_70_n1.npy      # 波场数据 (多震源)
├── marmousi_velocity_data_70_70_n1.npy    # (可选) 外部测试集
```

### 2. 配置修改

编辑 `config.py` 修改训练参数：

```python
class Args:
    load_path = '/path/to/your/data'
    device = 0                    # GPU 编号
    nvel_train = 1500             # 训练样本数
    batch_size = 700              # 坐标采样批次
    batch_size_v = 35             # 速度场批次
    NIter = 10000               # 训练轮数
```

### 3. 开始训练

```bash
python main2.py
```

### 4. 模型测试

```bash
python test.py
```

## 配置参数说明

### 数据与批次配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `nvel_train` | 1500 | 训练速度模型数量 |
| `ny_train` | 4900 | 空间采样点总数 |
| `batch_size` | 700 | Trunk Net 坐标批次大小 |
| `batch_size_v` | 35 | Branch Net 速度场批次大小 |
| `accumulation_steps` | 4 | 梯度累加步数 (等效增大 batch size) |
| `valid_rate` | 0.1 | 验证集划分比例 |

### 训练控制配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NIter` | 10000 | 总训练轮数 |
| `lr` | 1e-4 | 初始学习率 |
| `warmup_epochs` | 100 | 学习率预热轮数 |
| `weight_decay` | 1e-4 | L2 正则化系数 |
| `save_model_every` | 500 | 模型保存间隔 |
| `save_fig_every` | 50 | 可视化保存间隔 |

### 物理约束配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `a` | 1 | 数据损失权重 |
| `b` | 1 | PDE 损失权重 |
| `c` | 0 | 正则化损失权重 |
| `if_adjust` | True | 是否动态调整损失权重 |

### 网络架构配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `nx` | 70 | x 方向网格数 |
| `nz` | 70 | z 方向网格数 |
| `pml` | True | 是否启用 PML 边界 |
| `Lpml` | 9 | PML 层厚度 |

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
