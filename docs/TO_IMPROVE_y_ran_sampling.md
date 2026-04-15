# TO IMPROVE: y_ran 残差引导自适应采样

> 状态：待实施
> 日期：2026-04-13
> 核心约束：y_ran 在每个 batch 间必须共享同一张量 `[1, N_ran, 2]`，不可依赖当前 vel_batch

---

## 目录

1. [问题分析](#1-问题分析)
2. [设计方案](#2-设计方案)
3. [修改文件清单](#3-修改文件清单)
4. [实施细节](#4-实施细节)
   - 4.1 [config.py — 新增采样参数](#41-configpy)
   - 4.2 [model/PI_DeepOnet.py — 核心模型改动](#42-modelpi_deeponetpy)
   - 4.3 [model/train.py — epoch 级 y_ran 管理](#43-modeltrainpy)
   - 4.4 [model/utils.py — 初始重要性图计算](#44-modelutilspy)
   - 4.5 [model/train_distributed.py — 分布式兼容](#45-modeltrain_distributedpy)
5. [验证方法](#5-验证方法)

---

## 1. 问题分析

### 1.1 当前实现

文件：`model/PI_DeepOnet.py`，方法 `generate_structure_aware_y_ran()`（第 325-400 行）

```python
# 当前调用位置：model/train.py 第 111-112 行
with torch.no_grad():
    y_ran = model.generate_structure_aware_y_ran(vel_batch, num_pts=900)
```

当前策略（900 个点，每个 batch 重新生成）：

| 比例 | 策略 | 依据 |
|------|------|------|
| 50% | 按速度场梯度幅值 `multinomial` 采样 | 依赖当前 `vel_batch` 的梯度 |
| 50% | 在 z < 2 个网格深度的表层均匀采样 | 固定几何区域 |

### 1.2 存在的问题

1. **显存约束冲突**：当前 `y_ran` 形状为 `[B_v, 900, 2]`，每个 batch 依赖 `vel_batch` 重新计算梯度并采样，无法在 batch 间共享
2. **PML 过渡带盲区**：PML 区域是 PDE 残差最大的区域之一，当前完全未针对性采样
3. **表层范围过窄**：仅覆盖前 2 个网格点（约 40m），对自由表面波场约束不足
4. **比例硬编码**：50/50 不可配置，不随训练进度调整
5. **不感知模型表现**：无法知道模型当前在哪些区域 PDE 残差最大，采样策略是静态的

### 1.3 核心约束

由于显存限制，`y_ran` 必须：
- 形状为 `[1, N_ran, 2]`，通过 `expand` 广播到 `[B_v, N_ran, 2]`
- 在 epoch 级别（而非 batch 级别）生成和更新
- 不依赖当前 batch 的速度场数据

---

## 2. 设计方案

### 2.1 方案概述：残差引导 + Epoch 级更新

```
训练前
  └─ 计算数据集统计重要性图 → initial_importance_map [nz, nx]
  └─ 生成初始 y_ran [1, 900, 2]

Epoch 循环
  ├─ epoch == 0?
  │    └─ 使用初始 y_ran
  │
  ├─ epoch % K == 0 (K=50)?
  │    ├─ 取 5 个速度模型，在粗网格(35x35)上评估 PDE 残差
  │    ├─ 残差图 R [35,35] → 上采样到 [nz, nx]
  │    ├─ 混合：importance_map = 0.7*R + 0.3*initial_map
  │    └─ 从 importance_map 重新采样生成新 y_ran [1, 900, 2]
  │
  ├─ 非 update epoch?
  │    └─ 复用上一轮 y_ran
  │
  └─ 遍历所有 batch，共享同一个 y_ran
       └─ loss() 中 y_ran.expand(B_v, -1, -1) 后与 y 拼接
```

### 2.2 采样点构成（修改后）

| 比例 | 策略 | 目标 |
|------|------|------|
| 20% | 表层均匀采样（z < 5 个网格深度） | 自由表面边界条件约束 |
| 80% | 重要性加权采样（基于残差图或统计先验） | 模型弱点区域 + 物理复杂区域 |

### 2.3 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 残差评估频率 | 每 50 epoch 一次 | 平衡开销与适应性 |
| 粗网格倍率 | 4x 降采样（140→35） | 35x35=1225 点，5 个模型约 6000 点，显存可控 |
| 初始策略 | 数据集速度梯度 + 波长先验 | 训练初期无残差信息时的物理先验 |
| 混合权重 | 70% 残差 + 30% 先验 | 避免残差噪声导致采样退化，保持基本覆盖 |
| 表层深度 | 5 个网格（约 100m） | 覆盖自由表面附近波场剧变区域 |

---

## 3. 修改文件清单

| 文件 | 改动类型 | 改动量 |
|------|---------|--------|
| `config.py` | 新增参数 | +8 行 |
| `model/PI_DeepOnet.py` | 改方法签名 + 删旧方法 + 加 2 新方法 | ~100 行 |
| `model/train.py` | 删 per-batch 采样 + 加 epoch 级逻辑 | ~25 行 |
| `model/utils.py` | 新增 1 个函数 | ~25 行 |
| `model/train_distributed.py` | 同步 train.py 的改动 | ~15 行 |

---

## 4. 实施细节

### 4.1 config.py

在第 4 节（数据集与采样）末尾新增：

```python
    # ==========================================
    # 4b. y_ran 自适应采样配置 (Residual-guided)
    # ==========================================
    y_ran_num_pts = 900                     # 自适应采样点总数
    y_ran_surface_ratio = 0.2               # 表层采样比例（0.2 = 20%）
    y_ran_surface_depth_grids = 5           # 表层深度（网格数，原为 2）
    y_ran_update_interval = 50              # 残差图更新间隔（epoch）
    y_ran_residual_samples = 5              # 残差评估抽取的速度模型数量
    y_ran_coarse_scale = 4                  # 粗网格降采样倍率（140→35）
    y_ran_residual_weight = 0.7             # 残差图在混合重要性图中的权重
```

### 4.2 model/PI_DeepOnet.py

#### 4.2.1 修改 `_compute_pde_residual` 方法签名

**位置**：第 153 行

**当前**：
```python
def _compute_pde_residual(self, vel, y, UU0, Delta_U, freq_batch=None):
```

**改为**：
```python
def _compute_pde_residual(self, vel, y, UU0, Delta_U, freq_batch=None, return_per_point=False):
```

**当前最后一行**（第 271 行）：
```python
return torch.mean(residual_real ** 2 + residual_imag ** 2)
```

**改为**：
```python
per_point_residual = residual_real ** 2 + residual_imag ** 2  # [B_v, B_pts]
if return_per_point:
    return per_point_residual
return torch.mean(per_point_residual)
```

> 注意：此修改向后兼容，`return_per_point=False` 时行为与原版完全一致。

#### 4.2.2 新增 `compute_residual_map` 方法

在 `_compute_pde_residual` 方法之后添加：

```python
@torch.no_grad()
def compute_residual_map(self, vel_samples, UU0_samples, freq_samples=None, coarse_scale=4):
    """
    在粗网格上评估 PDE 残差，返回平均残差图（用于重要性采样）。

    Args:
        vel_samples: [N, 1, Z, X] 少量速度模型（通常 5 个）
        UU0_samples: [N, 2, Z, X] 对应的背景场
        freq_samples: [N] 对应的频率值，若 None 使用默认值
        coarse_scale: 粗网格降采样倍率（4 表示 140→35）

    Returns:
        residual_map: [cz, cx] 平均 PDE 残差图（CPU tensor）
    """
    device = next(self.parameters()).device
    N = vel_samples.shape[0]
    nz = vel_samples.shape[2]
    nx = vel_samples.shape[3]
    cz = nz // coarse_scale
    cx = nx // coarse_scale

    # 生成粗网格坐标（均匀分布在整个物理域）
    z_indices = torch.linspace(0, nz - 1, cz, device=device).long()
    x_indices = torch.linspace(0, nx - 1, cx, device=device).long()
    gz, gx = torch.meshgrid(z_indices, x_indices, indexing='ij')
    y_coarse = torch.stack([gz.flatten(), gx.flatten()], dim=1).float() * self.args.dh
    y_coarse = y_coarse.unsqueeze(0).expand(N, -1, -1).to(device)  # [N, cz*cx, 2]
    y_coarse.requires_grad_(True)

    # 前向传播 + PDE 残差（逐点）
    self.eval()
    Delta_U = self.forward(vel_samples.to(device), y_coarse, UU0_samples.to(device))
    per_point_res = self._compute_pde_residual(
        vel_samples.to(device), y_coarse, UU0_samples.to(device), Delta_U,
        freq_batch=freq_samples.to(device) if freq_samples is not None else None,
        return_per_point=True
    )  # [N, cz*cx]

    # 对所有样本取均值，reshape 为 2D 网格
    avg_residual = per_point_res.mean(dim=0).cpu().view(cz, cx)
    return avg_residual
```

#### 4.2.3 删除旧方法，新增 `generate_y_ran`

**删除**：`generate_structure_aware_y_ran` 方法（第 325-400 行）

**替换为**：

```python
def generate_y_ran(self, importance_map, args):
    """
    基于重要性概率图生成固定的 y_ran（epoch 级共享）。

    Args:
        importance_map: [H, W] 采样概率图（无需预先归一化）
        args: 配置对象，需包含 y_ran_num_pts, y_ran_surface_ratio,
              y_ran_surface_depth_grids, dh

    Returns:
        y_ran: [1, num_pts, 2] requires_grad=True
    """
    nz, nx = importance_map.shape
    num_pts = args.y_ran_num_pts
    num_surface = int(num_pts * args.y_ran_surface_ratio)
    num_importance = num_pts - num_surface
    dh = args.dh

    # --- 1. 重要性加权采样 ---
    prob = importance_map.view(-1).clone()
    prob = prob / (prob.sum() + 1e-8)
    sampled_indices = torch.multinomial(prob, num_samples=num_importance, replacement=True)
    z_idx = sampled_indices // nx
    x_idx = sampled_indices % nx
    # 网格内加随机偏移，避免所有点都在网格中心
    z_imp = z_idx.float() * dh + torch.rand(num_importance) * dh
    x_imp = x_idx.float() * dh + torch.rand(num_importance) * dh

    # --- 2. 表层采样（保留，比例和深度可配置）---
    surface_depth = args.y_ran_surface_depth_grids * dh
    z_surf = torch.rand(num_surface) * surface_depth
    x_surf = torch.rand(num_surface) * (nx * dh)

    # --- 3. 合并并返回 ---
    y_ran = torch.cat([
        torch.stack([z_imp, x_imp], dim=1),
        torch.stack([z_surf, x_surf], dim=1),
    ], dim=0)  # [num_pts, 2]

    return y_ran.unsqueeze(0).requires_grad_(True)  # [1, num_pts, 2]
```

#### 4.2.4 修改 `loss()` 方法中的 y_ran 拼接

**位置**：`loss()` 方法第 435 行附近

**当前**：
```python
y_combined = torch.cat([y, y_ran], dim=1)  # [B_v, B_pts + B_ran_pts, 2]
```

**改为**：
```python
# 兼容 [1, N, 2] 和 [B_v, N, 2] 两种形状
if y_ran.shape[0] != batch_size_v:
    y_ran = y_ran.expand(batch_size_v, -1, -1)
y_combined = torch.cat([y, y_ran], dim=1)  # [B_v, B_pts + B_ran_pts, 2]
```

> 同样的修改需要应用于 `compute_loss()` 方法（DDP 训练用）中对应的拼接位置。

### 4.3 model/train.py

#### 4.3.1 新增 import

在文件顶部 import 区域添加：
```python
from model.utils import compute_dataset_importance_map
```

#### 4.3.2 训练循环前：生成初始 y_ran

**位置**：第 75 行 `optimizer.zero_grad()` 之前，新增：

```python
# ==========================================
# 3b. 初始化 y_ran（基于数据集统计的重要性图）
# ==========================================
# 获取训练集速度场全量数据（从 dataloader 的 dataset 中提取）
vel_train_all = dataloader['train'].dataset.tensors[0]  # [N, 1, Z, X]
UU0_train_all = dataloader['train'].dataset.tensors[1]  # [N, 2, Z, X]
has_freq = len(dataloader['train'].dataset.tensors) >= 4
freq_train_all = dataloader['train'].dataset.tensors[3] if has_freq else None  # [N]

# 计算初始重要性图
initial_importance_map = compute_dataset_importance_map(
    vel_train_all,
    freq=getattr(args, 'default_freq', 5.0)
)  # [nz, nx]
importance_map = initial_importance_map.clone()
y_ran = model.generate_y_ran(importance_map, args).to(device)
print(f"[y_ran] 初始重要性图已生成，形状: {importance_map.shape}")
print(f"[y_ran] 初始采样点: {y_ran.shape}")
```

#### 4.3.3 Epoch 循环内：残差更新逻辑

**位置**：第 81 行 `for i in pbar:` 循环体开头，`model.train()` 之前，新增：

```python
# --- 每 K epoch 更新 y_ran ---
if i > 0 and i % args.y_ran_update_interval == 0:
    n_samples = args.y_ran_residual_samples
    total_vel = vel_train_all.shape[0]
    sample_idx = torch.randperm(total_vel)[:n_samples]

    vel_s = vel_train_all[sample_idx].to(device)
    u0_s = UU0_train_all[sample_idx].to(device)
    freq_s = freq_train_all[sample_idx].to(device) if has_freq else None

    residual_map = model.compute_residual_map(
        vel_s, u0_s,
        freq_samples=freq_s,
        coarse_scale=args.y_ran_coarse_scale
    )  # [cz, cx]

    # 上采样到完整网格尺寸
    import torch.nn.functional as F
    residual_map_upsampled = F.interpolate(
        residual_map.unsqueeze(0).unsqueeze(0),
        size=(args.nz, args.nx),
        mode='bilinear',
        align_corners=True
    ).squeeze()

    # 混合：残差图 + 初始统计先验
    alpha = args.y_ran_residual_weight
    importance_map = alpha * residual_map_upsampled + (1 - alpha) * initial_importance_map

    # 重新采样
    y_ran = model.generate_y_ran(importance_map, args).to(device)

    pbar.write(f'>>> Epoch {i} | y_ran 已更新（残差引导）')
    del vel_s, u0_s, freq_s, residual_map
    torch.cuda.empty_cache()
```

#### 4.3.4 删除 per-batch 的 y_ran 生成

**删除**：第 111-112 行
```python
# 删除以下两行：
with torch.no_grad():
    y_ran = model.generate_structure_aware_y_ran(vel_batch, num_pts=900)
```

y_ran 现在从 epoch 级别的作用域中读取，无需在此处生成。

### 4.4 model/utils.py

在文件末尾新增函数：

```python
def compute_dataset_importance_map(vel_tensor, freq=5.0):
    """
    基于训练集速度场统计生成初始重要性概率图。
    结合速度梯度（结构复杂度）和波长因子（振荡频率）。

    Args:
        vel_tensor: [N, 1, Z, X] 训练集全量速度场
        freq: 参考频率 (Hz)

    Returns:
        importance_map: [Z, X] 归一化的重要性图 (值域 [0, 1])
    """
    vel = vel_tensor.squeeze(1)  # [N, Z, X]

    # --- 1. 平均梯度幅值（捕获地层界面）---
    grad_z = vel[:, 2:, 1:-1] - vel[:, :-2, 1:-1]
    grad_x = vel[:, 1:-1, 2:] - vel[:, 1:-1, :-2]
    grad_mag = torch.sqrt(grad_z ** 2 + grad_x ** 2 + 1e-8)
    grad_mag = F.pad(grad_mag, (1, 1, 1, 1), mode='replicate')
    mean_grad = grad_mag.mean(dim=0)  # [Z, X]

    # --- 2. 波长因子（捕获低速区需要更密采样）---
    # 波长 λ = v/f，低 v → 短波长 → 需要更密采样
    # 采样密度 ∝ 1/λ² ∝ (f/v)² ∝ 1/v²
    mean_vel = vel.mean(dim=0)  # [Z, X]
    wavelength_factor = 1.0 / (mean_vel ** 2 + 1e-8)
    # 归一化到与 mean_grad 相同量级
    wavelength_factor = wavelength_factor / wavelength_factor.mean() * mean_grad.mean()

    # --- 3. 组合 ---
    importance = mean_grad + wavelength_factor
    importance = importance / (importance.max() + 1e-8)

    return importance
```

### 4.5 model/train_distributed.py

需要同步以下两处改动（与 4.3 节一致）：

1. **训练循环前**：初始化 `initial_importance_map` 和 `y_ran`（仅在 rank 0 上采样后广播）
2. **Epoch 循环内**：每 K epoch 更新 `y_ran`（仅在 rank 0 上计算残差图后广播新的 y_ran）
3. **删除**：per-batch 的 `generate_structure_aware_y_ran` 调用

分布式额外注意事项：
- 残差评估只在 rank 0 执行，结果通过 `dist.broadcast` 同步到所有进程
- `y_ran` 在所有进程上必须完全一致（保证 DDP 梯度同步正确）
- `importance_map` 只需在 rank 0 上维护，采样生成 `y_ran` 后广播

```python
# 分布式同步伪代码
if rank == 0:
    residual_map = model.module.compute_residual_map(vel_s, u0_s, ...)
    importance_map = alpha * residual_map_upsampled + (1 - alpha) * initial_importance_map
    y_ran = model.module.generate_y_ran(importance_map, args)
else:
    y_ran = torch.empty(1, args.y_ran_num_pts, 2)
# 广播
dist.broadcast(y_ran, src=0)
```

---

## 5. 验证方法

### 5.1 基础验证（实施后必做）

- [ ] `y_ran` 形状确认为 `[1, 900, 2]`，非 `[B_v, 900, 2]`
- [ ] 训练第一个 epoch 正常完成，loss 量级与修改前一致
- [ ] 在 epoch 50 处观察到 `y_ran 已更新` 日志输出
- [ ] 更新后的 `importance_map` 可视化合理（残差集中区域概率更高）

### 5.2 可视化验证（可选）

在 epoch 50 更新后，将 `importance_map` 保存为 PNG：

```python
# 在 train.py 的更新逻辑中添加
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.imshow(importance_map.numpy(), cmap='hot', aspect='equal')
plt.colorbar(label='Sampling Importance')
plt.title(f'Importance Map at Epoch {i}')
plt.savefig(os.path.join(args.save_doc, f'importance_map_epoch_{i}.png'), dpi=150)
plt.close()
```

### 5.3 性能对比

| 指标 | 修改前（基线） | 修改后 |
|------|---------------|--------|
| PDE Loss 收敛速度 | 记录 epoch 500/1000/2000 的值 | 对比 |
| 验证集 R² | 记录最终值 | 对比 |
| 每 epoch 训练耗时 | 记录 | 对比（残差评估的开销） |
| 显存占用 | 记录峰值 | 对比 |
