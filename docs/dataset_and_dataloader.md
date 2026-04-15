# 数据集构成与 DataLoader 构建说明

---

## 1. 原始数据文件

| 数据类型 | 文件名 | 形状 | 说明 |
|---------|--------|------|------|
| 速度场 | `freesurface_velocity_freq3to20_5sources_160_180_pml20_n1.npy` | `[N_vel, NZ_raw, NX_raw]` | N_vel 个速度模型，含 PML 边界 |
| 背景波场 | `freesurface_backgroundfield_...n1.npy` | `[N_vel*5, 2, NZ_raw, NX_raw]` | 5 个震源交织排列，通道 0/1 为实部/虚部 |
| 总波场 | `freesurface_wavefield_...n1.npy` | `[N_vel*5, 2, NZ_raw, NX_raw]` | 同上 |
| 频率 | `freesurface_freq_used_...n1.npy` | `[N_vel]` | 每个速度模型对应一个频率值 |

---

## 2. PML 边界裁剪

原始数据包含 PML 吸收层（20 网格），训练时裁剪掉部分 PML 区域：

| 参数 | 值 | 含义 |
|------|-----|------|
| `pml_total` | 20 | PML 总厚度（网格数） |
| `pml_crop` | 15 | 裁剪掉的 PML 网格数 |
| `pml_active` | 5 | 保留参与训练的 PML 网格数 |

**裁剪方式（按 `boundary_type` 区分）：**

- **`free_surface`**：顶部自由表面不裁剪，底部裁剪 15 格；左右各裁剪 15 格
  - z 方向：`[0 : -15]`
  - x 方向：`[15 : -15]`
  - 结果：160×180 → 145×150

- **`full_pml`**：四边各裁剪 15 格
  - z 方向：`[15 : -15]`
  - x 方向：`[15 : -15]`
  - 结果：160×180 → 130×150

---

## 3. 震源拆分

原始波场数据中 5 个震源按速度模型数 `N_vel` 交织排列，需按索引拆分：

```
UU_loc[i] = UU[i * N_vel : (i+1) * N_vel, :, :, :]    # i = 0,1,2,3,4
UU0_loc[i] = UU0[i * N_vel : (i+1) * N_vel, :, :, :]
```

- `source_list = [0, 1, 2, 3, 4]` 控制训练中使用哪些震源
- 震源坐标为固定物理位置，由 `source_coords` 列表定义

---

## 4. 训练集 / 验证集划分

### 4.1 速度模型划分

```
idx = np.random.choice(N_vel_total, nvel_train, replace=False)  # 随机选取训练集
remaining = 其余速度模型 → 验证集
valid_num = int(valid_rate * nvel_train) + 1                    # 默认 10%+1
```

| 参数 | 训练集 | 验证集 |
|------|--------|--------|
| 速度模型数 | `nvel_train` (如 1) | `valid_num` (如 2) |
| 震源数 | `len(source_list)` (如 5) | 同训练集 |
| **总样本数** | `nvel_train × len(source_list)` | `valid_num × len(source_list)` |

### 4.2 多震源拼接

对每个激活的震源，将速度模型复制、波场按震源索引提取，沿 batch 维拼接：

```
vel_out    = [vel_src0, vel_src1, ..., vel_src4]  → cat(dim=0)
UU0_out    = [UU0_src0, UU0_src1, ..., UU0_src4] → cat(dim=0)
labels_out = [(UU-UU0)_src0, ..., (UU-UU0)_src4]  → cat(dim=0)
```

### 4.3 标签定义

**标签 = 总波场 − 背景波场 = 散射场残差**

```
labels = UU - UU0    # shape: [N_samples, 2, NZ, NX]
```

通道 0 = 实部残差，通道 1 = 虚部残差。

### 4.4 速度场归一化

```
vel = vel / 1000.0    # 将速度值从 m/s 量级归一化到 1 量级
```

---

## 5. 坐标点采样

### 5.1 训练集坐标

| 模式 | 参数 | 采样量 | 说明 |
|------|------|--------|------|
| `full_grid` | — | NZ × NX | 全网格所有点 |
| `halton` | `halton_sample_ratio = 0.5` | NZ × NX × 0.5 | Halton 准随机采样，覆盖更均匀 |

- 坐标值 = 网格索引 × `dh`（物理坐标，单位：米）
- `dh = 20`（网格间距 20m）
- 训练/验证共用同一份坐标点

### 5.2 PDE 自适应采样（训练中动态生成）

每个 velocity batch 在训练循环中额外生成 **900 个结构感知采样点**：

- 50% 分布在速度梯度边界附近（检测速度场的空间梯度）
- 50% 分布在自由表面附近
- 与数据坐标拼接后一起前向传播，一次计算 Data Loss + PDE Loss

---

## 6. DataLoader 构建

### 6.1 训练阶段 DataLoader

共构建 6 个 DataLoader：

| 名称 | 数据内容 | batch_size | shuffle | 用途 |
|------|---------|-----------|---------|------|
| `train` | `(vel, UU0, labels[, freq])` | `batch_size_v = 1` | True | 遍历速度模型 |
| `train_y` | `(y_coords,)` | `batch_size = 800` | True | 遍历空间坐标 |
| `valid` | `(vel, UU0, labels[, freq])` | `valid_batch_size_v = 6` | True | 验证集速度模型 |
| `valid_y` | `(y_coords,)` | `valid_batch_size = 350` | True | 验证集坐标 |
| `pred` | `(y_full_grid,)` | `batch_size = 800` | False | 验证/绘图用全网格 |
| `test` | `(y_full_grid,)` | `batch_size = 800` | False | 训练集绘图 |

**嵌套迭代逻辑：**

```
for vel_batch, UU0_batch, labels_batch in train:          # 外层: 速度模型
    y_ran = model.generate_structure_aware_y_ran(...)      # 生成自适应采样点
    for y_batch in train_y:                                # 内层: 空间坐标
        y_combined = cat([y_batch, y_ran])                 # 拼接数据点+PDE点
        loss = model.loss(vel, y_combined, UU0, labels)    # 单次前向 → Data+PDE Loss
        loss.backward()                                    # 梯度累积
```

**梯度累积**：`accumulation_steps = 4`，每 4 个 coord batch 更新一次参数。

### 6.2 测试阶段 DataLoader

测试时不使用 DataLoader 加载速度/波场，而是直接操作 Tensor：

| 数据来源 | 说明 |
|---------|------|
| **训练集采样** | 从 `Training_data()` 的输出中提取 1 个速度模型 × 多震源 |
| **外部测试集** | Marmousi / Overthrust 等独立速度模型，通过 `prepare_external_val_dataset()` 加载 |
| **坐标 DataLoader** | 仅构建坐标的 DataLoader（全网格，`shuffle=False`），用于逐 batch 推理 |

外部数据集通过 `ext_val_datasets` 字典配置：

```python
ext_val_datasets = {
    'Marmousi':   {'prefix': 'marmousi_',    'loc_target': [2]},
    'Overthrust': {'prefix': 'overthrust_',  'loc_target': [2]},
}
```

---

## 7. 关键参数汇总

| 类别 | 参数 | 值 | 说明 |
|------|------|-----|------|
| **物理网格** | `dh` | 20 m | 网格间距 |
| | `nz × nx` | 145 × 150 (裁剪后) | 训练网格尺寸 |
| **采样** | `sampling_mode` | `halton` | 采样模式 |
| | `halton_sample_ratio` | 0.5 | 采样比例 |
| **批处理** | `batch_size_v` | 1 | 速度模型批次 |
| | `batch_size` | 800 | 坐标点批次 |
| | `accumulation_steps` | 4 | 梯度累积步数 |
| **划分** | `nvel_train` | 1 | 训练速度模型数 |
| | `valid_rate` | 0.1 | 验证集比例 |
| **训练** | `NIter` | 5001 | 总 epoch 数 |
| | `lr` | 1e-4 | 初始学习率 |
| | `optimizer` | Adam | 优化器 |
| **Loss** | `a / b / c` | 1 / 1 / 0 | Data / PDE / Reg 权重 |

---

## 8. 数据流全链路图

```
.npy 原始文件
    │
    ▼
PML 裁剪 (boundary_type 决定裁切策略)
    │
    ▼
震源拆分 (5 sources → UU_loc[0..4], UU0_loc[0..4])
    │
    ▼
训练/验证划分 (random seed=1, train:nvel_train, valid:10%)
    │
    ├─ 速度场: vel / 1000.0
    ├─ 标签: labels = UU - UU0 (散射场残差)
    └─ 坐标: Halton采样 或 全网格, 物理坐标 = index × dh
    │
    ▼
TensorDataset → DataLoader
    │
    ├─ train:      (vel, UU0, labels, freq) batch_size_v=1
    ├─ train_y:    (y_coords,)              batch_size=800
    ├─ valid:      (vel, UU0, labels, freq) batch_size_v=6
    └─ valid_y:    (y_coords,)              batch_size=350
    │
    ▼
嵌套循环: 外层遍历速度模型 → 内层遍历坐标 → 拼接自适应PDE采样点 → 单次前向 → 联合Loss
```
