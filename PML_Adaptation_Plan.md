# DeepONet 项目 PML 边界适配方案

> 创建时间: 2026-04-02
> 状态: 待实施

---

## 一、项目结构概述

### 1. 相关项目

| 项目 | 路径 | 功能 |
|------|------|------|
| data | `/home/zhangdaoguang/Code/data/` | 波场数据生成（有限差分法求解 Helmholtz 方程） |
| DeepONet | `/home/zhangdaoguang/Code/DeepONet/` | PI-DeepONet 模型训练 |
| Physical_Informed_DeepONet | `/home/zhangdaoguang/Code/Physical_Informed_DeepONet/` | 改进版 PI-DeepONet |

### 2. DeepONet 核心文件

| 文件 | 功能 |
|------|------|
| `config.py` | 超参数配置 |
| `model/PI_DeepOnet.py` | 模型架构：FNO(Branch) + FiLM(Trunk) |
| `model/dataloader.py` | 数据加载与预处理 |
| `model/ploting.py` | 可视化与绘图 |
| `model/train.py` | 训练循环 |

### 3. data 项目 PML 配置方式

**modeling.py 中的配置**:
```python
n = np.array([nz, nx])  # 原始模型尺寸 [70, 70]
n_pml = np.array([[Lpml, 0],  # [左侧PML, 顶部PML]
                  [Lpml, Lpml]])  # [右侧PML, 底部PML]
```

**PML 矩阵含义**:
```
n_pml[0, 0] = 左侧 PML 层数
n_pml[0, 1] = 顶部 PML 层数 (0 = 自由表面)
n_pml[1, 0] = 右侧 PML 层数
n_pml[1, 1] = 底部 PML 层数
```

---

## 二、问题分析

### 1. 当前代码的硬编码假设

**PI_DeepOnet.py 第 211-214 行**:
```python
ld = (Z_dim - 70) / 2

lx = F.relu(((ld - 0.5) * 40 - xx) / ((ld - 0.5) * 40)) + F.relu((xx - (69.5 + ld) * 40) / ((ld - 0.5) * 40))
lz = F.relu(((ld - 0.5) * 40 - zz) / ((ld - 0.5) * 40)) + F.relu((zz - (69.5 + ld) * 40) / ((ld - 0.5) * 40))
```

**问题**:
1. 硬编码 `70` 作为原始网格尺寸
2. 假设 PML 在四边**对称分布**
3. 无法处理自由表面（顶部无 PML）的情况

### 2. dataloader.py 硬编码切片

**第 162-166 行**:
```python
if args.pml:
    Lpml = args.Lpml
    vel = vel_original[:, Lpml:-Lpml, Lpml:-Lpml]
    UU0 = UU0_original[:, :, Lpml:-Lpml, Lpml:-Lpml]
    UU = UU_original[:, :, Lpml:-Lpml, Lpml:-Lpml]
```
假设四边对称切片。

### 3. ploting.py 硬编码切片

**第 42-43, 48-49, 142-143, 148-149 行**:
```python
U_pred_real = U_pred[L:-L, L:-L, 0]
U_pred_imag = U_pred[L:-L, L:-L, 1]
U_ref_real = U_ref[0, L:-L, L:-L][:, :]
U_ref_imag = U_ref[1, L:-L, L:-L][:, :]
```
假设四边对称裁切。

---

## 三、Shape 计算

### 边界类型对比

| 边界类型 | 原始数据 | 网络输入切片 | 网络输入 Shape | 有效输出裁切 | 有效输出 Shape |
|----------|----------|--------------|----------------|--------------|----------------|
| 全 PML | 90×90 | `[9:-9, 9:-9]` | 72×72 | `[1:-1, 1:-1]` | 70×70 |
| 自由表面 | 80×90 | `[0:-9, 9:-9]` | 71×72 | `[0:-1, 1:-1]` | 70×70 |

### 自由表面 PML 配置

```
左=9, 顶=0 (自由表面), 右=9, 底=9
```

**切片逻辑**:
- z 方向 (垂直): `[0:-9]` → 顶部不切，底部切 9 层 → 71 点
- x 方向 (水平): `[9:-9]` → 左右各切 9 层 → 72 点

---

## 四、修改方案

### 1. config.py 修改

**位置**: 第 55-67 行之后添加

```python
    # ==========================================
    # 6. 物理网格与边界条件 (Physical Grid & PML Boundaries)
    # ==========================================
    nx = 70                                   # 物理模型 x 方向网格数 (不含外延 PML)
    nz = 70                                   # 物理模型 z 方向网格数 (不含外延 PML)
    pml = True                                # 是否启用 PML 吸收边界
    Lpml = 9                                  # 实际截取的 PML 层数
    LD = 10 - Lpml                            # 边界补偿计算参数

    # 边界类型配置
    # 'full_pml': 四边 PML 吸收边界，原始数据 90×90 → 网络输入 72×72
    # 'free_surface': 顶部自由表面 + 其他三边 PML，原始数据 80×90 → 网络输入 71×72
    boundary_type = 'free_surface'            # 根据实际数据选择
```

---

### 2. dataloader.py 修改

#### 2.1 `prepare_training_dataloaders` 函数 (第 156-166 行)

**修改前**:
```python
if args.pml:
    Lpml = args.Lpml
    vel = vel_original[:, Lpml:-Lpml, Lpml:-Lpml]
    UU0 = UU0_original[:, :, Lpml:-Lpml, Lpml:-Lpml]
    UU = UU_original[:, :, Lpml:-Lpml, Lpml:-Lpml]
```

**修改后**:
```python
if args.pml:
    Lpml = args.Lpml
    # 根据边界类型确定切片范围
    if args.boundary_type == 'free_surface':
        z_slice = slice(0, -Lpml)      # 顶部不切，底部切 Lpml
    else:  # 'full_pml'
        z_slice = slice(Lpml, -Lpml)   # 上下都切 Lpml

    x_slice = slice(Lpml, -Lpml)       # 左右都切 Lpml

    vel = vel_original[:, z_slice, x_slice]
    UU0 = UU0_original[:, :, z_slice, x_slice]
    UU = UU_original[:, :, z_slice, x_slice]
else:
    vel, UU0, UU = vel_original, UU0_original, UU_original
```

#### 2.2 `prepare_external_val_dataset` 函数 (第 223-232 行)

**修改前**:
```python
if args.pml:
    Lpml = args.Lpml
    vel_ext = vel_ext.unsqueeze(0)[:, Lpml:-Lpml, Lpml:-Lpml]
    UU0_ext = UU0_ext[:, :, Lpml:-Lpml, Lpml:-Lpml]
    UU_ext = UU_ext[:, :, Lpml:-Lpml, Lpml:-Lpml]
```

**修改后**:
```python
if args.pml:
    Lpml = args.Lpml
    # 根据边界类型确定切片范围
    if args.boundary_type == 'free_surface':
        z_slice = slice(0, -Lpml)
    else:
        z_slice = slice(Lpml, -Lpml)
    x_slice = slice(Lpml, -Lpml)

    vel_ext = vel_ext.unsqueeze(0)[:, z_slice, x_slice]
    UU0_ext = UU0_ext[:, :, z_slice, x_slice]
    UU_ext = UU_ext[:, :, z_slice, x_slice]
else:
    vel_ext = vel_ext.unsqueeze(0)
```

---

### 3. ploting.py 修改

#### 3.1 `plot_sinlge` 函数 (第 42-49 行)

**修改前**:
```python
U_pred_test = U_pred_test.reshape(Nz, Nx, 2)
U_pred_real_test = U_pred_test[L1:-L1, L1:-L1, 0]
U_pred_imag_test = U_pred_test[L1:-L1, L1:-L1, 1]

y_pred_np = y_pred.detach().cpu().numpy()
labels_pred_np = labels_pred.detach().cpu().numpy()
U_test = labels_pred_np[0,:,:,:]
U_test_real = U_test[0, L:-L, L:-L][:, :]
U_test_imag = U_test[1, L:-L, L:-L][:, :]
```

**修改后**:
```python
U_pred_test = U_pred_test.reshape(Nz, Nx, 2)

# 根据边界类型确定切片范围
if args.boundary_type == 'free_surface':
    z_slice = slice(0, -L1)    # 顶部不切
else:
    z_slice = slice(L1, -L1)   # 上下都切

x_slice = slice(L1, -L1)       # 左右都切

U_pred_real_test = U_pred_test[z_slice, x_slice, 0]
U_pred_imag_test = U_pred_test[z_slice, x_slice, 1]

y_pred_np = y_pred.detach().cpu().numpy()
labels_pred_np = labels_pred.detach().cpu().numpy()
U_test = labels_pred_np[0,:,:,:]
U_test_real = U_test[0, z_slice, x_slice]
U_test_imag = U_test[1, z_slice, x_slice]
```

#### 3.2 `test_plot` 函数 (第 141-149 行)

**修改前**:
```python
U_pred = U_pred.reshape(args.nz, args.nx, 2)
U_pred_real = U_pred[L:-L, L:-L, 0]
U_pred_imag = U_pred[L:-L, L:-L, 1]

labels_np = labels.detach().cpu().numpy()
U_ref = labels_np[0,:,:,:]
U_ref_real = U_ref[0, L:-L, L:-L][:, :]
U_ref_imag = U_ref[1, L:-L, L:-L][:, :]
```

**修改后**:
```python
U_pred = U_pred.reshape(args.nz, args.nx, 2)

# 根据边界类型确定切片范围
if args.boundary_type == 'free_surface':
    z_slice = slice(0, -L)    # 顶部不切
else:
    z_slice = slice(L, -L)    # 上下都切

x_slice = slice(L, -L)        # 左右都切

U_pred_real = U_pred[z_slice, x_slice, 0]
U_pred_imag = U_pred[z_slice, x_slice, 1]

labels_np = labels.detach().cpu().numpy()
U_ref = labels_np[0,:,:,:]
U_ref_real = U_ref[0, z_slice, x_slice]
U_ref_imag = U_ref[1, z_slice, x_slice]
```

---

## 五、修改文件清单

| 文件 | 修改位置 | 修改内容 |
|------|----------|----------|
| `config.py` | 第 55-67 行后 | 添加 `boundary_type` 参数 |
| `model/dataloader.py` | 156-166 行 | 根据 `boundary_type` 调整切片 |
| `model/dataloader.py` | 223-232 行 | 根据 `boundary_type` 调整切片 |
| `model/ploting.py` | 42-49 行 | 根据 `boundary_type` 调整裁切 |
| `model/ploting.py` | 141-149 行 | 根据 `boundary_type` 调整裁切 |

---

## 六、待办事项

- [ ] 执行 config.py 修改
- [ ] 执行 dataloader.py 修改
- [ ] 执行 ploting.py 修改
- [ ] PDE Loss 部分修改（待实例代码）
- [ ] 测试验证

---

## 七、注意事项

1. **数据一致性**: 确保 `config.py` 中的 `boundary_type` 与实际生成数据时使用的边界配置一致

2. **PDE Loss 部分**: 暂不修改，待后续根据实例代码处理

3. **网格尺寸**: 
   - 全 PML: 网络输入 72×72，有效输出 70×70
   - 自由表面: 网络输入 71×72，有效输出 70×70

4. **数据文件命名**: 注意数据文件名 `velocity_data_70_70_n1.npy` 中的 70×70 是物理区域尺寸，实际存储的可能是扩展后的尺寸
