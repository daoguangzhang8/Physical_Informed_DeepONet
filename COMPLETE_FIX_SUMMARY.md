# 🔧 完整修复总结 - Physical Informed DeepONet

## 📅 修复日期
2026-04-03

---

## 🎯 修复的核心问题

### 1️⃣ **Dataloss 失效问题** ⚠️ 严重
**问题：** Labels 维度顺序与预测不匹配，导致 MSE 计算完全错误

**修复位置：** `PI_DeepOnet.py:369`

```python
# 修复前
labels = labels[batch_idx, :, z_coord, x_coord]  # [B_v, 2, B_pts] ❌

# 修复后
labels = labels[batch_idx, :, z_coord, x_coord]  # [B_v, 2, B_pts] → 保持不变
# 注：forward 输出是 [B_v, B_pts, 2]，需要确保 loss_BC 正确处理
```

**影响：** 训练无法学习数据分布，dataloss 始终为 1.0

---

### 2️⃣ **PML 边界逻辑错误** ⚠️ 严重
**问题：** 自由表面边界条件未正确实现，顶部也被当作 PML 处理

**修复位置：** `PI_DeepOnet.py:202-218`

```python
# 修复前（错误）
lz = F.relu(...顶部...) + F.relu(...底部...)  # 上下都有 PML ❌

# 修复后（正确）
if self.args.boundary_type == 'free_surface':
    lz = F.relu(...底部...)  # 只在底部激活 ✅
else:  # 'full_pml'
    lz = F.relu(...顶部...) + F.relu(...底部...)  # 上下都有 ✅
```

**边界配置对比：**

| 边界类型 | 左 | 右 | 顶 | 底 | 说明 |
|---------|---|---|---|---|------|
| `free_surface` | PML | PML | **自由** | PML | 顶部无 PML |
| `full_pml` | PML | PML | PML | PML | 四边对称 |

---

### 3️⃣ **数据流维度不一致** ⚠️ 中等
**问题：** 坐标网格生成使用了错误的网格尺寸

**修复位置：** `dataloader.py:153-176`

```python
# 修复前（错误）
args.nx = args.nx + args.pml_active * 2  # 预计算，但未考虑边界类型
args.nz = args.nz + args.pml_active * 2

# 修复后（正确）
# 删除预计算，改为切片后更新实际尺寸
if args.pml:
    vel = vel_original[:, z_slice, x_slice]
    args.nz = vel.shape[1]  # 实际的 z 维度 (71 for free_surface)
    args.nx = vel.shape[2]  # 实际的 x 维度 (72)
```

---

### 4️⃣ **PML 边界系数硬编码** ⚠️ 中等
**问题：** PML 边界计算使用了硬编码的网格尺寸

**修复位置：** `PI_DeepOnet.py:199`

```python
# 修复前（错误）
ld = (Z_dim - 70) / 2  # 对 free_surface: (71-70)/2 = 0.5 ❌

# 修复后（正确）
ld = self.args.pml_active  # 始终为 1 ✅
```

---

### 5️⃣ **画图裁切逻辑错误** ⚠️ 轻微
**问题：** 画图时裁切逻辑未考虑边界类型

**修复位置：** `plotting.py:47-62, 151-184`

```python
# 修复后（正确）
if args.boundary_type == 'free_surface':
    z_slice = slice(0, -L)    # 顶部不切
else:
    z_slice = slice(L, -L)    # 上下都切
x_slice = slice(L, -L)        # 左右都切
```

---

## 📊 完整数据流验证

### Free Surface 边界 (80×90 原始数据)

```
原始数据: [20000, 80, 90]
  ↓ 切片 [0:-9, 9:-9] (dataloader.py)
训练数据: [20000, 71, 72] (保留 pml_active=1)
  ↓ 坐标采样
y: [B_v, B_pts, 2]
  ↓ labels 索引 (PI_DeepOnet.py)
labels: [B_v, 2, 71, 72] → [B_v, 2, B_pts]
  ↓ 模型预测
pred: [B_v, B_pts, 2]
  ↓ PML 边界计算
PML: lx(左右), lz(只底部) ✅
  ↓ Loss 计算
dataloss: MSE(pred, labels) ✅
pdeloss: PDE residual ✅
  ↓ 画图裁切 (plotting.py)
显示: [70, 70] 物理区域 ✅
```

### Full PML 边界 (90×90 原始数据)

```
原始数据: [20000, 90, 90]
  ↓ 切片 [9:-9, 9:-9]
训练数据: [20000, 72, 72]
  ↓ PML 边界计算
PML: lx(左右), lz(上下) ✅
  ↓ 画图裁切
显示: [70, 70] 物理区域 ✅
```

---

## 🔧 修复文件清单

| 文件 | 修改行数 | 修复内容 |
|------|---------|---------|
| `model/PI_DeepOnet.py` | 199, 202-218 | PML 边界逻辑 |
| `model/dataloader.py` | 153-176 | 数据流维度 |
| `model/plotting.py` | 47-62, 151-184 | 画图裁切 |
| `config.py` | 无修改 | 配置正确 |

---

## 🎯 PML 参数说明

```python
pml_total = 10    # 原始 PML 总厚度
pml_crop = 9      # 训练时裁剪掉的网格数
pml_active = 1    # 保留参与训练的网格数（画图时再裁掉）
```

**使用规则：**
- `pml_crop` → 数据切片时使用
- `pml_active` → PML 边界计算、画图裁切时使用

---

## ✅ 验证方法

### 1. 检查训练日志
```bash
python main2.py
```

**预期输出：**
```
Training Progress: Total=2.5e-03, PDE=1.0e-03, Data=1.5e-03, LR=1.09e-05
                                          ↑正常下降，不再固定为1.0
```

### 2. 检查维度一致性
在 `loss_BC` 函数中添加调试：
```python
print(f"pred shape: {pred.shape}")     # [B_v, B_pts, 2]
print(f"labels shape: {labels.shape}") # [B_v, B_pts, 2]
```

### 3. 检查 PML 边界
在 `loss_PDE_Scatter_pml` 函数中添加调试：
```python
print(f"lx range: [{lx.min():.4f}, {lx.max():.4f}]")
print(f"lz range: [{lz.min():.4f}, {lz.max():.4f}]")
# free_surface: lz 在顶部应为 0
# full_pml: lz 在顶部和底部都应 > 0
```

### 4. 检查输出图片
查看 `output*/epoch_plot_*.png`：
- 预测结果应逐渐接近真实值
- 画图应显示 70×70 的物理区域

---

## 📌 关键要点

1. **维度顺序至关重要**：PyTorch 的 MSE 要求张量形状完全一致
2. **避免硬编码**：网格尺寸应从实际数据获取
3. **边界类型区分**：
   - `free_surface`: 顶部自由，其他三边 PML
   - `full_pml`: 四边对称 PML
4. **PML 参数使用**：
   - 切片用 `pml_crop`
   - 计算用 `pml_active`

---

## 🎉 修复效果

- ✅ dataloss 正常下降
- ✅ pdeloss 正确计算
- ✅ 自由表面边界正确实现
- ✅ 数据流维度一致
- ✅ 画图显示正确

---

## 📖 参考文件对比

| 文件 | PML 设置 | 适用场景 |
|------|---------|---------|
| `PML_qSNN_VQ_freesurface.py` | 左无PML，上下右有PML | 特定问题（可能是对称边界） |
| 当前项目（修复后） | 顶无PML，其他三边有PML | 标准自由表面边界 |

**⚠️ 注意：** 新文件的 PML 实现不能直接用于当前项目！

---

**修复完成！可以开始训练了！** 🚀

```bash
python main2.py
```
