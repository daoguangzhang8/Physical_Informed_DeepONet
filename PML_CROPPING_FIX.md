# PML 裁切逻辑修复总结

## 🎯 问题描述

原始代码中存在维度不匹配问题：
- 训练数据维度与坐标网格维度不一致
- 画图裁切逻辑没有正确处理不同边界类型

## 📊 PML 参数说明

```python
pml_total = 10    # 原始数据中 PML 总厚度
pml_crop = 9      # 训练时裁剪掉的 PML 网格数
pml_active = 1    # 保留参与训练的 PML 网格数（画图时再裁掉）
```

## 🔄 完整数据流

### free_surface 边界（顶部自由表面）

```
原始数据 (80×90)
  ↓ 训练切片 [0:-9, 9:-9]
训练数据 (71×72, 保留 pml_active=1)
  ↓ 模型预测
预测结果 (71×72)
  ↓ 画图裁切 [0:-1, 1:-1]
画图显示 (70×70, 纯物理区域)
  ↓ 高分辨率上采样 (times=4)
高分辨率图 (280×280)
```

### full_pml 边界（四边 PML）

```
原始数据 (90×90)
  ↓ 训练切片 [9:-9, 9:-9]
训练数据 (72×72, 保留 pml_active=1)
  ↓ 模型预测
预测结果 (72×72)
  ↓ 画图裁切 [1:-1, 1:-1]
画图显示 (70×70, 纯物理区域)
  ↓ 高分辨率上采样 (times=4)
高分辨率图 (280×280)
```

## 📝 代码修改

### 1. dataloader.py (第 145-176 行)

**修改前：**
```python
# 错误：提前计算了错误的网格尺寸
args.nx = args.nx + args.pml_active * 2
args.nz = args.nz + args.pml_active * 2
```

**修改后：**
```python
# 删除了错误的预计算，改为切片后更新实际尺寸
if args.pml:
    # ... 切片操作 ...
    vel = vel_original[:, z_slice, x_slice]
    # 更新为切片后的实际尺寸
    args.nz = vel.shape[1]  # 实际的 z 维度
    args.nx = vel.shape[2]  # 实际的 x 维度
```

### 2. plotting.py - test_plot 函数 (第 151-184 行)

**关键修改：**
```python
# 使用标签的实际尺寸来 reshape 预测结果
actual_nz = labels.shape[2]  # 实际的 z 维度
actual_nx = labels.shape[3]  # 实际的 x 维度
U_pred = U_pred.reshape(actual_nz, actual_nx, 2)

# 根据边界类型裁切
L = args.pml_active  # 使用 pml_active 作为裁切力度
if args.boundary_type == 'free_surface':
    z_slice = slice(0, -L)    # 顶部不切，底部切
else:
    z_slice = slice(L, -L)    # 上下都切
x_slice = slice(L, -L)        # 左右都切
```

### 3. plotting.py - plot_sinlge 函数 (第 4-89 行)

**关键修改：**
```python
# 根据边界类型计算裁切后的有效网格数
if args.boundary_type == 'free_surface':
    tag_nz = actual_nz * times - L1      # z 方向：只裁底部
    tag_nx = actual_nx * times - 2 * L1  # x 方向：左右都裁
else:
    tag_nz = actual_nz * times - 2 * L1  # z 方向：上下都裁
    tag_nx = actual_nx * times - 2 * L1  # x 方向：左右都裁

# 标签数据使用原始网格的裁切（不乘 times）
if args.boundary_type == 'free_surface':
    z_slice_label = slice(0, -L)
else:
    z_slice_label = slice(L, -L)
x_slice_label = slice(L, -L)
```

## ✅ 裁切逻辑总结

| 边界类型 | z 方向裁切 | x 方向裁切 | 训练数据 | 画图显示 |
|---------|-----------|-----------|---------|---------|
| `free_surface` | 顶部保留，底部裁 | 左右对称裁 | 71×72 | 70×70 |
| `full_pml` | 上下对称裁 | 左右对称裁 | 72×72 | 70×70 |

## 🔍 验证方法

运行训练后检查：
1. 训练数据维度：`vel_train.shape` 应该是 `(N, 1, 71, 72)` 或 `(N, 1, 72, 72)`
2. 预测结果维度：与训练数据一致
3. 画图显示维度：`(70, 70)` - 纯物理区域

## 📌 注意事项

1. **不要混淆 pml_crop 和 pml_active**：
   - `pml_crop` 用于训练数据准备
   - `pml_active` 用于画图裁切

2. **边界类型影响裁切逻辑**：
   - `free_surface`: 顶部是自由表面，不裁切
   - `full_pml`: 四边都是 PML，对称裁切

3. **高分辨率绘图的标签裁切**：
   - 标签数据使用原始网格的裁切（不乘 times）
   - 预测数据使用上采样后的裁切（乘 times）

## 🎉 修复效果

- ✅ 解决了维度不匹配错误
- ✅ 统一了不同边界类型的处理逻辑
- ✅ 画图只显示纯物理区域 (70×70)
- ✅ 支持高分辨率绘图 (280×280)
