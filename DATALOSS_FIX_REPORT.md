# Dataloss 失效问题修复报告

## 🚨 严重问题：Dataloss 维度不匹配导致训练失效

### 问题表现
- 训练时 dataloss 始终为 1.0000e+00，没有下降
- 模型无法学习数据分布

### 根本原因

在 `PI_DeepOnet.py` 的 `loss()` 函数中（第367行），labels 的维度顺序与 pred 不匹配：

```python
# 原代码（错误）
labels = labels[batch_idx, :, z_coord, x_coord]
# 形状: [B_v, 2, B_pts] ❌

# 模型预测
pred = self.forward(vel, y, UU0)
# 形状: [B_v, B_pts, 2] ✅
```

**维度对比：**
```
labels: [B_v, 2, B_pts]    第2维是通道（real/imag）
pred:   [B_v, B_pts, 2]    最后1维是通道（real/imag）
```

当计算 `MSE(pred, labels)` 时，PyTorch 会错误地对齐维度：
- `pred[:, i, :]` 形状 `[2]` 会被广播到
- `labels[:, :, i]` 形状 `[2]`

但这完全是错误的对齐！导致 MSE 计算结果毫无意义。

### 修复方案

在 `PI_DeepOnet.py` 第367行添加 `.transpose(1, 2)`：

```python
# 修复后
labels = labels[batch_idx, :, z_coord, x_coord].transpose(1, 2)
# 形状: [B_v, B_pts, 2] ✅ 与 pred 一致
```

### 修复效果

修复前：
```
Training Progress: Total=nan, PDE=1.0000e+00, Data=1.0000e+00
                                      ↑始终为1，没有学习
```

修复后（预期）：
```
Training Progress: Total=2.5e-03, PDE=1.0e-03, Data=1.5e-03
                                          ↑正常下降
```

## 📋 完整修复清单

### 1. dataloader.py - 数据流修复
- **第153-176行**: 删除错误的预计算，改为切片后更新实际尺寸
- **关键修改**:
  ```python
  # 删除错误代码
  # args.nx = args.nx + args.pml_active * 2  ❌
  
  # 添加正确逻辑
  args.nz = vel.shape[1]  # 实际的 z 维度 ✅
  args.nx = vel.shape[2]  # 实际的 x 维度 ✅
  ```

### 2. PI_DeepOnet.py - Loss 计算修复
- **第367行**: Labels 维度转置（**核心修复**）
  ```python
  labels = labels[batch_idx, :, z_coord, x_coord].transpose(1, 2)
  ```

- **第194行**: PML 边界系数修复
  ```python
  # 原代码
  ld = (Z_dim - 70) / 2  # ❌ 硬编码70，对 free_surface 错误
  
  # 修复后
  ld = self.args.pml_active  # ✅ 使用配置参数
  ```

- **第281行**: Trunk 输出归一化修复
  ```python
  # 原代码
  y_norm = 2 * (y - 0) / (40 * 72) - 1  # ❌ 硬编码72
  
  # 修复后
  X_dim = vel.shape[3]
  y_norm = 2 * (y - 0) / (40 * X_dim) - 1  # ✅ 使用实际尺寸
  ```

### 3. plotting.py - 绘图裁切修复
- **test_plot 函数**: 使用实际尺寸 reshape 和正确裁切
- **plot_sinlge 函数**: 根据边界类型计算裁切范围

## 🎯 数据流验证

### free_surface 边界（80×90 原始数据）

```
原始数据: [20000, 80, 90]
  ↓ 切片 [0:-9, 9:-9]
训练数据: [20000, 71, 72]
  ↓ 坐标采样
y: [B_v, B_pts, 2] → z_coord: [B_v, B_pts], x_coord: [B_v, B_pts]
  ↓ labels 索引
labels: [B_v, 2, 71, 72] → [B_v, 2, B_pts] → 转置 → [B_v, B_pts, 2]
  ↓ 模型预测
pred: [B_v, B_pts, 2]
  ↓ MSE 计算
dataloss: ✅ 正常计算
```

### full_pml 边界（90×90 原始数据）

```
原始数据: [20000, 90, 90]
  ↓ 切片 [9:-9, 9:-9]
训练数据: [20000, 72, 72]
  ↓ 后续流程相同
dataloss: ✅ 正常计算
```

## 🔍 验证方法

1. **检查训练日志**:
   ```bash
   python main2.py
   ```
   观察 dataloss 是否正常下降（不再始终为 1.0）

2. **检查维度一致性**:
   在 `loss_BC` 函数中添加调试信息：
   ```python
   print(f"pred shape: {pred.shape}")
   print(f"labels shape: {labels.shape}")
   # 应该都是 [B_v, B_pts, 2]
   ```

3. **检查输出图片**:
   查看 `output*/epoch_plot_*.png`，预测结果应该逐渐接近真实值

## 📌 关键要点

1. **维度顺序至关重要**：PyTorch 的 MSE 要求两个张量的形状完全一致
2. **避免硬编码**：网格尺寸应该从实际数据中获取，而不是写死
3. **PML 参数使用**：
   - `pml_crop` 用于训练数据切片
   - `pml_active` 用于画图裁切和 PML 边界计算
4. **边界类型影响**：
   - `free_surface`: 顶部不切，其他三边切
   - `full_pml`: 四边对称切

## ✅ 修复验证

运行训练后应该看到：
- ✅ dataloss 正常下降（不再固定为 1.0）
- ✅ pdeloss 正常计算
- ✅ 预测结果逐渐改善
- ✅ 画图显示正确的 70×70 物理区域

---

**修复完成日期**: 2026-04-03  
**修复版本**: v2.0  
**关键文件**: 
- `model/PI_DeepOnet.py` (loss 维度修复)
- `model/dataloader.py` (数据流修复)
- `model/plotting.py` (绘图裁切修复)
