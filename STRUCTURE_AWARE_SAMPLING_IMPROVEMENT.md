# 结构感知自适应采样改进

## 📅 改进日期
2026-04-03

---

## 🎯 改进目标

增强 `generate_structure_aware_y_ran` 函数，使其能够更好地采样地表附近的物理场特征。

---

## 🔧 改进内容

### 原始实现
```python
# 70% 结构边界点（基于速度场梯度）
# 30% 全局均匀分布点
```

**问题：**
- 全局均匀分布点可能在地表附近采样不足
- 对于自由表面边界条件，地表附近的物理场特征非常重要

### 改进后实现
```python
# 50% 结构边界点（基于速度场梯度）
# 50% 表层点（z < 2 个网格点）
```

**优势：**
- ✅ 保证地表附近有足够的采样点
- ✅ 更好地捕获自由表面边界条件
- ✅ 提高地表附近波场预测精度

---

## 📊 采样策略详解

### 1️⃣ 结构边界点（50%）

**目的：** 采样速度场变化剧烈的区域（如地层界面）

**方法：**
1. 计算速度场的空间梯度幅度
   ```python
   grad_z = vel[:, :, 2:, 1:-1] - vel[:, :, :-2, 1:-1]
   grad_x = vel[:, :, 1:-1, 2:] - vel[:, :, 1:-1, :-2]
   vel_grad_mag = sqrt(grad_z^2 + grad_x^2)
   ```

2. 将梯度幅度转换为概率分布
   ```python
   prob_dist = vel_grad_mag / vel_grad_mag.sum()
   ```

3. 按概率采样网格点
   ```python
   sampled_indices = torch.multinomial(prob_dist, num_samples=num_structure)
   ```

### 2️⃣ 表层点（50%）

**目的：** 采样地表附近的区域（自由表面边界）

**方法：**
1. 定义表层深度范围
   ```python
   surface_depth = 2.0 * dz  # 2 个网格点的深度
   ```

2. 在表层范围内均匀采样
   ```python
   z_surf = torch.rand(num_surface) * surface_depth  # z ∈ [0, 2*dz]
   x_surf = torch.rand(num_surface) * max_x          # x ∈ [0, max_x]
   ```

---

## 🎯 物理意义

### 为什么需要表层采样？

1. **自由表面边界条件**
   - 自由表面（z=0）是波场的重要边界
   - 地表反射波、面波等物理现象都在此区域

2. **波场特征**
   - 震源通常位于地表附近
   - 地表附近的波场梯度较大
   - 需要更密集的采样来捕获这些特征

3. **PDE 残差**
   - 自由表面边界条件：∂u/∂z = 0（在 z=0 处）
   - 需要足够的采样点来约束边界条件

---

## 📈 预期效果

### 训练效果改善

| 指标 | 改进前 | 改进后 | 说明 |
|------|--------|--------|------|
| 地表波场精度 | 一般 | **更高** | 更密集的地表采样 |
| 边界条件满足 | 一般 | **更好** | 更多约束点 |
| 收敛速度 | 基准 | **更快** | 更好的采样策略 |

### 采样分布示例

假设 `vel.shape = [1, 1, 71, 72]`，`num_pts = 900`：

```
结构点（450个）：
  - 主要分布在地层界面
  - 速度突变区域
  - 深部构造边界

表层点（450个）：
  - z ∈ [0, 2*40] = [0, 80] 米
  - x ∈ [0, 72*40] = [0, 2880] 米
  - 地表附近均匀分布
```

---

## 🔍 参数调整

### 可调参数

```python
def generate_structure_aware_y_ran(
    self, 
    vel, 
    num_pts=20000,      # 总采样点数量
    max_z=None,         # z方向最大坐标（自动计算）
    max_x=None          # x方向最大坐标（自动计算）
):
```

### 采样比例调整

如果需要调整表层采样比例，修改第 343-344 行：

```python
# 当前：50% 结构点，50% 表层点
num_structure = int(num_pts * 0.5)  # 可调整比例
num_surface = num_pts - num_structure

# 例如：30% 结构点，70% 表层点
num_structure = int(num_pts * 0.3)
num_surface = num_pts - num_structure
```

### 表层深度调整

修改第 356 行：

```python
# 当前：2 个网格点深度
surface_depth = 2.0 * dz

# 例如：3 个网格点深度
surface_depth = 3.0 * dz

# 或者：固定深度（如 100 米）
surface_depth = 100.0  # 米
```

---

## 🎯 与边界条件的配合

### Free Surface 边界

```
地表 (z=0): 自由表面
  ↑
  |  表层采样区 (z < 2*dz)
  |  - 450 个采样点
  |  - 捕获地表反射、面波
  |
  |  结构采样区 (全场)
  |  - 450 个采样点
  |  - 捕获地层界面、深部构造
  ↓
底部 (z=71*40): PML 吸收边界
```

### Full PML 边界

对于 `full_pml` 边界，表层采样同样有效：
- 虽然顶部有 PML，但地表附近仍然是波场的重要区域
- 震源通常位于浅层
- 浅层的波场对整体预测影响大

---

## 📊 验证方法

### 1. 可视化采样点分布

```python
import matplotlib.pyplot as plt

# 生成采样点
y_ran = model.generate_structure_aware_y_ran(vel, num_pts=900)

# 提取坐标
z_coords = y_ran[0, :, 0].cpu().numpy()
x_coords = y_ran[0, :, 1].cpu().numpy()

# 绘制散点图
plt.figure(figsize=(12, 8))
plt.scatter(x_coords[:450], z_coords[:450], s=1, alpha=0.5, label='Structure Points')
plt.scatter(x_coords[450:], z_coords[450:], s=1, alpha=0.5, label='Surface Points')
plt.xlabel('X (m)')
plt.ylabel('Z (m)')
plt.legend()
plt.title('Sampling Point Distribution')
plt.savefig('sampling_distribution.png')
```

### 2. 统计表层采样比例

```python
# 统计表层点（z < 2*dz）的比例
surface_depth = 2.0 * (71 * 40 / vel.shape[2])
surface_count = (y_ran[0, :, 0] < surface_depth).sum().item()
surface_ratio = surface_count / y_ran.shape[1]

print(f"表层采样比例: {surface_ratio:.2%}")  # 应该接近 50%
```

---

## 🎉 改进总结

### 关键改进

1. ✅ **自适应表层采样**：50% 的点集中在地表附近
2. ✅ **参数自动计算**：使用实际网格尺寸，避免硬编码
3. ✅ **物理驱动设计**：符合自由表面边界条件的物理特征

### 适用场景

- ✅ 自由表面边界条件
- ✅ 浅层震源问题
- ✅ 地震波场模拟
- ✅ 需要精确捕获地表特征的问题

---

## 📌 注意事项

1. **采样比例**：可根据具体问题调整结构和表层点的比例
2. **表层深度**：可根据震源深度和波场特征调整
3. **与其他技术结合**：可与梯度累加、动态权重等技术配合使用

---

**改进完成！现在训练将更好地捕获地表附近的波场特征！** 🚀
