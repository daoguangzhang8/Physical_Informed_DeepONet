# modeling.py 修改说明

## 修改概述

修改了 `modeling.py` 中的 PML 参数设置，使其与更新后的 `getFPML.py` 和 `getCPML.py` 接口兼容。

---

## 主要修改

### 1. **PML 参数定义（第 44-71 行）**

**修改前：**
```python
nz = 70;
nx = 70;
Lpml = 10
# ... 其他参数 ...
nz = nz + Lpml * 2  # 直接修改 nz 和 nx
nx = nx + Lpml * 2
```

**修改后：**
```python
nz = 70;
nx = 70;
Lpml = 10
# ... 其他参数 ...

# 定义模型尺寸和PML参数（用于新的 getFPML/getCPML 接口）
n = np.array([nz, nx])  # 原始模型尺寸 [nz, nx]
n_pml = np.array([[Lpml, Lpml],  # [顶部PML, 左侧PML]
                  [Lpml, Lpml]])  # [底部PML, 右侧PML]

# 计算包含PML的扩展尺寸
ne = n + np.sum(n_pml, axis=0)  # 扩展后的尺寸
nz_ext = ne[0]  # 扩展后的 nz
nx_ext = ne[1]  # 扩展后的 nx

# 为了兼容后续代码，更新 nz 和 nx 为扩展后的尺寸
nz = nz_ext
nx = nx_ext
```

**改进：**
- ✅ 明确区分原始模型尺寸 `n` 和扩展后尺寸 `ne`
- ✅ 使用 `n_pml` 矩阵支持每个边界不同的 PML 厚度
- ✅ 为未来支持自由表面（顶部 PML=0）做好准备

---

### 2. **getFPML/getCPML 调用（第 137-138 行）**

**修改前：**
```python
CPML = getCPML(Lpml, nx, nz)
FPML = getFPML(Lpml, nx, nz)
```

**修改后：**
```python
CPML = getCPML(n_pml, n)
FPML = getFPML(n_pml, n)
```

**说明：**
- 使用新的接口：`getFPML(n_pml, n)` 和 `getCPML(n_pml, n)`
- `n_pml`: PML 厚度矩阵 (2x2)
- `n`: 原始模型尺寸 [nz, nx]

---

### 3. **速度模型处理（第 153-161 行）**

**修改前：**
```python
v0 = np.ones(((nz - 2 * Lpml), (nx - 2 * Lpml))) * 1500
mv0 = rhot / (np.reshape(v0, ((nx - 2 * Lpml) * (nz - 2 * Lpml), 1), order = 'F') / 1000) ** 2
# ...
mv = rhot / (np.reshape(v, ((nx - Lpml * 2) * (nz - Lpml * 2), 1) , order = 'F') / 1000) ** 2
```

**修改后：**
```python
v0 = np.ones((n[0], n[1])) * 1500
mv0 = rhot / (np.reshape(v0, (n[0] * n[1], 1), order = 'F') / 1000) ** 2
# ...
mv = rhot / (np.reshape(v, (n[0] * n[1], 1) , order = 'F') / 1000) ** 2
```

**改进：**
- ✅ 使用 `n[0]` 和 `n[1]` 替代 `nz - 2*Lpml`，代码更清晰
- ✅ 明确表示这是原始模型尺寸

---

## 参数说明

### n_pml 矩阵结构

```python
n_pml = np.array([[顶部PML, 左侧PML],
                  [底部PML, 右侧PML]])
```

**示例：**

1. **对称 PML（四个边界相同）：**
   ```python
   n_pml = np.array([[10, 10],
                     [10, 10]])
   ```

2. **顶部自由表面：**
   ```python
   n_pml = np.array([[0, 10],   # 顶部 PML = 0
                     [10, 10]])
   ```

3. **非对称 PML：**
   ```python
   n_pml = np.array([[5, 20],
                     [15, 10]])
   ```

---

## 尺寸计算

| 变量 | 说明 | 值 |
|------|------|-----|
| `n` | 原始模型尺寸 | [70, 70] |
| `n_pml` | PML 厚度矩阵 | [[10, 10], [10, 10]] |
| `ne` | 扩展后尺寸 | [90, 90] |
| `nz`, `nx` | 扩展后（用于后续计算） | 90, 90 |

**计算公式：**
```python
ne = n + np.sum(n_pml, axis=0)
# ne = [70, 70] + [20, 20] = [90, 90]
```

---

## 验证测试

运行测试脚本：
```bash
python test_modeling_import.py
```

**测试结果：**
```
✓ 矩阵功能正确
✓ 速度模型扩展正确
✓ 还原误差: 0.00e+00
```

---

## 向后兼容性

- ✅ `getA9_PML.py` 的接口保持不变（使用扩展后的 nz, nx）
- ✅ 后续代码中的震源和检波器位置定义保持不变
- ✅ 所有矩阵运算保持一致

---

## 如何启用自由表面

如果需要模拟自由表面（顶部无 PML），只需修改 `n_pml` 定义：

```python
# 当前：四个边界都有 PML
n_pml = np.array([[10, 10],
                  [10, 10]])

# 修改为：顶部自由表面
n_pml = np.array([[0, 10],   # 顶部 PML = 0
                  [10, 10]])
```

**注意：** 修改后需要调整震源和检波器的位置，因为扩展尺寸会改变。

---

## 修改日期

2026-04-01

## 修改者

Claude Code Assistant
