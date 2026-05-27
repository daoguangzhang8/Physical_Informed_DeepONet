# n_pml 索引对应关系修正总结

## 问题发现

在将 MATLAB 代码 FDFD.m 移植到 Python 时，发现 `n_pml` 的索引对应关系存在错误。

## MATLAB 中的 n_pml 定义

在 MATLAB (FDFD.m) 中：
```matlab
model.n_pml = [round(model.n(1)/10) round(model.n(2)/10);
                round(model.n(1)/10) round(model.n(2)/10)];
```

这表示：
- `n_pml(1,1)`: 第一维度（z方向）起始的 PML - **顶部 PML**
- `n_pml(1,2)`: 第二维度（x方向）起始的 PML - **左侧 PML**
- `n_pml(2,1)`: 第一维度（z方向）结束的 PML - **底部 PML**
- `n_pml(2,2)`: 第二维度（x方向）结束的 PML - **右侧 PML**

## Python 中的正确对应关系

由于 Python 使用 0-based 索引，MATLAB 的 `n_pml(i,j)` 对应 Python 的 `n_pml[i-1, j-1]`：

| MATLAB 索引 | Python 索引 | 含义 | 方向 |
|------------|------------|------|------|
| n_pml(1,1) | n_pml[0, 0] | 顶部 PML | z 方向起始 |
| n_pml(1,2) | n_pml[0, 1] | 左侧 PML | x 方向起始 |
| n_pml(2,1) | n_pml[1, 0] | 底部 PML | z 方向结束 |
| n_pml(2,2) | n_pml[1, 1] | 右侧 PML | x 方向结束 |

## 修正的文件

### 1. getA9_PML.py

**修正前（错误）：**
```python
top_pml = n_pml[0, 1]    # 顶部 PML (z方向起始)
bottom_pml = n_pml[1, 1]  # 底部 PML (z方向结束)
left_pml = n_pml[0, 0]    # 左侧 PML (x方向起始)
right_pml = n_pml[1, 0]   # 右侧 PML (x方向结束)
```

**修正后（正确）：**
```python
top_pml = n_pml[0, 0]    # 顶部 PML (z方向起始)
bottom_pml = n_pml[1, 0]  # 底部 PML (z方向结束)
left_pml = n_pml[0, 1]    # 左侧 PML (x方向起始)
right_pml = n_pml[1, 1]   # 右侧 PML (x方向结束)
```

### 2. getFPML.py

getFPML.py 中的代码已经是正确的，但注释有误导性。已更新注释以明确说明：
- `n_pml[0, 0]`: 顶部 PML (z 方向)
- `n_pml[0, 1]`: 左侧 PML (x 方向)
- `n_pml[1, 0]`: 底部 PML (z 方向)
- `n_pml[1, 1]`: 右侧 PML (x 方向)

### 3. modeling.py

modeling.py 中的 n_pml 定义是正确的：
```python
n_pml = np.array([[10, 10],  # [顶部PML(z方向), 左侧PML(x方向)]
                  [10, 10]])  # [底部PML(z方向), 右侧PML(x方向)]
```

## 验证测试

创建了 `test_npml_mapping.py` 来验证 n_pml 的索引对应关系是否正确。测试结果显示：
- ✅ DUM1 (x 方向) 的构建正确
- ✅ DUM2 (z 方向) 的构建正确
- ✅ FPML (Kronecker 积) 的构建正确
- ✅ 扩展后的模型内容正确（原始区域保持不变，PML 区域正确扩展）

## 总结

修正后的 n_pml 索引对应关系为：
- **第一列（索引 0）**: z 方向（顶部/底部）
- **第二列（索引 1）**: x 方向（左侧/右侧）
- **第一行（索引 0）**: 起始边界（顶部/左侧）
- **第二行（索引 1）**: 结束边界（底部/右侧）

这与 MATLAB 原始代码完全一致，确保了 Python 实现的正确性。
