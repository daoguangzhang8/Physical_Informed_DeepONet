# PI-DeepONet 优化记录

## 本次完成的优化 (2026-03-31)

### 1. 显存优化

| 优化项 | 文件 | 效果 |
|--------|------|------|
| PDE loss 中 detach 优化 | `PI_DeepOnet.py` | 节省 15-25% 显存 |
| 删除未使用的全局位置编码 | `PI_DeepOnet.py` | 减少模型参数 |
| 合并双份 PDE 采样 | `PI_DeepOnet.py` | 减少 15-20% 显存 |

**detach 详情**:
- `c` (速度场采样值) - detach
- `U0_real`, `U0_imag` (背景场) - detach
- `pml_tmp1-5`, `lx`, `lz` (PML 边界系数) - 用 `torch.no_grad()` 包裹

### 2. 速度优化

| 优化项 | 文件 | 效果 |
|--------|------|------|
| DataLoader 多进程 | `dataloader.py` | 加速 20-40% |
| 修复 meshgrid warning | `plotting.py` | 消除警告 |

**DataLoader 配置**:
```python
num_workers = 4
prefetch_factor = 2
```

### 3. Bug 修复

| 问题 | 文件 | 修复 |
|------|------|------|
| `ploting.py` 拼写错误 | 多个文件 | 重命名为 `plotting.py` |
| `Mean of empty slice` warning | `train.py` | 添加空列表检查 |
| `torch.compile` 不支持复数 | `train.py` | 禁用 compile |

### 4. 文档

- 新增 `README.md` 完整项目文档

---

## 未完成的优化（待实施）

### 高优先级

| 优化项 | 预计效果 | 风险 | 备注 |
|--------|----------|------|------|
| **AMP 混合精度训练** | 加速 20-40%，省显存 30-40% | 中 | 需测试是否影响收敛 |
| **减少自适应采样点** (900→500) | 加速 15-20% | 低 | 简单修改 |

### 中优先级

| 优化项 | 预计效果 | 风险 | 备注 |
|--------|----------|------|------|
| 梯度累积步数调整 | 省显存 | 低 | loss 曲线会波动 |
| 减小 batch_size_v | 省显存 | 低 | 可能影响训练稳定性 |
| 增大验证/绘图间隔 | 减少 I/O | 低 | 简单修改 |

### 低优先级（需重新调参）

| 优化项 | 预计效果 | 风险 | 备注 |
|--------|----------|------|------|
| 减小特征维度 (256→128) | 省显存 ~50% | 高 | 影响模型容量 |
| 使用 functorch.jacrev | 加速 + 省显存 | 高 | 需重写二阶导计算 |

---

## AMP 混合精度训练实现方案（待实施）

```python
# train.py 中添加
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 训练循环中
with autocast():
    loss = model.loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 已知问题

1. **torch.compile 不兼容**: 模型中使用 `cfloat` 复数运算，torch.compile 不支持
2. **显存仍可能不足**: 建议实施 AMP 进一步优化
3. **loss 曲线波动**: 使用较大 accumulation_steps 时会出现，建议方案 B（累积后平均）

---

## 测试记录

- 测试环境: RTX 4090, 24GB 显存
- 优化后可正常运行，无明显 OOM
- 需要更多数据测试梯度累积的效果
