# Envelope Loss 集成设计

## 目标

在 PI-DeepONet 的损失函数中新增 envelope loss，用于显式约束预测波场与真实波场的包络（振幅包络）一致性，辅助提升高频表达能力。

## 背景

当前损失函数由两项组成：

```
L_total = a * L_data / data_norm_coe + b * L_pde / pde_norm_coe
```

其中 `data_norm_coe` 和 `pde_norm_coe` 在第一个 epoch 结束后用各自均值初始化，后续 loss 除以该系数实现归一化。

高频波场对相位极其敏感，MSE 意义上的 data loss 可能无法充分约束振幅结构。Envelope loss 直接约束包络幅度，帮助模型在相位拟合困难时至少保持正确的空间振幅分布。

## 设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 集成方式 | 在 `loss()` / `compute_loss()` 内部扩展 | 复用 `pred_y`，避免重复 forward |
| 计算范围 | 仅标签点 | 与 data loss 一致，不涉及 PDE 采样点 |
| Loss 形式 | MSE(env_pred, env_label) | 复用 `self.loss_function`，不引入新损失函数 |
| 归一化策略 | 首 epoch 均值归一化 | 与 data/PDE 完全对称 |
| 默认权重 d | 0.1 | 辅助 loss，不压过 data/PDE 主 loss |

## 修改清单

### 1. config.py

在"损失函数"区域新增：

```python
d = 0.1   # Envelope Loss 权重（MSE 形式，仅在标签点上计算）
```

### 2. model/PI_DeepOnet.py

#### 2.1 `loss()` 方法

签名新增参数 `d` 和 `env_norm_coe`：

```python
def loss(self, vel, y, UU0, labels, a, b, c, d,
         data_norm_coe=1., pde_norm_coe=1., env_norm_coe=1.,
         freq_batch=None, y_ran=None):
```

在 `pred_y` 计算后新增 envelope loss 计算：

```python
# Envelope loss
env_pred = torch.sqrt(pred_y[..., 0]**2 + pred_y[..., 1]**2 + 1e-8)
env_label = torch.sqrt(labels[..., 0]**2 + labels[..., 1]**2 + 1e-8)
loss_env = self.loss_function(env_pred, env_label) / env_norm_coe
```

加权求和中新增 `d * loss_env`：

```python
loss_val = a * loss_u + b * loss_f_combined + d * loss_env
```

返回值从 4 个变为 5 个：

```python
return loss_val, loss_f_combined, loss_u, loss_r, loss_env
```

#### 2.2 `compute_loss()` 方法（DDP 版本）

做完全对称的改动：新增 `d`、`env_norm_coe` 参数，新增 envelope 计算，返回 5 个值。

### 3. model/train.py

#### 3.1 初始化区域

```python
d = args.d
env_norm_coe = 1.
loss_env_log = []
```

#### 3.2 首轮归一化

```python
if first_flag:
    data_norm_coe = np.mean(batch_u_loss) if batch_u_loss else 1.0
    pde_norm_coe = np.mean(batch_f_loss) if batch_f_loss else 1.0
    env_norm_coe = np.mean(batch_env_loss) if batch_env_loss else 1.0
    loss_env_log.append(1.)
    first_flag = False
else:
    loss_env_log.append(np.mean(batch_env_loss) if batch_env_loss else 0)
```

#### 3.3 loss 调用处

所有调用 `model.loss()` 的地方：
- 传入新增参数 `d`、`env_norm_coe`
- 接收第 5 个返回值 `loss_env`
- 将 `loss_env.item()` 收集到 `batch_env_loss`

所有调用 `model.compute_loss()` 的地方（DDP 训练路径）做同样改动。

#### 3.4 进度条

```python
pbar.set_postfix({
    'Total': f"{avg_loss:.4e}",
    'PDE': f"{loss_pde_log[-1]:.4e}",
    'Data': f"{loss_data_log[-1]:.4e}",
    'Env': f"{loss_env_log[-1]:.4e}",
    'LR': f"{current_lr:.2e}"
})
```

#### 3.5 保存与绘图

新增 `np.save` 保存 `loss_env_log`。plotting 中的 loss 曲线增加 envelope 分量。

## 不变的部分

- forward 流程不变，复用已有 `pred_y`
- data / PDE loss 计算和归一化逻辑不变
- 动态权重调整（`if_adjust`）暂不涉及 `d`
- `loss_BC()`、`loss_PDE_Scatter_pml()` 等独立方法不变

## Envelope 计算细节

```python
env = sqrt(real^2 + imag^2 + 1e-8)
loss_env = MSE(env_pred, env_label)
```

`+ 1e-8` 防止零值处的梯度问题。复用 `self.loss_function`（即 `nn.MSELoss`），不引入新的损失函数类。
