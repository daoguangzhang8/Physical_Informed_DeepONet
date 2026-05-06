# Epoch-Level 共享 y_ran 采样设计

## 背景

当前 `generate_structure_aware_y_ran` 在每个 velocity batch 内按 per-model 速度梯度自适应采样。由于显存限制，batch 内所有速度模型共享同一组 `y_ran`，导致采样偏向当前 batch 的某个模型结构。

## 目标

将 `y_ran` 改为 epoch 级全局结构概率图 + 共享采样，消除 batch 间采样偏差，同时保持工程简洁性。

## 修改范围

| 文件 | 修改内容 |
|------|---------|
| `config.py` | 新增 `y_ran_*` 配置参数 |
| `model/utils.py` | 新增 `build_epoch_velocity_gradient_prob()` 和 `sample_shared_y_ran_from_epoch_prob()` |
| `model/train.py` | epoch 循环开头计算概率图，替换 y_ran 生成逻辑 |
| `model/train_distributed.py` | 同 train.py 的修改，DDP 版本 |

## 采样组成（实验A，num_pts=900）

- 60% epoch-structure（540点）：从全局平均梯度概率图采样
- 20% surface（180点）：表层深度 5dh 内均匀采样
- 20% uniform（180点）：全区域均匀采样

## 数据流

```
epoch 开始
  → build_epoch_velocity_gradient_prob(train_loader)
  → epoch_prob [Z*X]

每个 vel_batch
  → sample_shared_y_ran_from_epoch_prob(epoch_prob, args, ...)
  → y_shared [900, 2]
  → y_shared.unsqueeze(0).expand(B, -1, -1).clone().requires_grad_(True)
  → y_ran [B, 900, 2]

model.loss(..., y_ran=y_ran)  # 不变
```

## 新增函数设计

### 1. build_epoch_velocity_gradient_prob

位置：`model/utils.py`

输入训练 dataloader，遍历所有训练 velocity model，计算平均速度梯度图，输出归一化概率分布。

接口：
```python
def build_epoch_velocity_gradient_prob(
    train_loader,
    device,
    eps=1e-8,
    use_max_mix=False,
    mean_weight=0.7,
    max_weight=0.3,
):
```

核心逻辑：
- 遍历 train_loader，对每个 batch 计算速度梯度幅度 `sqrt(grad_z^2 + grad_x^2)`
- 累加所有 batch 的梯度幅度求均值
- 若 `use_max_mix=True`，则 `score = mean_weight * mean + max_weight * max`
- 归一化为概率分布 `prob = score / sum(score)`

返回：`(prob, score)`，prob 是 `[Z*X]` 的扁平概率分布，score 是 `[Z, X]` 的原始分数图（用于可视化诊断）

### 2. sample_shared_y_ran_from_epoch_prob

位置：`model/utils.py`

从 epoch-level 概率图中采样一组共享 y_ran，支持 structure / surface / source-near / uniform 四种采样类型的可配置比例。

接口：
```python
def sample_shared_y_ran_from_epoch_prob(
    prob,
    args,
    num_pts=900,
    structure_ratio=0.60,
    surface_ratio=0.20,
    uniform_ratio=0.20,
    source_ratio=0.0,
    source_coords=None,
    surface_depth_grids=5,
    source_r_min_grids=1.5,
    source_r_max_grids=8.0,
    replacement=True,
):
```

核心逻辑：
1. 按比例计算各类点数，source_ratio=0 时其份额归入 uniform
2. structure 点：`torch.multinomial(prob, ...)` 采样网格索引，再加 cell 内随机偏移
3. surface 点：z 在 `[0, surface_depth_grids * dh]` 内均匀采样，x 在 `[0, max_x]` 内均匀采样
4. source-near 点：随机选震源，以极坐标在 `[r_min, r_max]` 环形区域采样
5. uniform 点：z 在 `[0, max_z]`，x 在 `[0, max_x]` 内均匀采样
6. 四类点 cat 后裁剪/补齐到精确的 num_pts

返回：`y_shared [num_pts, 2]`，坐标格式 `[z, x]`，物理坐标

## 配置参数（新增到 config.py Args 类）

```python
# y_ran epoch-level shared sampling
use_epoch_shared_y_ran = True
y_ran_num_pts = 900
y_ran_structure_ratio = 0.60
y_ran_surface_ratio = 0.20
y_ran_uniform_ratio = 0.20
y_ran_source_ratio = 0.0
y_ran_surface_depth_grids = 5
y_ran_use_max_mix = False
y_ran_mean_weight = 0.7
y_ran_max_weight = 0.3
y_ran_prob_update_every = 1  # 1=每epoch, >1=每N epoch, 0=只算一次
```

## 训练循环修改

### train.py

在 epoch 循环内、velocity batch 循环前，计算概率图：

```python
epoch_prob = None

for i in pbar:
    model.train()

    # 计算/更新 epoch_prob
    should_update = (
        epoch_prob is None
        or args.y_ran_prob_update_every == 1
        or (args.y_ran_prob_update_every > 1 and i % args.y_ran_prob_update_every == 0)
    )
    if should_update:
        epoch_prob, epoch_score = build_epoch_velocity_gradient_prob(
            train_loader=dataloader['train'],
            device=device,
            use_max_mix=args.y_ran_use_max_mix,
            mean_weight=args.y_ran_mean_weight,
            max_weight=args.y_ran_max_weight,
        )

    for batch_data in dataloader['train']:
        # 替换原来的 model.generate_structure_aware_y_ran()
        with torch.no_grad():
            y_shared = sample_shared_y_ran_from_epoch_prob(
                prob=epoch_prob,
                args=args,
                num_pts=args.y_ran_num_pts,
                structure_ratio=args.y_ran_structure_ratio,
                surface_ratio=args.y_ran_surface_ratio,
                uniform_ratio=args.y_ran_uniform_ratio,
                source_ratio=args.y_ran_source_ratio,
                surface_depth_grids=args.y_ran_surface_depth_grids,
            )
        y_ran = y_shared.unsqueeze(0).expand(
            vel_batch.shape[0], -1, -1
        ).clone().requires_grad_(True)
```

### train_distributed.py

同 train.py 的修改逻辑，区别：
- 使用 `model.module` 而非 `model`（DDP wrapper）
- 只在 rank 0 计算概率图并广播，或各 rank 独立计算（数据相同结果一致）

## 设计决策

1. **保留原函数**：`generate_structure_aware_y_ran` 不删除，通过 `use_epoch_shared_y_ran` 配置切换，方便 A/B 对比
2. **采样在 no_grad 内**：避免污染计算图
3. **expand 后 clone + requires_grad**：保证 autograd 正确性
4. **坐标风格**：cell 内随机偏移（`z_idx * dh + rand * dh`），与现有 PDE loss 的坐标处理一致
5. **概率图更新频率**：默认每 epoch 重算，但因 velocity 不变，可配置为缓存模式

## 不涉及的部分

- `model/PI_DeepOnet.py`：`loss()` 函数不变，`y_ran` 的拼接和 PDE 计算逻辑不变
- `model/dataloader.py`：数据加载不变
- `test.py`：测试流程不变
- `num_pts` 不变，仍为 900

## 可视化诊断（可选）

每隔若干 epoch 保存：
- `epoch_score` 热力图（结构概率图）
- `y_shared` scatter 图（采样点分布）
- 采样点与 velocity model 的叠加图
