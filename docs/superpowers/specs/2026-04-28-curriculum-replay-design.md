# 课程学习 Replay 防遗忘设计

## 背景

当前三阶段课程学习中，每个阶段只加载自己频段的数据训练（Stage 0: 3-11Hz, Stage 1: 12-18Hz, Stage 2: 18-25Hz），阶段间仅传递模型权重。这导致后续阶段训练时模型会遗忘前阶段学到的低频特征。

## 目标

在 Stage 1 和 Stage 2 中混入所有前序阶段的完整训练数据，防止灾难性遗忘。

## 修改范围

| 文件 | 修改内容 |
|------|---------|
| `config.py` | 每个 stage 配置新增 `replay_stages` 列表 |
| `model/train.py` | `_train_stage` 增加前序数据加载和拼接逻辑 |

不涉及 `model/dataloader.py`、`model/train_distributed.py`（DDP 版本暂不支持 staged training）。

## 配置设计

每个 stage 配置中新增 `replay_stages` 字段，指定要 replay 的前序阶段编号列表：

```python
stages = [
    {
        'name': 'low_freq',
        'freq_range': '3to11',
        ...
        'replay_stages': [],           # Stage 0: 不 replay
    },
    {
        'name': 'mid_freq',
        'freq_range': '12to18',
        ...
        'replay_stages': [0],          # Stage 1: replay Stage 0
    },
    {
        'name': 'high_freq',
        'freq_range': '18to25',
        ...
        'replay_stages': [0, 1],       # Stage 2: replay Stage 0 和 1
    },
]
```

每个前序阶段完整复制所有速度模型和震源数据，不做子集采样。

## 数据加载流程

在 `_train_stage` 中，当前阶段数据加载完成后（`prepare_training_dataloaders` 返回 `dataloader, plot_data`），遍历 `replay_stages`：

1. 对每个 replay stage，按其 `freq_range` 替换文件名，调用 `prepare_training_dataloaders` 加载该阶段数据
2. 从返回结果中提取训练集 Tensor（vel, UU0, labels, freq）
3. 与当前阶段的训练集 Tensor 在 batch 维度上 `torch.cat` 拼接
4. 用拼接后的 Tensor 重建 DataLoader

拼接发生在 CPU 端，`batch_size_v` 不变，GPU 峰值显存不变。

### 伪代码

```python
# 在 _train_stage 中，prepare_training_dataloaders 之后：
if replay_stages and stage_idx > 0:
    replay_parts = [train_vel, train_UU0, train_labels]
    replay_freqs = [train_freq] if train_freq is not None else None

    for replay_idx in replay_stages:
        replay_config = args.stages[replay_idx]
        # 替换文件名为 replay 阶段的频段
        args.vel_filename = base_vel_filename.replace(base_freq_tag, f'freq{replay_config["freq_range"]}')
        args.backgroundfield_filename = base_bg_filename.replace(base_freq_tag, f'freq{replay_config["freq_range"]}')
        args.wavefield_filename = base_wf_filename.replace(base_freq_tag, f'freq{replay_config["freq_range"]}')
        args.freq_filename = base_freq_filename.replace('freq_used', f'freq{replay_config["freq_range"]}_used')

        replay_dl, _ = prepare_training_dataloaders(args, device)

        # 提取 replay 训练数据并拼接
        replay_vel, replay_UU0, replay_labels, replay_freq = extract_train_tensors(replay_dl)
        train_vel = torch.cat([train_vel, replay_vel], dim=0)
        train_UU0 = torch.cat([train_UU0, replay_UU0], dim=0)
        train_labels = torch.cat([train_labels, replay_labels], dim=0)
        if replay_freqs is not None and replay_freq is not None:
            replay_freqs.append(replay_freq)

    # 重建 DataLoader
    dataloader = rebuild_dataloader(train_vel, train_UU0, train_labels, replay_freqs, args, device)
```

### 提取辅助函数

新增一个辅助函数，从 `prepare_training_dataloaders` 返回的 DataLoader 中提取训练集 Tensor：

```python
def _extract_train_tensors_from_loader(dataloader):
    """从 DataLoader 的 dataset 中提取训练数据 Tensor"""
    ds = dataloader['train'].dataset
    tensors = ds.tensors
    if len(tensors) >= 4:
        return tensors[0], tensors[1], tensors[2], tensors[3]  # vel, UU0, labels, freq
    else:
        return tensors[0], tensors[1], tensors[2], None
```

### 重建 DataLoader

拼接后需要重建 DataLoader，复用现有的 `y_train`、`valid` 数据和 `plot_data`。重建逻辑直接在 `_train_stage` 中完成，从拼接后的 Tensor 构建 `TensorDataset` + `DataLoader`。

## 关键约束

1. 文件名恢复：加载 replay 数据后会修改 `args.vel_filename` 等字段，需要在 replay 加载完毕后**恢复为当前阶段的文件名**，避免影响后续逻辑
2. `nz/nx` 一致性：各阶段数据的网格尺寸必须一致（共享同一物理网格），否则 cat 会失败
3. `epoch_prob` 自动覆盖：由于 replay 数据已拼接到训练集 DataLoader 中，`build_epoch_velocity_gradient_prob` 会自动在全部数据上计算概率图
4. `plot_data` 不受影响：验证/可视化数据仍使用当前阶段的原始数据

## 不涉及的部分

- `model/train_distributed.py`：staged training 当前不支持 DDP
- `model/dataloader.py`：不修改 `prepare_training_dataloaders`，只通过其返回结果提取 Tensor
- `train_single`：单阶段训练不涉及 replay
- `test.py`：测试流程不变
