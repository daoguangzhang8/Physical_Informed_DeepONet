# DDP 课程学习 (Staged Curriculum Training) 设计

**日期**: 2026-04-30
**状态**: Approved

## Context

当前项目有三条互斥的训练路径：
- 单卡单阶段 (`train_single`)
- 单卡课程学习 (`train_staged` → `_train_stage`)
- 多卡 DDP (`_train_worker`)

DDP 路径不支持课程学习（`staged_training`），导致多卡环境无法利用低频→中频→高频的渐进训练策略。目标是在不改动单卡代码的前提下，让 DDP 路径支持课程学习。

运行环境: 2x RTX 4090 (23.5GB each)。

## DDP 策略分析

当前 DDP 使用标准 **DataParallel on DDP** 策略：
- `DistributedSampler` 将 velocity 样本按 GPU 切分
- `no_sync()` 压缩 all-reduce 次数（每个 velocity batch 只同步一次）
- `lr * world_size` 线性缩放学习率

该策略对本模型合理，主要注意事项：
1. **y_ran 一致性**: 使用 epoch 共享 y_ran 缓解了跨 rank 采样差异
2. **LR 缩放**: Adam 下线性缩放只是近似，2 卡场景下可接受

## 方案：在 `_train_worker` 上叠加课程学习循环

### 入口路由 (main2.py)

```
main():
  if use_parallel:
    if staged_training:
      train_distributed_staged(args)   ← 新入口
    else:
      train_distributed(args)          ← 不变
    return
  train(args)                          ← 不变
```

`train_distributed_staged` 用 `mp.spawn` 启动 `_train_worker_staged`，进程生命周期覆盖所有阶段。

### 核心循环 (_train_worker_staged)

```
setup_distributed → 创建模型 + DDP wrap → for each stage:
  1. 替换文件名、data_dir
  2. 加载上阶段权重 (rank 0 load → broadcast)
  3. prepare_training_dataloaders()
  4. Replay 合并
  5. DistributedSampler 包裹
  6. 创建 optimizer (lr=stage_lr * world_size)
  7. 训练循环 (DDP forward/compute_loss 分离 + no_sync)
  8. 保存最终权重 (rank 0 only)
→ cleanup_distributed
```

- 模型只创建一次，跨阶段复用
- 优化器每阶段重新创建
- 训练循环复用现有 DDP 逻辑

### Replay 合并

每个 rank 执行相同的合并操作：
1. 加载当前阶段数据 (`prepare_training_dataloaders`)
2. 对于每个 replay_stage:
   - 替换文件名，重新 `prepare_training_dataloaders()`
   - `torch.manual_seed(42 + replay_idx)` 固定 subsample
   - `torch.cat` 合并到 combined tensors
3. 重建 TensorDataset + DataLoader(DistributedSampler)

所有 rank 数据一致的原因:
- `np.random.seed(1)` 固定训练/验证划分
- `torch.manual_seed` 固定 replay subsample
- `DistributedSampler` 保证各 rank 各取 1/world_size

### 权重同步

阶段切换时：
1. Rank 0: `model.module.load_state_dict(checkpoint)`
2. 所有 rank: `dist.broadcast` 所有参数

Checkpoint 保存 (仅 rank 0):
- 文件名: `PI_DeepONet_pde_stage{idx}_{epoch}epoch_weights_{nz}.pth`
- 包含: model_state_dict, optimizer_state_dict, scheduler_state_dict, stage, epoch_in_stage

## 改动范围

| 文件 | 改动 |
|------|------|
| `main2.py` | 路由增加 staged_training 分支 |
| `model/train_distributed.py` | 新增 `_train_worker_staged` + `train_distributed_staged` |
| 不改动的文件 | `train.py`, `PI_DeepOnet.py`, `dataloader.py`, `utils.py` |

## 验证

1. 设置 `use_parallel=True, staged_training=True`，确认路由正确
2. 检查 Stage 0 训练日志：每 epoch 损失应与单卡版本在同一量级
3. 检查 Stage 1 日志：应正确加载 Stage 0 最终权重 + 合并 replay 数据
4. 检查 GPU 利用率：两张卡都应有计算负载
5. 对比单卡/双卡相同 epoch 的模型输出（应一致，因为 DistributedSampler 覆盖全部数据）
