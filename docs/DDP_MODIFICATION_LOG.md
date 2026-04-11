# DDP 分布式训练修改记录

**修改日期：** 2026-04-10
**修改目标：** 修复 DDP 多卡训练比单卡慢的根本性问题
**修改范围：** 仅影响 `use_parallel=True` 的多卡路径，单卡路径完全不受影响

---

## 修改前的问题

多卡 DDP 训练速度反而比单卡慢，经排查发现三个根本性 bug：

### Bug 1：梯度同步完全失效
- **位置：** `train_distributed.py:195`（修改前）
- **原因：** 调用 `model.module.loss(...)` 绕过了 DDP wrapper，DDP 的梯度 all-reduce hook 从未被注册
- **后果：** 每个 GPU 独立训练，没有任何梯度同步，唯一跨卡通信是 `reduce_tensor` 对 loss 值的汇总（仅用于日志）

### Bug 2：权重初始化竞态条件
- **位置：** `train_distributed.py:101-103`（修改前）
- **原因：** 只有 rank 0 调用了 `_init_weights()`，其他 rank 的模型参数是随机初始化的，且没有广播同步
- **后果：** 各 GPU 从不同的初始权重开始训练，即使梯度同步修复后也会导致训练不一致

### Bug 3：梯度累加逻辑无效
- **位置：** `train_distributed.py:204-211`（修改前）
- **原因：** `no_sync()` 和 `accumulation_steps` 逻辑依赖 DDP hook 注册，而 Bug 1 导致 hook 从未注册
- **后果：** `no_sync()` 实际上是空操作，所有 backward 调用行为完全一致

---

## 修改内容

### 修改 1：新增 `compute_loss()` 方法
- **文件：** `model/PI_DeepOnet.py`（line 456-498，在原 `loss()` 方法之后）
- **内容：** 从 `loss()` 中提取纯损失计算逻辑，接受 `Delta_U`（forward 输出）作为参数，不包含 forward 调用
- **目的：** 允许 DDP 训练先通过 wrapper 调用 `forward()`（注册梯度 hook），再调用此方法计算 loss
- **单卡影响：** 无。单卡路径仍调用原 `loss()` 方法

```
调用方式对比：
  单卡：model.loss(vel, y, UU0, labels, ...)          # 内部包含 forward
  多卡：model(vel, y_combined, UU0)                     # 通过 DDP wrapper 调用 forward
        model.module.compute_loss(Delta_U, vel, y, ...)  # 纯 loss 计算
```

### 修改 2：权重初始化广播
- **文件：** `model/train_distributed.py`（line 105-107）
- **内容：** 在 rank 0 初始化权重后，通过 `dist.broadcast(param.data, src=0)` 广播所有参数给其他 rank
- **位置：** 在 `_init_weights()` 之后、`wrap_model_for_distributed()` 之前
- **单卡影响：** 无。此代码仅在 `train_distributed.py` 中

### 修改 3：训练循环重写
- **文件：** `model/train_distributed.py`（line 160-233）
- **改动要点：**

| 项目 | 修改前 | 修改后 |
|------|--------|--------|
| 前向传播 | `model.module.loss()`（绕过 DDP） | `model()` 通过 DDP wrapper + `model.module.compute_loss()` |
| 梯度同步粒度 | 每个 coord batch（但因 Bug 1 实际未同步） | 每个 vel batch 同步一次（内层用 `no_sync` 累加） |
| loss 缩放 | `loss / accumulation_steps` | `loss / n_coord`（coord batch 数量） |
| optimizer 更新 | 按 `step_counter % accumulation_steps` 触发 | 每个 vel batch 结束后触发 |
| step_counter | 使用全局计数器 | 已移除 |
| coord batches | 直接遍历 `dataloader['train_y']` | 预收集为 `list(dataloader['train_y'])` 以获取 `n_coord` |

**梯度同步策略流程：**
```
for vel_batch in train_loader (DistributedSampler 分片):
    y_ran = generate_adaptive_pde_points(vel_batch)
    coord_batches = list(train_y)
    n_coord = len(coord_batches)

    for idx, coord_batch in enumerate(coord_batches):
        y_combined = cat(coord_batch, y_ran)
        Delta_U = model(vel, y_combined, UU0)        # DDP forward
        loss = model.module.compute_loss(Delta_U, ...) / n_coord

        if idx < n_coord - 1:
            with model.no_sync():
                loss.backward()                       # 累加梯度，不同步
        else:
            loss.backward()                           # 累加 + DDP all-reduce

    optimizer.step()
    optimizer.zero_grad()
```

### 修改 4：启用 `find_unused_parameters`
- **文件：** `model/utils.py`（line 283）
- **内容：** `find_unused_parameters=False` → `True`
- **原因：** 模型中存在未在 `forward()` 中使用的参数（`log_var_data`、`log_var_pde`、`fencoder` 模块参数）。修复 DDP forward 后，DDP 会检测这些未使用参数并在 backward 时报错。设为 `True` 是修复 Bug 1 的必要依赖
- **性能影响：** `True` 会增加少量通信开销（DDP 需要追踪哪些参数被使用），但对 2 卡场景影响可忽略
- **单卡影响：** 无。`wrap_model_for_distributed()` 仅在多卡路径调用

---

## 未修改的文件

| 文件 | 说明 |
|------|------|
| `model/train.py` | 单卡训练，未修改 |
| `model/test.py` | 测试脚本，未修改 |
| `model/dataloader.py` | 数据加载，未修改 |
| `model/plotting.py` | 绘图/微调，未修改 |
| `model/FNO.py` | FNO 模型，未修改 |
| `model/net_module.py` | 网络模块，未修改 |
| `config.py` | 配置参数，未修改 |
| `main2.py` | 入口文件，未修改 |

---

## 验证要点

### 1. 单卡回归验证（`use_parallel=False`）
- [ ] 单卡训练正常运行，loss 下降趋势与修改前一致
- [ ] 验证/绘图/保存功能正常
- [ ] 微调（fine_tuning）功能正常

### 2. DDP 功能验证（`use_parallel=True, num_gpus=2`）
- [ ] 两张 GPU 均被使用（`nvidia-smi` 观察显存占用）
- [ ] 训练正常启动，无 DDP 报错
- [ ] Loss 下降趋势合理（与单卡可比）
- [ ] 每个 epoch 的训练速度比单卡快 ~1.7-1.9 倍
- [ ] 模型保存/加载正常（保存的是 `model.module.state_dict()`，可直接用于单卡加载）

### 3. 如遇问题排查
- **DDP 报错 "RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one"**
  → 检查内层循环中 `no_sync()` 是否正确包裹了除最后一个 coord batch 外的所有 backward
- **某张 GPU 显存为 0**
  → 检查 `dist.broadcast()` 是否在 `wrap_model_for_distributed()` 之前执行
- **训练 loss 不下降**
  → 检查 `optimizer.step()` 是否在内层循环结束后（外层 vel batch 末尾）执行

---

## 后续优化方向（本次未实施）

| 优化项 | 说明 | 优先级 |
|--------|------|--------|
| 数据加载 mmap | `np.load(mmap_mode='r')` 减少 80GB 数据的重复加载 | 中 |
| 验证集 DistributedSampler | 验证集也分片到各 GPU，减少 rank 0 验证耗时 | 低 |
| 混合精度训练 (AMP) | `torch.cuda.amp` 进一步加速 | 低 |
| 去除未使用参数 | 移除 `log_var_data`、`log_var_pde`、`fencoder`，可将 `find_unused_parameters` 改回 `False` | 低 |
