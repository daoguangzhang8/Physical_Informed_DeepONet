# 课程学习 Replay 防遗忘 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在课程学习的 Stage 1/2 中混入所有前序阶段数据，防止灾难性遗忘。

**Architecture:** 修改 `_train_stage` 函数，在加载当前阶段数据后遍历 `replay_stages` 配置，依次加载前序阶段数据并 cat 拼接到训练集 Tensor，重建 DataLoader。

**Tech Stack:** PyTorch

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `config.py` | 每个 stage 新增 `replay_stages` 配置 |
| Modify | `model/train.py` | `_train_stage` 中增加 replay 数据加载和拼接 |

---

### Task 1: 在 config.py 的 stages 配置中新增 replay_stages

**Files:**
- Modify: `config.py:153-181` (stages 列表)

- [ ] **Step 1: 为每个 stage 配置添加 replay_stages 字段**

在 `config.py` 中，将现有的 `stages` 列表替换为（每个 dict 新增 `replay_stages` 键）：

```python
    stages = [
        {
            'name': 'low_freq',
            'freq_range': '3to11',
            'freq_min': 3.0, 'freq_max': 11.0,
            'NIter': 3500,
            'lr': 1e-4,
            'warmup_epochs': 100,
            'a': 1, 'b': 1, 'c': 0,
            'replay_stages': [],           # Stage 0: 不 replay
        },
        {
            'name': 'mid_freq',
            'freq_range': '12to18',
            'freq_min': 12.0, 'freq_max': 18.0,
            'NIter': 3500,
            'lr': 5e-5,
            'warmup_epochs': 50,
            'a': 1, 'b': 1, 'c': 0,
            'replay_stages': [0],          # Stage 1: replay Stage 0
        },
        {
            'name': 'high_freq',
            'freq_range': '18to25',
            'freq_min': 18.0, 'freq_max': 25.0,
            'NIter': 3500,
            'lr': 2e-5,
            'warmup_epochs': 50,
            'a': 1, 'b': 1, 'c': 0,
            'replay_stages': [0, 1],       # Stage 2: replay Stage 0 和 1
        },
    ]
```

- [ ] **Step 2: 验证语法正确**

Run: `cd /home/zhangdaoguang/Code/Physical_Informed_DeepONet && python -c "from config import Args; s = Args.stages; print('Stage 0 replay:', s[0].get('replay_stages', [])); print('Stage 1 replay:', s[1].get('replay_stages', [])); print('Stage 2 replay:', s[2].get('replay_stages', []))"`

Expected: 输出 `Stage 0 replay: []`, `Stage 1 replay: [0]`, `Stage 2 replay: [0, 1]`

---

### Task 2: 在 _train_stage 中实现 replay 数据加载与拼接

**Files:**
- Modify: `model/train.py:65-66` (`_train_stage` 中 `prepare_training_dataloaders` 调用处及之后)

- [ ] **Step 1: 在 `_train_stage` 中，替换数据加载部分**

在 `_train_stage` 函数中，找到以下代码（约第 65-66 行）：

```python
    # ---- 3. 加载数据 ----
    dataloader, plot_data = prepare_training_dataloaders(args, device)
```

替换为：

```python
    # ---- 3. 加载数据 ----
    dataloader, plot_data = prepare_training_dataloaders(args, device)

    # ---- 3.5 Replay 前序阶段数据（防遗忘） ----
    replay_stages = stage_config.get('replay_stages', [])
    if replay_stages and stage_idx > 0:
        # 保存当前阶段的文件名（replay 后需恢复）
        cur_vel_fn = args.vel_filename
        cur_bg_fn = args.backgroundfield_filename
        cur_wf_fn = args.wavefield_filename
        cur_freq_fn = args.freq_filename

        # 从当前 DataLoader 提取训练集 Tensor
        train_ds = dataloader['train'].dataset
        train_tensors = train_ds.tensors
        has_freq = len(train_tensors) >= 4
        if has_freq:
            combined_vel = train_tensors[0]
            combined_UU0 = train_tensors[1]
            combined_labels = train_tensors[2]
            combined_freq = train_tensors[3]
        else:
            combined_vel = train_tensors[0]
            combined_UU0 = train_tensors[1]
            combined_labels = train_tensors[2]
            combined_freq = None

        for replay_idx in replay_stages:
            replay_config = args.stages[replay_idx]
            replay_freq_tag = f'freq{replay_config["freq_range"]}'

            # 替换文件名为 replay 阶段
            args.vel_filename = base_vel_filename.replace(base_freq_tag, replay_freq_tag)
            args.backgroundfield_filename = base_bg_filename.replace(base_freq_tag, replay_freq_tag)
            args.wavefield_filename = base_wf_filename.replace(base_freq_tag, replay_freq_tag)
            args.freq_filename = base_freq_filename.replace('freq_used', f'freq{replay_config["freq_range"]}_used')

            print(f'    [Replay] 加载 Stage {replay_idx} [{replay_config["name"]}] 数据: {args.vel_filename}')

            replay_dl, _ = prepare_training_dataloaders(args, device)
            replay_ds = replay_dl['train'].dataset
            replay_tensors = replay_ds.tensors

            combined_vel = torch.cat([combined_vel, replay_tensors[0]], dim=0)
            combined_UU0 = torch.cat([combined_UU0, replay_tensors[1]], dim=0)
            combined_labels = torch.cat([combined_labels, replay_tensors[2]], dim=0)
            if has_freq and len(replay_tensors) >= 4:
                combined_freq = torch.cat([combined_freq, replay_tensors[3]], dim=0)

        # 恢复当前阶段的文件名
        args.vel_filename = cur_vel_fn
        args.backgroundfield_filename = cur_bg_fn
        args.wavefield_filename = cur_wf_fn
        args.freq_filename = cur_freq_fn

        # 重建训练 DataLoader
        pin_mem = device.type == 'cuda'
        num_workers = 4
        prefetch_factor = 2

        if has_freq:
            new_train_ds = TensorDataset(combined_vel, combined_UU0, combined_labels, combined_freq)
        else:
            new_train_ds = TensorDataset(combined_vel, combined_UU0, combined_labels)

        dataloader['train'] = DataLoader(
            new_train_ds,
            batch_size=args.batch_size_v, shuffle=True, drop_last=True,
            pin_memory=pin_mem, num_workers=num_workers, prefetch_factor=prefetch_factor,
        )

        print(f'    [Replay] 训练集合并完成: {combined_vel.shape[0]} 样本 (含 replay)')
```

注意：需要在文件顶部确认 `TensorDataset` 和 `DataLoader` 的 import 可用。当前 `train.py` 通过 `from Labconfig import *` 已导入 `DataLoader` 和 `TensorDataset`，无需额外 import。

- [ ] **Step 2: 验证语法正确**

Run: `cd /home/zhangdaoguang/Code/Physical_Informed_DeepONet && python -c "from model.train import _train_stage; print('_train_stage loaded OK')"`

Expected: 输出 `_train_stage loaded OK`

---

### Task 3: 冒烟测试 — 验证 replay 逻辑可正确运行

**Files:**
- No file changes, verification only

- [ ] **Step 1: 运行冒烟测试**

Run:
```bash
cd /home/zhangdaoguang/Code/Physical_Informed_DeepONet && python -c "
from config import Args
args = Args()

# 验证 replay_stages 配置正确
for i, s in enumerate(args.stages):
    rp = s.get('replay_stages', [])
    print(f'Stage {i} [{s[\"name\"]}]: replay_stages = {rp}')

# 验证 replay 配置语义
assert args.stages[0].get('replay_stages', []) == [], 'Stage 0 should have no replay'
assert args.stages[1].get('replay_stages', []) == [0], 'Stage 1 should replay Stage 0'
assert args.stages[2].get('replay_stages', []) == [0, 1], 'Stage 2 should replay Stage 0 and 1'

# 验证 _train_stage 可正常导入
from model.train import _train_stage
print()
print('ALL TESTS PASSED')
"
```

Expected: 输出每个 stage 的 replay 配置和 `ALL TESTS PASSED`
