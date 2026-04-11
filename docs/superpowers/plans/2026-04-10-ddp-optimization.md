# DDP Distributed Training Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three critical DDP bugs that prevent gradient synchronization from working, so multi-GPU training actually accelerates.

**Architecture:** Split `Pi_DeepONet.loss()` into `forward()` (through DDP wrapper) + `compute_loss()` (post-forward loss calculation). Restructure training loop to use `no_sync()` for inner coordinate loop, syncing once per velocity batch. Broadcast initial weights from rank 0.

**Tech Stack:** PyTorch DDP (nccl), torch.distributed, torch.autograd

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `model/PI_DeepOnet.py` | Modify | Add `compute_loss()` method (extracted from `loss()`) |
| `model/train_distributed.py` | Modify | Fix weight init, training loop, gradient sync |
| `model/utils.py` | Modify | Change `find_unused_parameters` to `True` |
| `model/train.py` | No change | Single-GPU path unaffected |

---

### Task 1: Add `compute_loss()` to `Pi_DeepONet`

**Files:**
- Modify: `model/PI_DeepOnet.py` (add method after `loss()` at line ~454)

**Why:** The current `loss()` method combines `forward()` + loss computation in one call. DDP needs `forward()` to go through the wrapper to register gradient hooks. We extract the loss computation into a separate method that takes the forward output as input.

- [ ] **Step 1: Add `compute_loss()` method to `Pi_DeepONet`**

Insert the following method after the existing `loss()` method (after line 454 in `model/PI_DeepOnet.py`):

```python
    def compute_loss(self, Delta_U, vel, y, UU0, labels, y_combined,
                     a, b, c, data_norm_coe=1., pde_norm_coe=1., freq_batch=None):
        """
        在 DDP forward 之后计算损失 (不包含 forward 调用)。
        供 DDP 训练使用：先通过 DDP wrapper 调用 forward()，再调用此方法计算 loss。

        Args:
            Delta_U: model.forward() 的输出 [B_v, B_pts, 2]
            vel: 速度场 [B_v, 1, Z, X]
            y: 数据坐标点 [B_v, B_data_pts, 2]
            UU0: 背景波场 [B_v, 2, Z, X]
            labels: 标签波场 [B_v, 2, Z, X]
            y_combined: 拼接后的坐标 [B_v, B_data_pts + B_ran_pts, 2]，requires_grad=True
            a, b, c: 损失权重
            data_norm_coe: 数据损失归一化系数
            pde_norm_coe: PDE 损失归一化系数
            freq_batch: 频率值 [B_v]
        Returns:
            (total_loss, loss_f, loss_u, loss_r) — 与 loss() 返回格式一致
        """
        batch_size_v = vel.shape[0]
        nz, nx = vel.shape[2], vel.shape[3]
        n_y = y.shape[1]

        # 1. 提取标签值 (与 loss() 中相同)
        batch_idx = torch.arange(batch_size_v, device=labels.device)[:, None]
        z_coord = (y[:, :, 0] / self.args.dh).long().clamp(0, nz - 1)
        x_coord = (y[:, :, 1] / self.args.dh).long().clamp(0, nx - 1)
        labels_extracted = labels[batch_idx, :, z_coord, x_coord]

        # 2. 数据拟合损失
        pred_y = Delta_U[:, :n_y, :]
        loss_u = self.loss_function(pred_y, labels_extracted) / data_norm_coe

        # 3. PDE 物理残差损失
        loss_f = self._compute_pde_residual(vel, y_combined, UU0, Delta_U, freq_batch=freq_batch) / pde_norm_coe

        loss_r = 0.0

        # 4. 加权求和
        loss_val = (a * loss_u) + b * loss_f

        return loss_val, loss_f, loss_u, loss_r
```

- [ ] **Step 2: Verify no syntax errors**

Run: `cd /home/zhangdaoguang/Code/Physical_Informed_DeepONet && python -c "from model.PI_DeepOnet import Pi_DeepONet; print('Import OK')"`
Expected: `Import OK`

- [ ] **Step 3: Commit**

```bash
git add model/PI_DeepOnet.py
git commit -m "feat: add compute_loss() method for DDP-compatible loss computation"
```

---

### Task 2: Fix weight initialization broadcast

**Files:**
- Modify: `model/train_distributed.py:99-106`

**Why:** Currently only rank 0 calls `_init_weights()`. Other ranks get random weights and are never synchronized. DDP snapshots model state at wrapper construction, so broadcast must happen before wrapping.

- [ ] **Step 1: Add weight broadcast between init and DDP wrap**

In `model/train_distributed.py`, replace lines 99-106 (from `model = Pi_DeepONet(args).to(device)` through `model = wrap_model_for_distributed(model, rank)`) with:

```python
    model = Pi_DeepONet(args).to(device)

    if is_main_process(rank):
        model._init_weights()
        print(f"PI_DeepONet 模型总参数数量: {count_parameters(model)}")

    # 广播所有参数从 rank 0 到其他 rank (必须在 DDP wrap 之前)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # 包装模型为 DDP
    model = wrap_model_for_distributed(model, rank)
    if is_main_process(rank):
        print("模型已包装为 DistributedDataParallel")
```

Key changes:
- Added `dist.broadcast(param.data, src=0)` loop between init and DDP wrap
- This ensures all ranks start with identical weights before DDP snapshots them

- [ ] **Step 2: Commit**

```bash
git add model/train_distributed.py
git commit -m "fix: broadcast model weights from rank 0 before DDP wrap"
```

---

### Task 3: Fix DDP forward pass and gradient sync strategy

**Files:**
- Modify: `model/train_distributed.py:156-218` (main training loop)

**Why:** `model.module.loss()` bypasses DDP wrapper, so gradient all-reduce never fires. Restructure to call `model()` through DDP wrapper, then `model.module.compute_loss()` for loss calculation. Use `no_sync()` for all but the last coordinate batch to sync once per velocity batch.

- [ ] **Step 1: Replace the training loop body**

In `model/train_distributed.py`, replace the entire training loop body from line 156 (`step_counter = 0`) through line 218 (`del loss, loss_f, loss_u, loss_r, y_batch`).

Replace from `step_counter = 0` through the end of the inner loop (line 218) with:

```python
    for i in pbar:
        # 动态调整损失权重
        if args.if_adjust and i > args.adjust_from and (i - args.adjust_from) % args.adjust_every == 0:
            decay_times = i // args.adjust_every
            a = max(a * (args.adjust_speed ** (-decay_times)), 2e-1)
            b, c = 1, 0

        model.train()
        batch_loss, batch_u_loss, batch_f_loss, batch_r_loss = [], [], [], []

        # 设置 epoch 以确保每个 epoch shuffle 不同
        dataloader['train'].sampler.set_epoch(i)

        # 遍历训练数据
        for batch_data in dataloader['train']:
            if has_freq:
                vel_batch, UU0_batch, labels_batch, freq_batch = batch_data
                freq_batch = freq_batch.to(device)
            else:
                vel_batch, UU0_batch, labels_batch = batch_data
                freq_batch = None
            vel_batch, UU0_batch = vel_batch.to(device), UU0_batch.to(device)

            if args.use_fno_as_label:
                with torch.no_grad():
                    labels_batch = fno(vel_batch, UU0_batch).to(device)
            else:
                labels_batch = labels_batch.to(device)

            # 每个 velocity batch 只生成一次自适应采样点
            with torch.no_grad():
                y_ran = model.module.generate_structure_aware_y_ran(vel_batch, num_pts=900)

            # 预收集 coordinate batches 以确定总数
            coord_batches = list(dataloader['train_y'])
            n_coord = len(coord_batches)

            for idx, batch in enumerate(coord_batches):
                y_batch = batch[0].to(device)
                y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                # 拼接数据坐标和 PDE 采样坐标
                y_combined = torch.cat([y_batch, y_ran], dim=1)
                y_combined.requires_grad_(True)

                # 通过 DDP wrapper 调用 forward (触发梯度同步 hook)
                Delta_U = model(vel_batch, y_combined, UU0_batch)

                # 计算损失 (不调用 forward)
                loss, loss_f, loss_u, loss_r = model.module.compute_loss(
                    Delta_U, vel_batch, y_batch, UU0_batch, labels_batch,
                    y_combined, a, b, c, data_norm_coe, pde_norm_coe,
                    freq_batch=freq_batch
                )

                loss = loss / n_coord

                # 梯度同步策略: 仅最后一个 coord batch 触发 DDP all-reduce
                if idx < n_coord - 1:
                    with model.no_sync():
                        loss.backward()
                else:
                    loss.backward()           # all-reduce 在此触发

                batch_loss.append(loss.item() * n_coord)
                batch_u_loss.append(loss_u.item())
                batch_f_loss.append(loss_f.item())
                batch_r_loss.append(loss_r.item() if isinstance(loss_r, torch.Tensor) else loss_r)

                del loss, loss_f, loss_u, loss_r, y_batch, Delta_U

            # 每个 velocity batch 结束后更新参数
            optimizer.step()
            optimizer.zero_grad()
```

Key differences from original:
1. **`Delta_U = model(vel_batch, y_combined, UU0_batch)`** — calls forward through DDP wrapper (was `model.module.loss()`)
2. **`model.module.compute_loss(Delta_U, ...)`** — computes loss without forward (new method from Task 1)
3. **`coord_batches = list(dataloader['train_y'])`** — materialize once to get `n_coord`
4. **`loss = loss / n_coord`** — average over coord batches instead of `accumulation_steps`
5. **`no_sync()` on all but last coord batch** — sync once per velocity batch
6. **`optimizer.step()` + `zero_grad()` moved to outer loop** — once per velocity batch
7. Removed `step_counter` and `accumulation_steps` logic

- [ ] **Step 2: Verify no syntax errors**

Run: `cd /home/zhangdaoguang/Code/Physical_Informed_DeepONet && python -c "from model.train_distributed import train_distributed; print('Import OK')"`
Expected: `Import OK`

- [ ] **Step 3: Commit**

```bash
git add model/train_distributed.py
git commit -m "fix: DDP forward through wrapper + no_sync gradient sync per vel batch"
```

---

### Task 4: Enable `find_unused_parameters`

**Files:**
- Modify: `model/utils.py:283`

**Why:** The model has parameters that are never used in `forward()`: `log_var_data`, `log_var_pde`, and the `fencoder` module's parameters. With DDP properly intercepting `forward()`, `find_unused_parameters=False` will crash DDP during backward. This is a **necessary consequence** of fixing the DDP forward pass (Task 3), not an optional optimization.

- [ ] **Step 1: Change `find_unused_parameters` to `True`**

In `model/utils.py` line 283, change:

```python
# Before:
    model = DDP.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)

# After:
    model = DDP.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
```

- [ ] **Step 2: Commit**

```bash
git add model/utils.py
git commit -m "fix: enable find_unused_parameters for DDP compatibility"
```

---

### Task 5: Smoke test — verify DDP initialization and gradient sync

**Files:**
- No new files

**Why:** Before running a full training run, verify that DDP setup, weight broadcast, forward through wrapper, and gradient all-reduce all work correctly.

- [ ] **Step 1: Create a minimal DDP smoke test**

Run the following command to test DDP initialization and a single forward-backward pass:

```bash
cd /home/zhangdaoguang/Code/Physical_Informed_DeepONet && python -c "
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from model.PI_DeepOnet import Pi_DeepONet
from model.utils import setup_distributed, cleanup_distributed

class TestArgs:
    device = 0
    dh = 40
    nz = 140
    nx = 140
    input_shape_branch1 = (30, 1, 140, 140)
    input_shape_branch2 = (30, 2, 140, 140)
    batch_size = 700
    pml_active = 5
    boundary_type = 'free_surface'
    default_freq = 5.0

def worker(rank, world_size):
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    args = TestArgs()
    args.device = rank
    model = Pi_DeepONet(args).to(device)

    if rank == 0:
        model._init_weights()

    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], find_unused_parameters=True
    )

    # Single forward-backward
    vel = torch.randn(2, 1, 140, 140, device=device)
    y = torch.rand(2, 100, 2, device=device) * 5000
    y.requires_grad_(True)
    UU0 = torch.randn(2, 2, 140, 140, device=device)

    out = model(vel, y, UU0)
    loss = out.sum()
    loss.backward()

    print(f'[Rank {rank}] Forward shape: {out.shape}, Backward OK')

    cleanup_distributed()

mp.spawn(worker, args=(2,), nprocs=2, join=True)
print('DDP smoke test PASSED')
"
```

Expected output:
```
[Rank 0] Forward shape: torch.Size([2, 100, 2]), Backward OK
[Rank 1] Forward shape: torch.Size([2, 100, 2]), Backward OK
DDP smoke test PASSED
```

If this fails, debug before proceeding.

- [ ] **Step 2: If smoke test passes, commit no changes (verification only)**

---

## Self-Review

**Spec coverage:**
- Section 1 (DDP forward fix) → Task 1 + Task 3
- Section 2 (no_sync strategy) → Task 3
- Section 3 (weight init broadcast) → Task 2
- Section 4 (find_unused_parameters) → Task 4 (necessary dependency of Section 1)

**Placeholder scan:** No TBD, TODO, or vague steps. All code blocks contain complete implementation.

**Type consistency:**
- `compute_loss()` signature matches the call in Task 3: `model.module.compute_loss(Delta_U, vel_batch, y_batch, UU0_batch, labels_batch, y_combined, a, b, c, data_norm_coe, pde_norm_coe, freq_batch=freq_batch)`
- Return type matches: `(loss_val, loss_f, loss_u, loss_r)` — same as `loss()`
- `y_combined` constructed identically in Task 3 as in original `loss()`: `torch.cat([y_batch, y_ran], dim=1)` with `requires_grad_(True)`
