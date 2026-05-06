# Epoch-Level 共享 y_ran 采样 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 y_ran 采样从 per-model 自适应改为 epoch 级全局结构概率图 + 共享采样（实验A：60% epoch-structure + 20% surface + 20% uniform）

**Architecture:** 每个 epoch 开始时遍历训练集所有 velocity model 计算平均速度梯度概率图，每个 velocity batch 从该概率图采样一组共享 y_ran，expand 到 batch 维度后传入 loss 计算。

**Tech Stack:** PyTorch, NumPy

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `config.py` | 新增 y_ran 采样配置参数 |
| Modify | `model/utils.py` | 新增 `build_epoch_velocity_gradient_prob()` 和 `sample_shared_y_ran_from_epoch_prob()` |
| Modify | `model/train.py` | `train_single()` 和 `_train_stage()` 中替换 y_ran 生成逻辑 |
| Modify | `model/train_distributed.py` | `_train_worker()` 中替换 y_ran 生成逻辑 |

---

### Task 1: 在 config.py 新增 y_ran 采样配置参数

**Files:**
- Modify: `config.py:181` (在 `stages` 定义之后追加)

- [ ] **Step 1: 在 Args 类末尾追加配置参数**

在 `config.py` 的 `Args` 类中，`stages` 列表定义之后（第 181 行之后），追加以下代码：

```python

    # ==========================================
    # 13. y_ran Epoch-Level 共享采样 (Epoch Shared Sampling)
    # ==========================================
    use_epoch_shared_y_ran = True              # True: 使用 epoch 级共享采样 | False: 使用原始 per-model 采样

    y_ran_num_pts = 900                        # y_ran 采样点总数
    y_ran_structure_ratio = 0.60               # epoch-structure 采样点比例
    y_ran_surface_ratio = 0.20                 # 表层采样点比例
    y_ran_uniform_ratio = 0.20                 # 均匀采样点比例
    y_ran_source_ratio = 0.0                   # 震源附近采样点比例 (实验A不使用)

    y_ran_surface_depth_grids = 5              # 表层深度（网格点数）
    y_ran_use_max_mix = False                  # True: score = mean_weight*mean + max_weight*max | False: 纯 mean
    y_ran_mean_weight = 0.7                    # mean+max 混合时的 mean 权重
    y_ran_max_weight = 0.3                     # mean+max 混合时的 max 权重

    # 概率图更新频率: 1=每epoch, >1=每N个epoch, 0=只计算一次并缓存
    y_ran_prob_update_every = 1
```

- [ ] **Step 2: 验证语法正确**

Run: `cd /home/zhangdaoguang/Code/Physical_Informed_DeepONet && python -c "from config import Args; print('use_epoch_shared_y_ran:', Args.use_epoch_shared_y_ran); print('y_ran_num_pts:', Args.y_ran_num_pts); print('y_ran_structure_ratio:', Args.y_ran_structure_ratio)"`

Expected: 输出三个参数值，无报错

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add epoch-level shared y_ran sampling config parameters"
```

---

### Task 2: 在 model/utils.py 新增 build_epoch_velocity_gradient_prob 函数

**Files:**
- Modify: `model/utils.py:505` (在文件末尾 `get_helmholtz_spatial_weights` 函数之后追加)

- [ ] **Step 1: 在 model/utils.py 末尾追加函数**

在 `model/utils.py` 文件末尾（第 506 行之后），追加以下代码：

```python

def build_epoch_velocity_gradient_prob(
    train_loader,
    device,
    eps=1e-8,
    use_max_mix=False,
    mean_weight=0.7,
    max_weight=0.3,
):
    """
    基于整个训练集 velocity model 构造 epoch-level 结构采样概率图。

    Args:
        train_loader: 训练数据 DataLoader，每个 batch 至少包含 vel 在 index 0
        device: 计算设备
        eps: 数值稳定项
        use_max_mix: True 时 score = mean_weight*mean + max_weight*max
        mean_weight: mean+max 混合的 mean 权重
        max_weight: mean+max 混合的 max 权重

    Returns:
        prob: [Z*X] 扁平概率分布，归一化后 sum=1
        score: [Z, X] 原始分数图（用于可视化诊断）
    """
    score_sum = None
    score_max_global = None
    count = 0

    with torch.no_grad():
        for batch_data in train_loader:
            vel_batch = batch_data[0]
            vel = vel_batch.to(device)  # [B, 1, Z, X]
            B, _, Z, X = vel.shape

            grad_z = vel[:, :, 2:, 1:-1] - vel[:, :, :-2, 1:-1]
            grad_x = vel[:, :, 1:-1, 2:] - vel[:, :, 1:-1, :-2]

            grad_mag = torch.sqrt(grad_z ** 2 + grad_x ** 2 + eps)
            grad_mag = F.pad(grad_mag, (1, 1, 1, 1), mode='replicate').squeeze(1)  # [B, Z, X]

            batch_sum = grad_mag.sum(dim=0)  # [Z, X]

            if score_sum is None:
                score_sum = torch.zeros_like(batch_sum)
                score_max_global = torch.zeros_like(batch_sum)

            score_sum += batch_sum
            count += B

            if use_max_mix:
                batch_max = grad_mag.max(dim=0).values
                score_max_global = torch.maximum(score_max_global, batch_max)

    score_mean = score_sum / max(count, 1)

    if use_max_mix:
        score = mean_weight * score_mean + max_weight * score_max_global
    else:
        score = score_mean

    score = torch.clamp(score, min=0.0)
    prob = score.reshape(-1)
    prob = prob / (prob.sum() + eps)

    return prob, score
```

- [ ] **Step 2: 验证语法正确**

Run: `cd /home/zhangdaoguang/Code/Physical_Informed_DeepONet && python -c "from model.utils import build_epoch_velocity_gradient_prob; print('build_epoch_velocity_gradient_prob loaded OK')"`

Expected: 输出 `build_epoch_velocity_gradient_prob loaded OK`

- [ ] **Step 3: Commit**

```bash
git add model/utils.py
git commit -m "feat: add build_epoch_velocity_gradient_prob function"
```

---

### Task 3: 在 model/utils.py 新增 sample_shared_y_ran_from_epoch_prob 函数

**Files:**
- Modify: `model/utils.py` (在 Task 2 新增函数之后追加)

- [ ] **Step 1: 在 build_epoch_velocity_gradient_prob 函数之后追加函数**

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
    """
    从 epoch-level 概率图中采样一组共享 y_ran。

    Args:
        prob: [Z*X] 扁平概率分布 (from build_epoch_velocity_gradient_prob)
        args: 配置参数，需包含 nz, nx, dh
        num_pts: 总采样点数
        structure_ratio: epoch-structure 采样比例
        surface_ratio: 表层采样比例
        uniform_ratio: 均匀采样比例
        source_ratio: 震源附近采样比例
        source_coords: 震源坐标 [N_src, 2] (物理坐标 [z, x])
        surface_depth_grids: 表层深度（网格点数）
        source_r_min_grids: 震源采样内环半径（网格点数）
        source_r_max_grids: 震源采样外环半径（网格点数）
        replacement: multinomial 是否允许重复

    Returns:
        y_shared: [num_pts, 2]，坐标格式 [z, x]，物理坐标
    """
    device = prob.device
    nz, nx, dh = args.nz, args.nx, args.dh

    max_z = nz * dh
    max_x = nx * dh

    # 如果 source 未启用，将其份额归入 uniform
    if source_coords is None or source_ratio <= 0:
        uniform_ratio = uniform_ratio + source_ratio
        source_ratio = 0.0

    num_structure = int(num_pts * structure_ratio)
    num_surface = int(num_pts * surface_ratio)
    num_source = int(num_pts * source_ratio)
    num_uniform = num_pts - num_structure - num_surface - num_source

    y_parts = []

    # 1. epoch-level structure points
    if num_structure > 0:
        sampled_indices = torch.multinomial(
            prob,
            num_samples=num_structure,
            replacement=replacement,
        )

        z_idx = sampled_indices // nx
        x_idx = sampled_indices % nx

        z = z_idx.float() * dh + torch.rand(num_structure, device=device) * dh
        x = x_idx.float() * dh + torch.rand(num_structure, device=device) * dh

        z = z.clamp(0.0, max_z)
        x = x.clamp(0.0, max_x)

        y_struct = torch.stack([z, x], dim=-1)
        y_parts.append(y_struct)

    # 2. surface points
    if num_surface > 0:
        surface_depth = surface_depth_grids * dh

        z = torch.rand(num_surface, device=device) * surface_depth
        x = torch.rand(num_surface, device=device) * max_x

        z = z.clamp(0.0, max_z)
        x = x.clamp(0.0, max_x)

        y_surface = torch.stack([z, x], dim=-1)
        y_parts.append(y_surface)

    # 3. source-near points
    if num_source > 0 and source_coords is not None:
        source_coords = source_coords.to(device)

        src_id = torch.randint(
            low=0,
            high=source_coords.shape[0],
            size=(num_source,),
            device=device,
        )
        src = source_coords[src_id]

        theta = 2.0 * torch.pi * torch.rand(num_source, device=device)

        r_min = source_r_min_grids * dh
        r_max = source_r_max_grids * dh
        r = r_min + (r_max - r_min) * torch.rand(num_source, device=device)

        z = src[:, 0] + r * torch.cos(theta)
        x = src[:, 1] + r * torch.sin(theta)

        z = z.clamp(0.0, max_z)
        x = x.clamp(0.0, max_x)

        y_source = torch.stack([z, x], dim=-1)
        y_parts.append(y_source)

    # 4. uniform points
    if num_uniform > 0:
        z = torch.rand(num_uniform, device=device) * max_z
        x = torch.rand(num_uniform, device=device) * max_x

        z = z.clamp(0.0, max_z)
        x = x.clamp(0.0, max_x)

        y_uniform = torch.stack([z, x], dim=-1)
        y_parts.append(y_uniform)

    y_shared = torch.cat(y_parts, dim=0)

    # 修正四舍五入造成的点数误差
    if y_shared.shape[0] > num_pts:
        y_shared = y_shared[:num_pts]
    elif y_shared.shape[0] < num_pts:
        extra = num_pts - y_shared.shape[0]
        z = torch.rand(extra, device=device) * max_z
        x = torch.rand(extra, device=device) * max_x
        y_extra = torch.stack([z, x], dim=-1)
        y_shared = torch.cat([y_shared, y_extra], dim=0)

    return y_shared
```

- [ ] **Step 2: 验证语法正确**

Run: `cd /home/zhangdaoguang/Code/Physical_Informed_DeepONet && python -c "from model.utils import sample_shared_y_ran_from_epoch_prob; print('sample_shared_y_ran_from_epoch_prob loaded OK')"`

Expected: 输出 `sample_shared_y_ran_from_epoch_prob loaded OK`

- [ ] **Step 3: Commit**

```bash
git add model/utils.py
git commit -m "feat: add sample_shared_y_ran_from_epoch_prob function"
```

---

### Task 4: 修改 model/train.py 的 train_single 函数

**Files:**
- Modify: `model/train.py:1-9` (import 区域)
- Modify: `model/train.py:406-498` (train_single 内的 y_ran 逻辑)

- [ ] **Step 1: 更新 import**

在 `model/train.py` 第 8 行 `from model.utils import *` 之后追加：

```python
from model.utils import build_epoch_velocity_gradient_prob, sample_shared_y_ran_from_epoch_prob
```

- [ ] **Step 2: 在 train_single 函数中添加 epoch_prob 初始化**

在 `model/train.py` 的 `train_single` 函数中，找到以下代码（约第 416 行）：

```python
    optimizer.zero_grad()
    pbar = tqdm(range(args.NIter), desc="Training Progress", dynamic_ncols=True)
    step_counter = 0
```

在其后、`for i in pbar:` 之前（约第 417 行位置），插入：

```python
    # epoch-level 共享 y_ran 采样状态
    epoch_prob = None
    epoch_score = None
```

- [ ] **Step 3: 替换 train_single 中非 sobol 分支的 y_ran 生成逻辑**

在 `train_single` 函数中，找到以下代码块（约第 470-498 行）：

```python
            else:
                with torch.no_grad():
                    y_ran = model.generate_structure_aware_y_ran(vel_batch, num_pts=900)

                for batch in dataloader['train_y']:
                    y_batch = batch[0].to(device)
                    y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                    loss, loss_f, loss_u, loss_r, loss_env = model.loss(
                        vel_batch, y_batch, UU0_batch, labels_batch,
                        a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe, freq_batch=freq_batch,
                        y_ran=y_ran
                    )

                    loss = loss / args.accumulation_steps
                    loss.backward()

                    step_counter += 1
                    if step_counter % args.accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    batch_loss.append(loss.item() * args.accumulation_steps)
                    batch_u_loss.append(loss_u.item())
                    batch_f_loss.append(loss_f.item())
                    batch_r_loss.append(loss_r.item() if isinstance(loss_r, torch.Tensor) else loss_r)
                    batch_env_loss.append(loss_env.item())

                    del loss, loss_f, loss_u, loss_r, loss_env, y_batch
```

替换为：

```python
            else:
                # epoch-level 共享 y_ran 采样
                if getattr(args, 'use_epoch_shared_y_ran', False):
                    should_update_prob = (
                        epoch_prob is None
                        or args.y_ran_prob_update_every == 1
                        or (args.y_ran_prob_update_every > 1 and i % args.y_ran_prob_update_every == 0)
                    )
                    if should_update_prob:
                        with torch.no_grad():
                            epoch_prob, epoch_score = build_epoch_velocity_gradient_prob(
                                train_loader=dataloader['train'],
                                device=device,
                                use_max_mix=args.y_ran_use_max_mix,
                                mean_weight=args.y_ran_mean_weight,
                                max_weight=args.y_ran_max_weight,
                            )

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
                else:
                    with torch.no_grad():
                        y_ran = model.generate_structure_aware_y_ran(vel_batch, num_pts=900)

                for batch in dataloader['train_y']:
                    y_batch = batch[0].to(device)
                    y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                    loss, loss_f, loss_u, loss_r, loss_env = model.loss(
                        vel_batch, y_batch, UU0_batch, labels_batch,
                        a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe, freq_batch=freq_batch,
                        y_ran=y_ran
                    )

                    loss = loss / args.accumulation_steps
                    loss.backward()

                    step_counter += 1
                    if step_counter % args.accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    batch_loss.append(loss.item() * args.accumulation_steps)
                    batch_u_loss.append(loss_u.item())
                    batch_f_loss.append(loss_f.item())
                    batch_r_loss.append(loss_r.item() if isinstance(loss_r, torch.Tensor) else loss_r)
                    batch_env_loss.append(loss_env.item())

                    del loss, loss_f, loss_u, loss_r, loss_env, y_batch
```

- [ ] **Step 4: 验证语法正确**

Run: `cd /home/zhangdaoguang/Code/Physical_Informed_DeepONet && python -c "from model.train import train_single; print('train_single loaded OK')"`

Expected: 输出 `train_single loaded OK`

- [ ] **Step 5: Commit**

```bash
git add model/train.py
git commit -m "feat: integrate epoch-level y_ran sampling into train_single"
```

---

### Task 5: 修改 model/train.py 的 _train_stage 函数

**Files:**
- Modify: `model/train.py:15-312` (_train_stage 函数)

- [ ] **Step 1: 在 _train_stage 函数中添加 epoch_prob 初始化**

在 `_train_stage` 函数中，找到以下代码（约第 94 行）：

```python
    optimizer.zero_grad()
    step_counter = 0
    pbar = tqdm(range(stage_niter), desc=f"Stage {stage_idx} [{stage_name}]", dynamic_ncols=True)
```

在其后、`for i in pbar:` 之前，插入：

```python
    # epoch-level 共享 y_ran 采样状态
    epoch_prob = None
    epoch_score = None
```

- [ ] **Step 2: 替换 _train_stage 中非 sobol 分支的 y_ran 生成逻辑**

在 `_train_stage` 函数中，找到以下代码块（约第 149-177 行）：

```python
            else:
                with torch.no_grad():
                    y_ran = model.generate_structure_aware_y_ran(vel_batch, num_pts=900)

                for batch in dataloader['train_y']:
                    y_batch = batch[0].to(device)
                    y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                    loss, loss_f, loss_u, loss_r, loss_env = model.loss(
                        vel_batch, y_batch, UU0_batch, labels_batch,
                        a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe, freq_batch=freq_batch,
                        y_ran=y_ran
                    )

                    loss = loss / args.accumulation_steps
                    loss.backward()

                    step_counter += 1
                    if step_counter % args.accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    batch_loss.append(loss.item() * args.accumulation_steps)
                    batch_u_loss.append(loss_u.item())
                    batch_f_loss.append(loss_f.item())
                    batch_r_loss.append(loss_r.item() if isinstance(loss_r, torch.Tensor) else loss_r)
                    batch_env_loss.append(loss_env.item())

                    del loss, loss_f, loss_u, loss_r, loss_env, y_batch
```

替换为：

```python
            else:
                # epoch-level 共享 y_ran 采样
                if getattr(args, 'use_epoch_shared_y_ran', False):
                    should_update_prob = (
                        epoch_prob is None
                        or args.y_ran_prob_update_every == 1
                        or (args.y_ran_prob_update_every > 1 and i % args.y_ran_prob_update_every == 0)
                    )
                    if should_update_prob:
                        with torch.no_grad():
                            epoch_prob, epoch_score = build_epoch_velocity_gradient_prob(
                                train_loader=dataloader['train'],
                                device=device,
                                use_max_mix=args.y_ran_use_max_mix,
                                mean_weight=args.y_ran_mean_weight,
                                max_weight=args.y_ran_max_weight,
                            )

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
                else:
                    with torch.no_grad():
                        y_ran = model.generate_structure_aware_y_ran(vel_batch, num_pts=900)

                for batch in dataloader['train_y']:
                    y_batch = batch[0].to(device)
                    y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                    loss, loss_f, loss_u, loss_r, loss_env = model.loss(
                        vel_batch, y_batch, UU0_batch, labels_batch,
                        a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe, freq_batch=freq_batch,
                        y_ran=y_ran
                    )

                    loss = loss / args.accumulation_steps
                    loss.backward()

                    step_counter += 1
                    if step_counter % args.accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    batch_loss.append(loss.item() * args.accumulation_steps)
                    batch_u_loss.append(loss_u.item())
                    batch_f_loss.append(loss_f.item())
                    batch_r_loss.append(loss_r.item() if isinstance(loss_r, torch.Tensor) else loss_r)
                    batch_env_loss.append(loss_env.item())

                    del loss, loss_f, loss_u, loss_r, loss_env, y_batch
```

- [ ] **Step 3: 验证语法正确**

Run: `cd /home/zhangdaoguang/Code/Physical_Informed_DeepONet && python -c "from model.train import _train_stage; print('_train_stage loaded OK')"`

Expected: 输出 `_train_stage loaded OK`

- [ ] **Step 4: Commit**

```bash
git add model/train.py
git commit -m "feat: integrate epoch-level y_ran sampling into _train_stage"
```

---

### Task 6: 修改 model/train_distributed.py 的 _train_worker 函数

**Files:**
- Modify: `model/train_distributed.py:21-27` (import 区域)
- Modify: `model/train_distributed.py:140-148` (epoch_prob 初始化)
- Modify: `model/train_distributed.py:194-231` (y_ran 生成逻辑)

- [ ] **Step 1: 更新 import**

在 `model/train_distributed.py` 中，将第 21-27 行的 import 块：

```python
from model.utils import (
    setup_distributed,
    cleanup_distributed,
    wrap_model_for_distributed,
    is_main_process,
    reduce_tensor
)
```

替换为：

```python
from model.utils import (
    setup_distributed,
    cleanup_distributed,
    wrap_model_for_distributed,
    is_main_process,
    reduce_tensor,
    build_epoch_velocity_gradient_prob,
    sample_shared_y_ran_from_epoch_prob,
)
```

- [ ] **Step 2: 添加 epoch_prob 初始化**

在 `_train_worker` 函数中，找到以下代码（约第 141 行）：

```python
    loss_log, loss_pde_log, loss_data_log, loss_reg_log, loss_env_log = [], [], [], [], []
    valid_u_loss, valid_f_loss = [], []
```

在其后插入：

```python
    # epoch-level 共享 y_ran 采样状态
    epoch_prob = None
    epoch_score = None
```

- [ ] **Step 3: 替换 y_ran 生成逻辑**

在 `_train_worker` 函数中，找到以下代码块（约第 194-231 行）：

```python
            # 每个 velocity batch 只生成一次自适应采样点
            with torch.no_grad():
                y_ran = model.module.generate_structure_aware_y_ran(vel_batch, num_pts=900)

            for idx, batch in enumerate(coord_batches):
                y_batch = batch[0].to(device)
                y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                # 拼接数据坐标和 PDE 采样坐标
                y_combined = torch.cat([y_batch, y_ran], dim=1)
                y_combined.requires_grad_(True)

                # 通过 DDP wrapper 调用 forward (触发梯度同步 hook)
                Delta_U = model(vel_batch, y_combined, UU0_batch, freq_batch=freq_batch)

                # 计算损失 (不调用 forward)
                loss, loss_f, loss_u, loss_r, loss_env = model.module.compute_loss(
                    Delta_U, vel_batch, y_batch, UU0_batch, labels_batch,
                    y_combined, a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
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
                batch_env_loss.append(loss_env.item())

                del loss, loss_f, loss_u, loss_r, loss_env, y_batch, Delta_U
```

替换为：

```python
            # 每个 velocity batch 生成一组共享 y_ran
            if getattr(args, 'use_epoch_shared_y_ran', False):
                should_update_prob = (
                    epoch_prob is None
                    or args.y_ran_prob_update_every == 1
                    or (args.y_ran_prob_update_every > 1 and i % args.y_ran_prob_update_every == 0)
                )
                if should_update_prob:
                    with torch.no_grad():
                        epoch_prob, epoch_score = build_epoch_velocity_gradient_prob(
                            train_loader=dataloader['train'],
                            device=device,
                            use_max_mix=args.y_ran_use_max_mix,
                            mean_weight=args.y_ran_mean_weight,
                            max_weight=args.y_ran_max_weight,
                        )

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
            else:
                with torch.no_grad():
                    y_ran = model.module.generate_structure_aware_y_ran(vel_batch, num_pts=900)

            for idx, batch in enumerate(coord_batches):
                y_batch = batch[0].to(device)
                y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                # 拼接数据坐标和 PDE 采样坐标
                y_combined = torch.cat([y_batch, y_ran], dim=1)
                y_combined.requires_grad_(True)

                # 通过 DDP wrapper 调用 forward (触发梯度同步 hook)
                Delta_U = model(vel_batch, y_combined, UU0_batch, freq_batch=freq_batch)

                # 计算损失 (不调用 forward)
                loss, loss_f, loss_u, loss_r, loss_env = model.module.compute_loss(
                    Delta_U, vel_batch, y_batch, UU0_batch, labels_batch,
                    y_combined, a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
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
                batch_env_loss.append(loss_env.item())

                del loss, loss_f, loss_u, loss_r, loss_env, y_batch, Delta_U
```

- [ ] **Step 4: 验证语法正确**

Run: `cd /home/zhangdaoguang/Code/Physical_Informed_DeepONet && python -c "from model.train_distributed import train_distributed; print('train_distributed loaded OK')"`

Expected: 输出 `train_distributed loaded OK`

- [ ] **Step 5: Commit**

```bash
git add model/train_distributed.py
git commit -m "feat: integrate epoch-level y_ran sampling into distributed training"
```

---

### Task 7: 冒烟测试 — 验证新采样函数可正确运行

**Files:**
- No file changes, verification only

- [ ] **Step 1: 编写并运行冒烟测试脚本**

创建临时测试脚本运行后删除：

Run:
```bash
cd /home/zhangdaoguang/Code/Physical_Informed_DeepONet && python -c "
import torch
from config import Args
from model.utils import build_epoch_velocity_gradient_prob, sample_shared_y_ran_from_epoch_prob

args = Args()
device = torch.device('cpu')

# 模拟一个包含 3 个 batch 的 train_loader
# 每个 batch: (vel, UU0, labels)，vel shape = [B, 1, Z, X]
dummy_dataset = torch.utils.data.TensorDataset(
    torch.randn(9, 1, args.nz, args.absmax_z),
    torch.randn(9, 1, args.nz, args.nx),
    torch.randn(9, 2, args.nz, args.nx),
)
dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=3)

print('--- Test 1: build_epoch_velocity_gradient_prob ---')
prob, score = build_epoch_velocity_gradient_prob(dummy_loader, device)
print(f'prob shape: {prob.shape}')
print(f'prob sum: {prob.sum().item():.6f}')
print(f'score shape: {score.shape}')
assert prob.shape == (args.nz * args.nx,), f'Expected {(args.nz * args.nx,)}, got {prob.shape}'
assert abs(prob.sum().item() - 1.0) < 1e-5, f'prob sum should be ~1.0, got {prob.sum().item()}'

print('--- Test 2: sample_shared_y_ran_from_epoch_prob ---')
y_shared = sample_shared_y_ran_from_epoch_prob(
    prob=prob,
    args=args,
    num_pts=900,
    structure_ratio=0.60,
    surface_ratio=0.20,
    uniform_ratio=0.20,
    surface_depth_grids=5,
)
print(f'y_shared shape: {y_shared.shape}')
assert y_shared.shape == (900, 2), f'Expected (900, 2), got {y_shared.shape}'
z_max = y_shared[:, 0].max().item()
x_max = y_shared[:, 1].max().item()
print(f'z range: [0, {z_max:.1f}], expected max: {args.nz * args.dh}')
print(f'x range: [0, {x_max:.1f}], expected max: {args.nx * args.dh}')
assert z_max <= args.nz * args.dh + 1, 'z exceeds grid'
assert x_max <= args.nx * args.dh + 1, 'x exceeds grid'

print()
print('ALL TESTS PASSED')
"
```

Expected: 输出 `ALL TESTS PASSED`

注意：此测试使用随机数据验证函数逻辑正确性，不验证训练效果。

- [ ] **Step 2: 修正冒烟测试中的维度问题（如有）并 re-run**

如果 Step 1 因维度不匹配报错，检查 `args.nz` 和 `args.nx` 的实际值与 dummy 数据是否匹配，调整后重试。

- [ ] **Step 3: Final commit (如果需要修正)**

```bash
git add -A
git commit -m "fix: address smoke test issues in epoch-level y_ran sampling"
```
