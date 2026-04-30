# DDP Staged Curriculum Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add curriculum learning (staged training) support to the DDP multi-GPU training path, so that 2x RTX 4090 can run low→mid→high frequency progressive training in parallel.

**Architecture:** Add a new `_train_worker_staged` function in `train_distributed.py` that wraps the existing DDP training loop inside a stage loop (borrowed from `_train_stage` in `train.py`). Route to it from `main2.py` when both `use_parallel=True` and `staged_training=True`.

**Tech Stack:** PyTorch DDP (mp.spawn), DistributedSampler, existing model/dataloader infrastructure.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `main2.py` | Modify | Add routing branch for staged DDP |
| `model/train_distributed.py` | Modify | Add `_train_worker_staged` + `train_distributed_staged` |
| `model/train.py` | No change | Single-GPU staged training stays untouched |
| `model/PI_DeepOnet.py` | No change | Model code untouched |
| `model/dataloader.py` | No change | Data loading untouched |
| `model/utils.py` | No change | Utility functions untouched |

---

### Task 1: Add routing in main2.py

**Files:**
- Modify: `main2.py:24-49`

- [ ] **Step 1: Update the routing logic in main2.py**

In `main2.py`, replace lines 24-49 with logic that checks `staged_training` when `use_parallel=True`:

```python
    # ==========================================
    # 检查训练模式
    # ==========================================
    use_parallel = getattr(args, 'use_parallel', False)
    use_staged = getattr(args, 'staged_training', False)

    if use_parallel:
        # 多 GPU 并行模式
        print("=" * 60)
        print("多 GPU 并行训练模式")
        print("=" * 60)

        # 检测可用 GPU
        num_gpus = getattr(args, 'num_gpus', 2)
        min_gpu_memory = getattr(args, 'min_gpu_memory', 10240)  # MB

        available_gpus = get_available_gpus(min_memory_mb=min_gpu_memory, require_count=num_gpus)

        if len(available_gpus) < num_gpus:
            print(f"⚠️ 可用 GPU 不足 ({len(available_gpus)} < {num_gpus})，回退到单 GPU 模式")
            args.use_parallel = False
            args.device = available_gpus[0] if available_gpus else 0
        else:
            print(f"✅ 检测到 {len(available_gpus)} 个可用 GPU: {available_gpus[:num_gpus]}")
            print("=" * 60)

            if use_staged:
                from model.train_distributed import train_distributed_staged
                train_distributed_staged(args)
            else:
                from model.train_distributed import train_distributed
                train_distributed(args)
            return
```

Lines 51 onward (single GPU path) remain unchanged.

- [ ] **Step 2: Commit**

```bash
git add main2.py
git commit -m "feat: route to staged DDP when use_parallel and staged_training are both True"
```

---

### Task 2: Add `train_distributed_staged` entry function

**Files:**
- Modify: `model/train_distributed.py`

- [ ] **Step 1: Add the entry function at the end of `model/train_distributed.py`**

Append after the existing `train_distributed` function (after line 438):

```python
def train_distributed_staged(args):
    """
    单机多卡分布式 + 课程学习训练入口函数

    使用 mp.spawn 启动多进程，进程生命周期覆盖所有阶段。

    Args:
        args: 配置参数对象 (config.Args)
    """
    world_size = getattr(args, 'num_gpus', 1)

    print("=" * 60)
    print(f"启动单机多卡分布式课程学习训练")
    print(f"GPU 数量: {world_size}")
    stages = getattr(args, 'stages', [])
    print(f"训练阶段数: {len(stages)}")
    for si, s in enumerate(stages):
        print(f'  Stage {si}: {s["name"]} | freq [{s["freq_min"]}-{s["freq_max"]}] Hz | '
              f'{s.get("NIter", "?")} epochs | lr={s.get("lr", "?")}')
    print("=" * 60)

    # 使用 mp.spawn 启动多进程
    mp.spawn(
        _train_worker_staged,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )
```

- [ ] **Step 2: Commit**

```bash
git add model/train_distributed.py
git commit -m "feat: add train_distributed_staged entry function"
```

---

### Task 3: Add `_train_worker_staged` — model init + stage loop skeleton

**Files:**
- Modify: `model/train_distributed.py`

This is the core function. It will be added before `train_distributed_staged`. The implementation is split across Tasks 3-6.

- [ ] **Step 1: Add the function skeleton with model initialization and stage loop**

Insert before `train_distributed_staged`:

```python
def _train_worker_staged(rank, world_size, args):
    """
    分布式课程学习训练工作进程。

    在单个 mp.spawn 生命周期内依次执行所有阶段。
    每个阶段: 加载数据 → replay 合并 → DDP 训练 → 保存权重。
    """
    # 初始化分布式环境
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    args.device = rank

    # ==========================================
    # 创建模型（全局一次，跨阶段复用）
    # ==========================================
    model = Pi_DeepONet(args).to(device)

    if is_main_process(rank):
        model._init_weights()
        print(f"[Stage DDP] PI_DeepONet 模型总参数数量: {count_parameters(model)}")

    # 广播所有参数从 rank 0 到其他 rank
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # DDP wrap
    model = wrap_model_for_distributed(model, rank)

    # FNO（用于可选的软标签）
    fno = FNO(args).to(device)
    if args.use_fno_as_label and args.fno_weights_path:
        fno.load_state_dict(torch.load(args.fno_weights_path, map_location=device)['model_state_dict'])
        if is_main_process(rank):
            print(f"已加载 FNO 权重: {args.fno_weights_path}")
    fno.eval()

    save_doc = args.save_doc
    if is_main_process(rank):
        os.makedirs(save_doc, exist_ok=True)

    # 保存基础文件名模板
    base_vel_filename = args.vel_filename
    base_bg_filename = args.backgroundfield_filename
    base_wf_filename = args.wavefield_filename
    base_freq_filename = args.freq_filename

    # ==========================================
    # 阶段循环
    # ==========================================
    stages = args.stages

    for stage_idx, stage_config in enumerate(stages):
        if is_main_process(rank):
            print(f'\n{"=" * 60}')
            print(f'>>> [DDP] 开始 Stage {stage_idx}: {stage_config["name"]} '
                  f'[{stage_config["freq_min"]}-{stage_config["freq_max"]} Hz]')
            print(f'{"=" * 60}')

        model, save_doc = _train_stage_ddp(
            args, model, fno, device, rank, world_size,
            stage_idx, stage_config, save_doc,
            base_vel_filename, base_bg_filename, base_wf_filename, base_freq_filename,
        )

    if is_main_process(rank):
        print(f'\n{"=" * 60}')
        print(f'全部 {len(stages)} 个阶段训练完毕！')
        print(f'{"=" * 60}')

    cleanup_distributed()
```

- [ ] **Step 2: Commit**

```bash
git add model/train_distributed.py
git commit -m "feat: add _train_worker_staged skeleton with model init and stage loop"
```

---

### Task 4: Add `_train_stage_ddp` — data loading + replay + optimizer

**Files:**
- Modify: `model/train_distributed.py`

- [ ] **Step 1: Add `_train_stage_ddp` function with data loading, replay merge, and optimizer init**

Insert before `_train_worker_staged`. This function handles a single stage's setup and training:

```python
def _train_stage_ddp(args, model, fno, device, rank, world_size,
                     stage_idx, stage_config, save_doc,
                     base_vel_filename, base_bg_filename, base_wf_filename, base_freq_filename):
    """
    DDP 单阶段训练函数。

    Returns:
        (model, save_doc): 更新后的模型和保存路径
    """
    stage_name = stage_config['name']
    freq_range = stage_config['freq_range']
    stage_niter = stage_config.get('NIter', args.NIter)
    stage_lr = stage_config.get('lr', args.lr)
    stage_warmup = stage_config.get('warmup_epochs', args.warmup_epochs)
    a = stage_config.get('a', args.a)
    b = stage_config.get('b', args.b)
    c = stage_config.get('c', args.c)
    d = getattr(args, 'd', 0.1)

    # ---- 1. 替换文件名 ----
    base_freq_tag = 'freq3to20'
    stage_freq_tag = f'freq{freq_range}'

    args.vel_filename = base_vel_filename.replace(base_freq_tag, stage_freq_tag)
    args.backgroundfield_filename = base_bg_filename.replace(base_freq_tag, stage_freq_tag)
    args.wavefield_filename = base_wf_filename.replace(base_freq_tag, stage_freq_tag)
    args.freq_filename = base_freq_filename

    original_load_path = args.load_path
    if 'data_dir' in stage_config:
        args.load_path = stage_config['data_dir']
        args.vel_filename = os.path.basename(args.vel_filename)
        args.backgroundfield_filename = os.path.basename(args.backgroundfield_filename)
        args.wavefield_filename = os.path.basename(args.wavefield_filename)
        args.freq_filename = os.path.basename(args.freq_filename)
    current_stage_load_path = args.load_path

    if is_main_process(rank):
        print(f'\n[*] Stage {stage_idx} [{stage_name}] 数据文件:')
        print(f'    load_path: {args.load_path}')
        print(f'    vel:   {args.vel_filename}')
        print(f'    bg:    {args.backgroundfield_filename}')
        print(f'    wf:    {args.wavefield_filename}')
        print(f'    freq:  {args.freq_filename}')

    # ---- 2. 加载上一阶段权重 ----
    if stage_idx > 0:
        prev_path = os.path.join(save_doc, f'{args.filename}_stage{stage_idx - 1}_final_weights_{args.nz}.pth')
        if is_main_process(rank):
            if os.path.exists(prev_path):
                print(f'[*] 加载上一阶段权重: {prev_path}')
                ckpt = torch.load(prev_path, map_location=device)
                model.module.load_state_dict(ckpt['model_state_dict'])
            else:
                print(f'⚠️ 未找到上一阶段权重: {prev_path}，将使用当前模型权重继续')
        # 广播参数确保所有 rank 一致
        for param in model.module.parameters():
            dist.broadcast(param.data, src=0)
    else:
        if is_main_process(rank):
            print(f'[*] Stage 0: 权重已在 _train_worker_staged 中初始化')

    # ---- 3. 加载数据 ----
    dataloader, plot_data = prepare_training_dataloaders(args, device)

    # ---- 3.5 Replay ----
    replay_stages_list = stage_config.get('replay_stages', [])
    if replay_stages_list and stage_idx > 0:
        cur_vel_fn = args.vel_filename
        cur_bg_fn = args.backgroundfield_filename
        cur_wf_fn = args.wavefield_filename
        cur_freq_fn = args.freq_filename

        train_ds = dataloader['train'].dataset
        train_tensors = train_ds.tensors
        has_freq_replay = len(train_tensors) >= 4
        if has_freq_replay:
            combined_vel = train_tensors[0]
            combined_UU0 = train_tensors[1]
            combined_labels = train_tensors[2]
            combined_freq = train_tensors[3]
        else:
            combined_vel = train_tensors[0]
            combined_UU0 = train_tensors[1]
            combined_labels = train_tensors[2]
            combined_freq = None

        for replay_idx in replay_stages_list:
            replay_config = args.stages[replay_idx]
            replay_freq_tag = f'freq{replay_config["freq_range"]}'

            args.vel_filename = base_vel_filename.replace(base_freq_tag, replay_freq_tag)
            args.backgroundfield_filename = base_bg_filename.replace(base_freq_tag, replay_freq_tag)
            args.wavefield_filename = base_wf_filename.replace(base_freq_tag, replay_freq_tag)
            args.freq_filename = base_freq_filename

            if 'data_dir' in replay_config:
                args.load_path = replay_config['data_dir']
                args.vel_filename = os.path.basename(args.vel_filename)
                args.backgroundfield_filename = os.path.basename(args.backgroundfield_filename)
                args.wavefield_filename = os.path.basename(args.wavefield_filename)
                args.freq_filename = os.path.basename(args.freq_filename)

            if is_main_process(rank):
                print(f'    [Replay] 加载 Stage {replay_idx} [{replay_config["name"]}] 数据: {args.load_path}/{args.vel_filename}')

            replay_dl, _ = prepare_training_dataloaders(args, device)
            replay_ds = replay_dl['train'].dataset
            replay_tensors = replay_ds.tensors

            replay_ratio = replay_config.get('replay_ratio', 1.0)
            n_replay = replay_tensors[0].shape[0]
            if replay_ratio < 1.0:
                # 固定 seed 确保所有 rank subsample 相同
                torch.manual_seed(42 + replay_idx)
                n_sample = max(1, int(n_replay * replay_ratio))
                perm = torch.randperm(n_replay)[:n_sample]
                replay_tensors = tuple(t[perm] for t in replay_tensors)
                if is_main_process(rank):
                    print(f'    [Replay] Stage {replay_idx}: {n_sample}/{n_replay} 样本 (ratio={replay_ratio})')

            combined_vel = torch.cat([combined_vel, replay_tensors[0]], dim=0)
            combined_UU0 = torch.cat([combined_UU0, replay_tensors[1]], dim=0)
            combined_labels = torch.cat([combined_labels, replay_tensors[2]], dim=0)
            if has_freq_replay and len(replay_tensors) >= 4:
                combined_freq = torch.cat([combined_freq, replay_tensors[3]], dim=0)

        # 恢复文件名
        args.vel_filename = cur_vel_fn
        args.backgroundfield_filename = cur_bg_fn
        args.wavefield_filename = cur_wf_fn
        args.freq_filename = cur_freq_fn
        args.load_path = current_stage_load_path

        # 重建训练 DataLoader（用 DistributedSampler）
        pin_mem = device.type == 'cuda'

        if has_freq_replay:
            new_train_ds = TensorDataset(combined_vel, combined_UU0, combined_labels, combined_freq)
        else:
            new_train_ds = TensorDataset(combined_vel, combined_UU0, combined_labels)

        replay_sampler = DistributedSampler(
            new_train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        dataloader['train'] = DataLoader(
            new_train_ds,
            batch_size=args.batch_size_v, sampler=replay_sampler, drop_last=True,
            pin_memory=pin_mem, num_workers=4, prefetch_factor=2,
        )

        if is_main_process(rank):
            print(f'    [Replay] 训练集合并完成: {combined_vel.shape[0]} 样本 (含 replay)')

    else:
        # 无 replay：用 DistributedSampler 包裹原始 DataLoader
        train_sampler = DistributedSampler(
            dataloader['train'].dataset,
            num_replicas=world_size, rank=rank, shuffle=True
        )
        dataloader['train'] = DataLoader(
            dataloader['train'].dataset,
            batch_size=args.batch_size_v,
            sampler=train_sampler,
            num_workers=4, pin_memory=True
        )

    has_freq = len(dataloader['train'].dataset.tensors) >= 4

    if is_main_process(rank):
        print(f"已启用 DistributedSampler (world_size={world_size})")

    # ---- 4. 初始化优化器与调度器 ----
    optimizer = torch.optim.Adam(
        model.parameters(), lr=stage_lr * world_size, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.factor, patience=args.patience,
        min_lr=args.min_lr * world_size
    )
    warmup_scheduler = WarmupScheduler(
        optimizer, warmup_epochs=stage_warmup, base_lr=stage_lr * world_size,
        warmup_start_lr=stage_lr * world_size / 10., warmup_strategy="linear"
    )

    # ---- 5. 运行 DDP 训练循环 ----
    model = _run_stage_training_loop(
        args, model, fno, device, rank, world_size,
        dataloader, plot_data, has_freq,
        optimizer, scheduler, warmup_scheduler,
        stage_idx, stage_name, stage_niter, stage_warmup,
        a, b, c, d, save_doc,
    )

    return model, save_doc
```

- [ ] **Step 2: Commit**

```bash
git add model/train_distributed.py
git commit -m "feat: add _train_stage_ddp with data loading, replay merge, and optimizer init"
```

---

### Task 5: Add `_run_stage_training_loop` — the DDP training loop

**Files:**
- Modify: `model/train_distributed.py`

- [ ] **Step 1: Add the DDP training loop function**

Insert before `_train_stage_ddp`. This is adapted from the existing `_train_worker` training loop (lines 158-400 of the current file) with stage-aware logging and checkpoint naming:

```python
def _run_stage_training_loop(args, model, fno, device, rank, world_size,
                             dataloader, plot_data, has_freq,
                             optimizer, scheduler, warmup_scheduler,
                             stage_idx, stage_name, stage_niter, stage_warmup,
                             a, b, c, d, save_doc):
    """
    单阶段 DDP 训练循环。
    复用现有 _train_worker 的 DDP 训练逻辑，增加 stage 后缀的日志和保存。
    """
    # 训练状态
    loss_log, loss_pde_log, loss_data_log, loss_reg_log, loss_env_log = [], [], [], [], []
    valid_u_loss, valid_f_loss = [], []

    # epoch-level 共享 y_ran
    epoch_prob = None
    epoch_score = None

    first_flag = True
    pde_norm_coe = 1.
    data_norm_coe = 1.
    env_norm_coe = 1.

    optimizer.zero_grad()

    if is_main_process(rank):
        pbar = tqdm(range(stage_niter), desc=f"Stage {stage_idx} [{stage_name}]", dynamic_ncols=True)
    else:
        pbar = range(stage_niter)

    for i in pbar:
        # 动态权重调整
        if args.if_adjust and i > args.adjust_from and (i - args.adjust_from) % args.adjust_every == 0:
            decay_times = i // args.adjust_every
            a = max(a * (args.adjust_speed ** (-decay_times)), 2e-1)
            b, c = 1, 0

        model.train()
        batch_loss, batch_u_loss, batch_f_loss, batch_r_loss, batch_env_loss = [], [], [], [], []

        # 设置 epoch 以确保每个 epoch shuffle 不同
        dataloader['train'].sampler.set_epoch(i)

        # 预收集坐标 batches
        coord_batches = list(dataloader['train_y'])
        n_coord = len(coord_batches)

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

            # epoch-level 共享 y_ran
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
                        prob=epoch_prob, args=args,
                        num_pts=args.y_ran_num_pts,
                        structure_ratio=args.y_ran_structure_ratio,
                        surface_ratio=args.y_ran_surface_ratio,
                        uniform_ratio=args.y_ran_uniform_ratio,
                        source_ratio=args.y_ran_source_ratio,
                        surface_depth_grids=args.y_ran_surface_depth_grids,
                    )
                y_ran = y_shared.unsqueeze(0).expand(vel_batch.shape[0], -1, -1).clone().requires_grad_(True)
            else:
                with torch.no_grad():
                    y_ran = model.module.generate_structure_aware_y_ran(vel_batch, num_pts=900)

            for idx, batch in enumerate(coord_batches):
                y_batch = batch[0].to(device)
                y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                y_combined = torch.cat([y_batch, y_ran], dim=1)
                y_combined.requires_grad_(True)

                # DDP forward
                Delta_U = model(vel_batch, y_combined, UU0_batch, freq_batch=freq_batch)

                # Loss (不触发 DDP forward)
                loss, loss_f, loss_u, loss_r, loss_env = model.module.compute_loss(
                    Delta_U, vel_batch, y_batch, UU0_batch, labels_batch,
                    y_combined, a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
                    freq_batch=freq_batch
                )

                loss = loss / n_coord

                # 梯度同步：仅最后一个 coord batch 触发 all-reduce
                if idx < n_coord - 1:
                    with model.no_sync():
                        loss.backward()
                else:
                    loss.backward()

                batch_loss.append(loss.item() * n_coord)
                batch_u_loss.append(loss_u.item())
                batch_f_loss.append(loss_f.item())
                batch_r_loss.append(loss_r.item() if isinstance(loss_r, torch.Tensor) else loss_r)
                batch_env_loss.append(loss_env.item())

                del loss, loss_f, loss_u, loss_r, loss_env, y_batch, Delta_U

            # 每个 velocity batch 结束后更新参数
            optimizer.step()
            optimizer.zero_grad()

        # ---- 跨进程平均损失 ----
        avg_loss = np.mean(batch_loss) if batch_loss else 0
        avg_u_loss = np.mean(batch_u_loss) if batch_u_loss else 0
        avg_f_loss = np.mean(batch_f_loss) if batch_f_loss else 0
        avg_env_loss = np.mean(batch_env_loss) if batch_env_loss else 0

        loss_tensor = torch.tensor([avg_loss, avg_u_loss, avg_f_loss], device=device)
        loss_tensor = reduce_tensor(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss, avg_u_loss, avg_f_loss = loss_tensor.cpu().numpy()

        if first_flag:
            data_norm_coe = avg_u_loss if avg_u_loss > 0 else 1.0
            pde_norm_coe = avg_f_loss if avg_f_loss > 0 else 1.0
            env_norm_coe = avg_env_loss if avg_env_loss > 0 else 1.0
            loss_log.append(a + b)
            loss_data_log.append(1.)
            loss_pde_log.append(1.)
            loss_env_log.append(1.)
            first_flag = False
        else:
            loss_log.append(avg_loss)
            loss_data_log.append(avg_u_loss)
            loss_pde_log.append(avg_f_loss)
            loss_env_log.append(avg_env_loss)
            loss_reg_log.append(np.mean(batch_r_loss) / (args.batch_size * args.batch_size_v) if batch_r_loss else 0)

        # 更新进度条
        if is_main_process(rank):
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Total': f"{avg_loss:.4e}",
                'PDE': f"{loss_pde_log[-1]:.4e}",
                'Data': f"{loss_data_log[-1]:.4e}",
                'Env': f"{loss_env_log[-1]:.4e}",
                'LR': f"{current_lr:.2e}",
                'GPU': f"{world_size}"
            })

        # 学习率调度
        if i <= stage_warmup:
            warmup_scheduler.step(i)
        else:
            scheduler.step(avg_loss)

        # ---- 验证 (仅主进程) ----
        if i % args.validate_every == 0 and is_main_process(rank):
            model.eval()
            vb_u_loss, vb_f_loss = [], []

            for batch_data in dataloader['valid']:
                if has_freq:
                    vel_batch, UU0_batch, labels_batch, freq_batch = batch_data
                    freq_batch = freq_batch.to(device)
                else:
                    vel_batch, UU0_batch, labels_batch = batch_data
                    freq_batch = None
                vel_batch = vel_batch.to(device)
                UU0_batch = UU0_batch.to(device)
                labels_batch = labels_batch.to(device)

                for batch in dataloader['valid_y']:
                    y_batch = batch[0].to(device)
                    y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                    _, loss_f_valid, loss_u_valid, _, _ = model.module.loss(
                        vel_batch, y_batch, UU0_batch, labels_batch,
                        a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
                        freq_batch=freq_batch
                    )
                    vb_u_loss.append(loss_u_valid.item())
                    vb_f_loss.append(loss_f_valid.item())

            valid_u_loss.append(np.mean(vb_u_loss) if vb_u_loss else 0.0)
            valid_f_loss.append(np.mean(vb_f_loss) if vb_f_loss else 1.0)

        # ---- 可视化 (仅主进程) ----
        if i % args.save_fig_every == 0 and is_main_process(rank):
            vel_pred = plot_data["vel_pred"]
            UU0_pred = plot_data["UU0_pred"]
            labels_pred = plot_data["labels_pred"]
            vel_test = plot_data["vel_test"]
            UU0_test = plot_data["UU0_test"]
            labels_test = plot_data["labels_test"]
            freq_pred = plot_data["freq_valid"][0:1] if has_freq else None
            freq_test = plot_data["freq_train"][0:1] if has_freq else None

            plot_loss(i, save_doc, loss_log, loss_data_log, loss_pde_log, valid_u_loss, valid_f_loss,
                      suffix=f'_stage{stage_idx}')

            test_plot(args, model.module, fno, i, dataloader["pred"],
                      vel_pred, UU0_pred, labels_pred, f'valid_stage{stage_idx}',
                      if_fine_tune=False, freq=freq_pred)
            test_plot(args, model.module, fno, i, dataloader["test"],
                      vel_test, UU0_test, labels_test, f'train_stage{stage_idx}',
                      if_fine_tune=False, freq=freq_test)
            plot_sinlge(model.module, args, 6, vel_test, UU0_test, labels_test, freq=freq_test)

        # ---- 模型保存 (仅主进程) ----
        if i % args.save_model_every == 0 and is_main_process(rank):
            if isinstance(pbar, tqdm):
                pbar.write(f'>>> Stage {stage_idx} Epoch {i} | Total Loss {loss_log[-1]:.4e} | PDE Loss {loss_pde_log[-1]:.4e}')

            checkpoint = {
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'stage': stage_idx,
                'epoch_in_stage': i,
            }
            torch.save(checkpoint, os.path.join(
                save_doc, f'{args.filename}_stage{stage_idx}_{i}epoch_weights_{args.nz}.pth'))
            np.save(os.path.join(save_doc, f'loss_log_stage{stage_idx}.npy'), loss_log)
            np.save(os.path.join(save_doc, f'loss_data_log_stage{stage_idx}.npy'), loss_data_log)
            np.save(os.path.join(save_doc, f'loss_pde_log_stage{stage_idx}.npy'), loss_pde_log)
            np.save(os.path.join(save_doc, f'loss_env_log_stage{stage_idx}.npy'), loss_env_log)

    # ---- 阶段结束：保存最终权重 ----
    if is_main_process(rank):
        final_path = os.path.join(save_doc, f'{args.filename}_stage{stage_idx}_final_weights_{args.nz}.pth')
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'stage': stage_idx,
        }, final_path)
        print(f'✅ Stage {stage_idx} [{stage_name}] 完成，权重已保存: {final_path}')

        np.save(os.path.join(save_doc, f'loss_log_stage{stage_idx}.npy'), loss_log)
        np.save(os.path.join(save_doc, f'loss_data_log_stage{stage_idx}.npy'), loss_data_log)
        np.save(os.path.join(save_doc, f'loss_pde_log_stage{stage_idx}.npy'), loss_pde_log)
        np.save(os.path.join(save_doc, f'loss_env_log_stage{stage_idx}.npy'), loss_env_log)

    return model
```

- [ ] **Step 2: Commit**

```bash
git add model/train_distributed.py
git commit -m "feat: add _run_stage_training_loop with full DDP training logic"
```

---

### Task 6: Verify the complete implementation

- [ ] **Step 1: Syntax check — import the module**

```bash
cd /home/zhangdaoguang/Code/Physical_Informed_DeepONet
python -c "from model.train_distributed import train_distributed_staged; print('Import OK')"
```

Expected: `Import OK` with no errors.

- [ ] **Step 2: Verify routing with a dry-run config**

Temporarily set `use_parallel=True` and `staged_training=True` in `config.py`, then run:

```bash
cd /home/zhangdaoguang/Code/Physical_Informed_DeepONet
python -c "
from config import Args
args = Args()
print(f'use_parallel={args.use_parallel}, staged_training={args.staged_training}')
print(f'stages count={len(args.stages)}')
print(f'num_gpus={args.num_gpus}')
"
```

Expected: `use_parallel=True, staged_training=True, stages count=3, num_gpus=2`

- [ ] **Step 3: Commit final state**

```bash
git add -A
git commit -m "feat: complete DDP staged curriculum training support"
```

---

## Self-Review

1. **Spec coverage**: All spec sections covered — routing (Task 1), model init (Task 3), data loading + replay (Task 4), DDP training loop (Task 5), checkpoint save (Task 5), weight sync (Task 4 stage loading).

2. **Placeholder scan**: No TBD/TODO found. All code blocks contain complete implementations.

3. **Type consistency**: All function signatures match between callers and callees. `_train_worker_staged` calls `_train_stage_ddp` which calls `_run_stage_training_loop`. Return types consistent (`model` is always DDP-wrapped, accessed via `model.module`).
