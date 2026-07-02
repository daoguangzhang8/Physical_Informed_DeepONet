import os
import numpy as np
import torch
from tqdm import tqdm

from Labconfig import *
from model.utils import *
from model.utils import build_epoch_velocity_gradient_prob, sample_shared_y_ran_from_epoch_prob
from model.dataloader import *
from model.PI_DeepOnet import Pi_DeepONet
from model.FNO import FNO
from model.plotting import *


def _init_sobol_sampling(args, device, rank=0):
    """Initialize persistent train Sobol sequence and fixed validation points."""
    seed = int(getattr(args, 'sobol_seed', 0))
    train_engine = make_sobol_engine(seed + rank)
    valid_engine = make_sobol_engine(seed + 100000)
    points_per_step = int(getattr(
        args, 'sobol_points_per_step',
        getattr(args, 'sobol_points_per_epoch', 800),
    ))
    steps_per_velocity_batch = int(getattr(args, 'sobol_steps_per_velocity_batch', 1))
    valid_points = draw_sobol_grid_points(
        valid_engine, getattr(args, 'valid_sobol_points', 800),
        args.nz, args.nx, args.dh, device,
    )
    print(
        f"Sobol 模式: 每次更新 {points_per_step} 点, "
        f"每个 velocity batch 更新 {steps_per_velocity_batch} 次, "
        f"固定验证 {len(valid_points)} 点"
    )
    return train_engine, valid_points, points_per_step, steps_per_velocity_batch


def _draw_sobol_batch(engine, num_pts, args, device, batch_size):
    points = draw_sobol_grid_points(engine, num_pts, args.nz, args.nx, args.dh, device)
    return points.unsqueeze(0).expand(batch_size, -1, -1).clone().requires_grad_(True)


def _maybe_add_lpips_loss(args, model, loss, loss_env,
                          vel_batch, UU0_batch, labels_batch, freq_batch, epoch):
    """Optionally add LPIPS perceptual loss on a separate image grid."""
    if not getattr(args, 'use_lpips_loss', False):
        return loss, loss_env
    if epoch < int(getattr(args, 'lpips_start_epoch', 0)):
        return loss, loss_env
    interval = max(1, int(getattr(args, 'lpips_interval', 1)))
    if epoch % interval != 0:
        return loss, loss_env

    loss_lpips = model.lpips_loss_on_grid(
        vel_batch, UU0_batch, labels_batch, freq_batch=freq_batch
    )
    loss = loss + float(getattr(args, 'lpips_weight', 0.01)) * loss_lpips
    return loss, loss_lpips


def evaluate_valid_loader(model, loader, coord_loader, device, has_freq,
                          a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
                          use_sobol=False, valid_sobol_points=None):
    batch_u_loss, batch_f_loss = [], []
    for batch_data in loader:
        if has_freq:
            vel_batch, UU0_batch, labels_batch, freq_batch = batch_data
            freq_batch = freq_batch.to(device)
        else:
            vel_batch, UU0_batch, labels_batch = batch_data
            freq_batch = None
        vel_batch = vel_batch.to(device)
        UU0_batch = UU0_batch.to(device)
        labels_batch = labels_batch.to(device)

        if use_sobol:
            y_valid = valid_sobol_points.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)
            _, loss_f_valid, loss_u_valid, _, _ = model.loss(
                vel_batch, y_valid, UU0_batch, labels_batch,
                a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
                freq_batch=freq_batch,
                apply_frequency_data_weight=False,
            )
            batch_u_loss.append(loss_u_valid.item())
            batch_f_loss.append(loss_f_valid.item())
            del loss_f_valid, loss_u_valid, y_valid
        else:
            for batch in coord_loader:
                y_batch = batch[0].to(device)
                y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)
                _, loss_f_valid, loss_u_valid, _, _ = model.loss(
                    vel_batch, y_batch, UU0_batch, labels_batch,
                    a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
                    freq_batch=freq_batch,
                    apply_frequency_data_weight=False,
                )
                batch_u_loss.append(loss_u_valid.item())
                batch_f_loss.append(loss_f_valid.item())

    return (
        np.mean(batch_u_loss) if batch_u_loss else 0.0,
        np.mean(batch_f_loss) if batch_f_loss else 1.0,
    )


def _train_stage(args, model, fno, device, stage_idx, stage_config,
                 base_vel_filename, base_bg_filename, base_wf_filename, base_freq_filename):
    """
    单阶段训练函数
    - stage_idx: 当前阶段编号 (0, 1, 2)
    - stage_config: dict, 包含 name/freq_range/NIter/lr 等
    - base_*_filename: 原始文件名模板，用于替换 freq 标签
    """
    stage_name = stage_config['name']
    freq_range = stage_config['freq_range']
    stage_niter = stage_config.get('NIter', args.NIter)
    stage_lr = stage_config.get('lr', args.lr)
    stage_warmup = stage_config.get('warmup_epochs', args.warmup_epochs)
    a = stage_config.get('a', args.a)
    b = stage_config.get('b', args.b)
    pde_target_weight = b
    c = stage_config.get('c', args.c)
    d = getattr(args, 'd', 0.1)

    # ---- 1. 根据阶段替换数据文件名 ----
    # vel/bg/wf 文件名含 'freq3to20'，直接替换为 'freq{range}'
    # freq 文件名含 'freq_used'，替换为 'freq{range}_used'
    base_freq_tag = 'freq3to20'
    stage_freq_tag = f'freq{freq_range}'

    args.vel_filename = base_vel_filename.replace(base_freq_tag, stage_freq_tag)
    args.backgroundfield_filename = base_bg_filename.replace(base_freq_tag, stage_freq_tag)
    args.wavefield_filename = base_wf_filename.replace(base_freq_tag, stage_freq_tag)
    args.freq_filename = base_freq_filename  # freq 文件不含 stage 标签，保持原文件名

    # 如果 stage 配置了 data_dir，覆盖 load_path（课程学习独立数据路径）
    original_load_path = args.load_path
    if 'data_dir' in stage_config:
        args.load_path = stage_config['data_dir']
        # data_dir 直接指向数据文件所在目录，文件名不需要子目录前缀
        args.vel_filename = os.path.basename(args.vel_filename)
        args.backgroundfield_filename = os.path.basename(args.backgroundfield_filename)
        args.wavefield_filename = os.path.basename(args.wavefield_filename)
        args.freq_filename = os.path.basename(args.freq_filename)
    current_stage_load_path = args.load_path

    print(f'\n[*] Stage {stage_idx} [{stage_name}] 数据文件:')
    print(f'    load_path: {args.load_path}')
    print(f'    vel:   {args.vel_filename}')
    print(f'    bg:    {args.backgroundfield_filename}')
    print(f'    wf:    {args.wavefield_filename}')
    print(f'    freq:  {args.freq_filename}')

    # ---- 2. 课程学习：后续阶段加载前一阶段权重 ----
    save_doc = args.save_doc
    if stage_idx > 0:
        prev_path = os.path.join(save_doc, f'{args.filename}_stage{stage_idx - 1}_final_weights_{args.nz}.pth')
        if os.path.exists(prev_path):
            print(f'[*] 加载上一阶段权重: {prev_path}')
            ckpt = torch.load(prev_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            print(f'⚠️ 未找到上一阶段权重: {prev_path}，将使用当前模型权重继续')
    else:
        model._init_weights()
        print(f'[*] Stage 0: 从头初始化模型权重')

    # ---- 3. 加载数据 ----
    dataloader, plot_data = prepare_training_dataloaders(args, device)

    # ---- 3.5 Replay 前序阶段数据（防遗忘） ----
    replay_stages_list = stage_config.get('replay_stages', [])
    if replay_stages_list and stage_idx > 0:
        # 保存当前阶段的文件名（replay 后需恢复）
        cur_vel_fn = args.vel_filename
        cur_bg_fn = args.backgroundfield_filename
        cur_wf_fn = args.wavefield_filename
        cur_freq_fn = args.freq_filename

        # 从当前 DataLoader 提取训练集 Tensor
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

            # 替换文件名为 replay 阶段
            args.vel_filename = base_vel_filename.replace(base_freq_tag, replay_freq_tag)
            args.backgroundfield_filename = base_bg_filename.replace(base_freq_tag, replay_freq_tag)
            args.wavefield_filename = base_wf_filename.replace(base_freq_tag, replay_freq_tag)
            args.freq_filename = base_freq_filename  # freq 文件不含 stage 标签，保持原文件名

            # 使用 replay 阶段的 data_dir
            if 'data_dir' in replay_config:
                args.load_path = replay_config['data_dir']
                args.vel_filename = os.path.basename(args.vel_filename)
                args.backgroundfield_filename = os.path.basename(args.backgroundfield_filename)
                args.wavefield_filename = os.path.basename(args.wavefield_filename)
                args.freq_filename = os.path.basename(args.freq_filename)

            print(f'    [Replay] 加载 Stage {replay_idx} [{replay_config["name"]}] 数据: {args.load_path}/{args.vel_filename}')

            replay_dl, _ = prepare_training_dataloaders(args, device)
            replay_ds = replay_dl['train'].dataset
            replay_tensors = replay_ds.tensors

            # 按 replay_ratio 随机采样子集
            replay_ratio = replay_config.get('replay_ratio', 1.0)
            n_replay = replay_tensors[0].shape[0]
            if replay_ratio < 1.0:
                n_sample = max(1, int(n_replay * replay_ratio))
                perm = torch.randperm(n_replay)[:n_sample]
                replay_tensors = tuple(t[perm] for t in replay_tensors)
                print(f'    [Replay] Stage {replay_idx}: {n_sample}/{n_replay} 样本 (ratio={replay_ratio})')

            combined_vel = torch.cat([combined_vel, replay_tensors[0]], dim=0)
            combined_UU0 = torch.cat([combined_UU0, replay_tensors[1]], dim=0)
            combined_labels = torch.cat([combined_labels, replay_tensors[2]], dim=0)
            if has_freq_replay and len(replay_tensors) >= 4:
                combined_freq = torch.cat([combined_freq, replay_tensors[3]], dim=0)

        # 恢复当前阶段的文件名和 load_path
        args.vel_filename = cur_vel_fn
        args.backgroundfield_filename = cur_bg_fn
        args.wavefield_filename = cur_wf_fn
        args.freq_filename = cur_freq_fn
        args.load_path = current_stage_load_path
        

        # 重建训练 DataLoader
        pin_mem = device.type == 'cuda'
        num_workers = 4
        prefetch_factor = 2

        if has_freq_replay:
            new_train_ds = TensorDataset(combined_vel, combined_UU0, combined_labels, combined_freq)
        else:
            new_train_ds = TensorDataset(combined_vel, combined_UU0, combined_labels)

        dataloader['train'] = DataLoader(
            new_train_ds,
            batch_size=args.batch_size_v, shuffle=True, drop_last=True,
            pin_memory=pin_mem, num_workers=num_workers, prefetch_factor=prefetch_factor,
        )

        print(f'    [Replay] 训练集合并完成: {combined_vel.shape[0]} 样本 (含 replay)')

    # ---- 4. 初始化优化器与调度器 ----
    optimizer = optim.Adam(model.parameters(), lr=stage_lr, weight_decay=args.weight_decay)

    scheduler_type = getattr(args, 'scheduler_type', 'plateau')
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=getattr(args, 'cosine_T_0', 1000),
            T_mult=getattr(args, 'cosine_T_mult', 2),
            eta_min=getattr(args, 'cosine_eta_min', 1e-6),
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.factor, patience=args.patience, min_lr=args.min_lr
        )

    use_warmup = getattr(args, 'use_warmup', False) and stage_warmup > 0
    warmup_scheduler = None
    if use_warmup:
        warmup_scheduler = WarmupScheduler(
            optimizer, warmup_epochs=stage_warmup, base_lr=stage_lr,
            warmup_start_lr=stage_lr / 10., warmup_strategy="linear"
        )

    # ---- 5. 训练状态 ----
    loss_log, loss_pde_log, loss_data_log, loss_reg_log, loss_env_log = [], [], [], [], []
    valid_u_loss, valid_f_loss = [], []
    category_valid_u_loss = {name: [] for name in dataloader.get('valid_by_category', {})}
    category_valid_f_loss = {name: [] for name in dataloader.get('valid_by_category', {})}
    first_flag = True
    pde_norm_coe, data_norm_coe, env_norm_coe = 1., 1., 1.

    # ---- 6. Sobol 引擎 ----
    use_sobol = getattr(args, 'sampling_strategy', 'original') == 'sobol'
    if use_sobol:
        sobol_engine, valid_sobol_points, sobol_pts, sobol_steps = _init_sobol_sampling(args, device)

    # ---- 7. 主训练循环 ----
    optimizer.zero_grad()
    step_counter = 0
    pbar = tqdm(range(stage_niter), desc=f"Stage {stage_idx} [{stage_name}]", dynamic_ncols=True)

    # epoch-level 共享 y_ran 采样状态
    epoch_prob = None
    epoch_score = None

    for i in pbar:
        b = cosine_pde_weight(args, i, pde_target_weight)
        if (not getattr(args, 'use_pde_weight_ramp', False)
                and args.if_adjust and i > args.adjust_from
                and (i - args.adjust_from) % args.adjust_every == 0):
            decay_times = i // args.adjust_every
            a = max(a * (args.adjust_speed ** (-decay_times)), 2e-1)
            b, c = 1, 0

        model.train()
        batch_loss, batch_u_loss, batch_f_loss, batch_r_loss, batch_env_loss = [], [], [], [], []

        # y_ran: 每 epoch 生成一次（epoch shared 路径）
        y_ran_epoch_shared = None
        if getattr(args, 'use_y_ran', False) and getattr(args, 'use_epoch_shared_y_ran', False):
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
                y_ran_epoch_shared = sample_shared_y_ran_from_epoch_prob(
                    prob=epoch_prob,
                    args=args,
                    num_pts=args.y_ran_num_pts,
                    structure_ratio=args.y_ran_structure_ratio,
                    surface_ratio=args.y_ran_surface_ratio,
                    uniform_ratio=args.y_ran_uniform_ratio,
                    source_ratio=args.y_ran_source_ratio,
                    surface_depth_grids=args.y_ran_surface_depth_grids,
                )

        has_freq = len(dataloader['train'].dataset.tensors) >= 4
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

            if y_ran_epoch_shared is not None:
                y_ran = y_ran_epoch_shared.unsqueeze(0).expand(
                    vel_batch.shape[0], -1, -1
                ).clone().requires_grad_(True)
            elif getattr(args, 'use_y_ran', False):
                with torch.no_grad():
                    y_ran = model.generate_structure_aware_y_ran(
                        vel_batch, num_pts=args.y_ran_num_pts
                    )
            else:
                y_ran = None

            if use_sobol:
                for _ in range(sobol_steps):
                    y_sobol = _draw_sobol_batch(
                        sobol_engine, sobol_pts, args, device, vel_batch.shape[0]
                    )
                    loss, loss_f, loss_u, loss_r, loss_env = model.loss(
                        vel_batch, y_sobol, UU0_batch, labels_batch,
                        a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
                        freq_batch=freq_batch, y_ran=y_ran, epoch=i
                    )
                    loss, loss_env = _maybe_add_lpips_loss(
                        args, model, loss, loss_env,
                        vel_batch, UU0_batch, labels_batch, freq_batch, i
                    )

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    batch_loss.append(loss.item())
                    batch_u_loss.append(loss_u.item())
                    batch_f_loss.append(loss_f.item())
                    batch_r_loss.append(loss_r.item() if isinstance(loss_r, torch.Tensor) else loss_r)
                    batch_env_loss.append(loss_env.item())

                    del loss, loss_f, loss_u, loss_r, loss_env, y_sobol

            else:

                for batch in dataloader['train_y']:
                    y_batch = batch[0].to(device)
                    y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                    loss, loss_f, loss_u, loss_r, loss_env = model.loss(
                        vel_batch, y_batch, UU0_batch, labels_batch,
                        a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe, freq_batch=freq_batch,
                        y_ran=y_ran, epoch=i
                    )
                    loss, loss_env = _maybe_add_lpips_loss(
                        args, model, loss, loss_env,
                        vel_batch, UU0_batch, labels_batch, freq_batch, i
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

        # ---- 记录损失 ----
        avg_loss = np.mean(batch_loss) if batch_loss else 0

        if first_flag:
            data_norm_coe = np.mean(batch_u_loss) if batch_u_loss else 1.0
            pde_norm_coe = np.mean(batch_f_loss) if batch_f_loss else 1.0
            env_norm_coe = np.mean(batch_env_loss) if batch_env_loss else 1.0
            loss_log.append(a + b)
            loss_data_log.append(1.)
            loss_pde_log.append(1.)
            loss_env_log.append(1.)
            loss_reg_log.append(np.mean(batch_r_loss) / (args.batch_size * args.batch_size_v) if batch_r_loss else 0)
            first_flag = False
        else:
            loss_log.append(avg_loss)
            loss_data_log.append(np.mean(batch_u_loss) if batch_u_loss else 0)
            loss_pde_log.append(np.mean(batch_f_loss) if batch_f_loss else 0)
            loss_env_log.append(np.mean(batch_env_loss) if batch_env_loss else 0)
            loss_reg_log.append(np.mean(batch_r_loss) if batch_r_loss else 0)

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'Total': f"{avg_loss:.4e}",
            'PDE': f"{loss_pde_log[-1]:.4e}",
            'Data': f"{loss_data_log[-1]:.4e}",
            'Env': f"{loss_env_log[-1]:.4e}",
            'PDE_W': f"{b:.3f}",
            'LR': f"{current_lr:.2e}"
        })

        if use_warmup and warmup_scheduler is not None and i <= stage_warmup:
            warmup_scheduler.step(i)
        elif scheduler_type == 'cosine':
            scheduler.step()
        else:
            scheduler.step(avg_loss)

        # ---- 验证 ----
        if i % args.validate_every == 0:
            model.eval()
            u_loss, f_loss = evaluate_valid_loader(
                model, dataloader['valid'], dataloader['valid_y'], device, has_freq,
                a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
                use_sobol, valid_sobol_points if use_sobol else None,
            )
            valid_u_loss.append(u_loss)
            valid_f_loss.append(f_loss)

            for category_name, category_loader in dataloader.get('valid_by_category', {}).items():
                cat_u_loss, cat_f_loss = evaluate_valid_loader(
                    model, category_loader, dataloader['valid_y'], device, has_freq,
                    a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
                    use_sobol, valid_sobol_points if use_sobol else None,
                )
                category_valid_u_loss.setdefault(category_name, []).append(cat_u_loss)
                category_valid_f_loss.setdefault(category_name, []).append(cat_f_loss)

        # ---- 可视化 ----
        if i % args.save_fig_every == 0:
            vel_pred = plot_data["vel_pred"]
            UU0_pred = plot_data["UU0_pred"]
            labels_pred = plot_data["labels_pred"]
            vel_test = plot_data["vel_test"]
            UU0_test = plot_data["UU0_test"]
            labels_test = plot_data["labels_test"]
            has_freq_plot = plot_data.get("has_freq", False)
            freq_pred = plot_data["freq_valid"] if has_freq_plot else None
            freq_test = plot_data["freq_train"] if has_freq_plot else None

            plot_loss(i, save_doc, loss_log, loss_data_log, loss_pde_log, valid_u_loss, valid_f_loss,
                      suffix=f'_stage{stage_idx}')
            plot_category_valid_loss(
                i, save_doc, category_valid_u_loss, category_valid_f_loss,
                suffix=f'_stage{stage_idx}'
            )

            test_plot(args, model, fno, i, dataloader["pred"], vel_pred, UU0_pred, labels_pred,
                      f'valid_stage{stage_idx}', if_fine_tune=False, freq=freq_pred)
            test_plot(args, model, fno, i, dataloader["test"], vel_test, UU0_test, labels_test,
                      f'train_stage{stage_idx}', if_fine_tune=False, freq=freq_test)
            plot_sinlge(model, args, 6, vel_test, UU0_test, labels_test, freq=freq_test)

        # ---- 模型保存 ----
        if i % args.save_model_every == 0:
            pbar.write(f'>>> Stage {stage_idx} Epoch {i} | Total Loss {loss_log[-1]:.4e} | PDE Loss {loss_pde_log[-1]:.4e}')

            checkpoint = {
                'model_state_dict': model.state_dict(),
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
            np.save(os.path.join(save_doc, f'valid_category_data_loss_stage{stage_idx}.npy'), category_valid_u_loss)
            np.save(os.path.join(save_doc, f'valid_category_pde_loss_stage{stage_idx}.npy'), category_valid_f_loss)

    # ---- 8. 阶段结束：保存最终权重 ----
    final_path = os.path.join(save_doc, f'{args.filename}_stage{stage_idx}_final_weights_{args.nz}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'stage': stage_idx,
    }, final_path)
    print(f'✅ Stage {stage_idx} [{stage_name}] 完成，权重已保存: {final_path}')

    np.save(os.path.join(save_doc, f'loss_log_stage{stage_idx}.npy'), loss_log)
    np.save(os.path.join(save_doc, f'loss_data_log_stage{stage_idx}.npy'), loss_data_log)
    np.save(os.path.join(save_doc, f'loss_pde_log_stage{stage_idx}.npy'), loss_pde_log)
    np.save(os.path.join(save_doc, f'loss_env_log_stage{stage_idx}.npy'), loss_env_log)

    return model


def train_staged(args, device):
    """三阶段课程学习训练"""
    model = Pi_DeepONet(args).to(device)
    print(f"PI_DeepONet 模型总参数数量：{count_parameters(model)}")

    fno = None
    if args.use_fno_as_label:
        fno = FNO(args).to(device)
        if args.fno_weights_path:
            fno.load_state_dict(torch.load(args.fno_weights_path, map_location=device)['model_state_dict'])
            print(f"已加载 FNO 权重: {args.fno_weights_path}")
        fno.eval()

    save_doc = args.save_doc
    os.makedirs(save_doc, exist_ok=True)

    # 保存基础文件名模板（供各阶段替换 freq 标签用）
    base_vel_filename = args.vel_filename
    base_bg_filename = args.backgroundfield_filename
    base_wf_filename = args.wavefield_filename
    base_freq_filename = args.freq_filename

    # 检查外部验证集配置
    if hasattr(args, 'ext_val_datasets') and args.ext_val_datasets:
        print(f'⚠️ staged_training 模式下暂不支持外部验证集 (ext_val_datasets)，已忽略')

    stages = args.stages
    print(f'\n{"=" * 60}')
    print(f'三阶段渐进训练计划：共 {len(stages)} 个阶段')
    for si, s in enumerate(stages):
        print(f'  Stage {si}: {s["name"]} | freq [{s["freq_min"]}-{s["freq_max"]}] Hz | '
              f'{s.get("NIter", "?")} epochs | lr={s.get("lr", "?")}')
    print(f'{"=" * 60}\n')

    for stage_idx, stage_config in enumerate(stages):
        print(f'\n{"=" * 60}')
        print(f'>>> 开始 Stage {stage_idx}: {stage_config["name"]} '
              f'[{stage_config["freq_min"]}-{stage_config["freq_max"]} Hz]')
        print(f'{"=" * 60}')

        model = _train_stage(
            args, model, fno, device, stage_idx, stage_config,
            base_vel_filename, base_bg_filename, base_wf_filename, base_freq_filename
        )

    print(f'\n{"=" * 60}')
    print(f'全部 {len(stages)} 个阶段训练完毕！')
    print(f'{"=" * 60}')


def train_single(args, device):
    """原始单阶段训练（原有 train() 逻辑完整保留）"""
    dataloader, plot_data = prepare_training_dataloaders(args, device)

    # 加载外部验证集
    ext_val_sets = {}
    if hasattr(args, 'ext_val_datasets'):
        for name, config in args.ext_val_datasets.items():
            loader, p_data = prepare_external_val_dataset(
                args,
                prefix=config['prefix'],
                loc_target=config['loc_target'],
                y_pred_grid=plot_data["y_pred"]
            )
            ext_val_sets[name] = {"loader": loader, "plot_data": p_data}

    model = Pi_DeepONet(args).to(device)
    model._init_weights()
    print(f"PI_DeepONet 模型总参数数量：{count_parameters(model)}")

    fno = None
    if args.use_fno_as_label:
        fno = FNO(args).to(device)
        if args.fno_weights_path:
            fno.load_state_dict(torch.load(args.fno_weights_path, map_location=device)['model_state_dict'])
            print(f"已加载 FNO 权重: {args.fno_weights_path}")
        fno.eval()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler 选择
    scheduler_type = getattr(args, 'scheduler_type', 'plateau')
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=getattr(args, 'cosine_T_0', 1000),
            T_mult=getattr(args, 'cosine_T_mult', 2),
            eta_min=getattr(args, 'cosine_eta_min', 1e-6),
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.factor, patience=args.patience, min_lr=args.min_lr
        )

    use_warmup = getattr(args, 'use_warmup', False)
    warmup_scheduler = None
    if use_warmup:
        warmup_scheduler = WarmupScheduler(
            optimizer, warmup_epochs=args.warmup_epochs,
            base_lr=args.lr, warmup_start_lr=args.lr / 10.,
            warmup_strategy="linear"
        )

    print(f"Scheduler: {scheduler_type}" + (f" (warmup {args.warmup_epochs} epochs)" if use_warmup else ""))

    loss_log, loss_pde_log, loss_data_log, loss_reg_log, loss_env_log = [], [], [], [], []
    valid_u_loss, valid_f_loss = [], []
    category_valid_u_loss = {name: [] for name in dataloader.get('valid_by_category', {})}
    category_valid_f_loss = {name: [] for name in dataloader.get('valid_by_category', {})}

    a, b, c, d = args.a, args.b, args.c, args.d
    pde_target_weight = b
    first_flag = True
    pde_norm_coe, data_norm_coe, env_norm_coe = 1., 1., 1.

    use_sobol = getattr(args, 'sampling_strategy', 'original') == 'sobol'
    if use_sobol:
        sobol_engine, valid_sobol_points, sobol_pts, sobol_steps = _init_sobol_sampling(args, device)

    optimizer.zero_grad()
    pbar = tqdm(range(args.NIter), desc="Training Progress", dynamic_ncols=True)
    step_counter = 0

    # epoch-level 共享 y_ran 采样状态
    epoch_prob = None
    epoch_score = None

    for i in pbar:
        b = cosine_pde_weight(args, i, pde_target_weight)
        if (not getattr(args, 'use_pde_weight_ramp', False)
                and args.if_adjust and i > args.adjust_from
                and (i - args.adjust_from) % args.adjust_every == 0):
            decay_times = i // args.adjust_every
            a = max(a * (args.adjust_speed ** (-decay_times)), 2e-1)
            b, c = 1, 0

        model.train()
        batch_loss, batch_u_loss, batch_f_loss, batch_r_loss, batch_env_loss = [], [], [], [], []

        # y_ran: 每 epoch 生成一次（epoch shared 路径）
        y_ran_epoch_shared = None
        if getattr(args, 'use_y_ran', False) and getattr(args, 'use_epoch_shared_y_ran', False):
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
                y_ran_epoch_shared = sample_shared_y_ran_from_epoch_prob(
                    prob=epoch_prob,
                    args=args,
                    num_pts=args.y_ran_num_pts,
                    structure_ratio=args.y_ran_structure_ratio,
                    surface_ratio=args.y_ran_surface_ratio,
                    uniform_ratio=args.y_ran_uniform_ratio,
                    source_ratio=args.y_ran_source_ratio,
                    surface_depth_grids=args.y_ran_surface_depth_grids,
                )

        has_freq = len(dataloader['train'].dataset.tensors) >= 4
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

            if y_ran_epoch_shared is not None:
                y_ran = y_ran_epoch_shared.unsqueeze(0).expand(
                    vel_batch.shape[0], -1, -1
                ).clone().requires_grad_(True)
            elif getattr(args, 'use_y_ran', False):
                with torch.no_grad():
                    y_ran = model.generate_structure_aware_y_ran(
                        vel_batch, num_pts=args.y_ran_num_pts
                    )
            else:
                y_ran = None

            if use_sobol:
                for _ in range(sobol_steps):
                    y_sobol = _draw_sobol_batch(
                        sobol_engine, sobol_pts, args, device, vel_batch.shape[0]
                    )
                    loss, loss_f, loss_u, loss_r, loss_env = model.loss(
                        vel_batch, y_sobol, UU0_batch, labels_batch,
                        a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
                        freq_batch=freq_batch, y_ran=y_ran, epoch=i
                    )
                    loss, loss_env = _maybe_add_lpips_loss(
                        args, model, loss, loss_env,
                        vel_batch, UU0_batch, labels_batch, freq_batch, i
                    )

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    batch_loss.append(loss.item())
                    batch_u_loss.append(loss_u.item())
                    batch_f_loss.append(loss_f.item())
                    batch_r_loss.append(loss_r.item() if isinstance(loss_r, torch.Tensor) else loss_r)
                    batch_env_loss.append(loss_env.item())

                    del loss, loss_f, loss_u, loss_r, loss_env, y_sobol

            else:

                for batch in dataloader['train_y']:
                    y_batch = batch[0].to(device)
                    y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                    loss, loss_f, loss_u, loss_r, loss_env = model.loss(
                        vel_batch, y_batch, UU0_batch, labels_batch,
                        a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe, freq_batch=freq_batch,
                        y_ran=y_ran, epoch=i
                    )
                    loss, loss_env = _maybe_add_lpips_loss(
                        args, model, loss, loss_env,
                        vel_batch, UU0_batch, labels_batch, freq_batch, i
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

        avg_loss = np.mean(batch_loss) if batch_loss else 0

        if first_flag:
            data_norm_coe = np.mean(batch_u_loss) if batch_u_loss else 1.0
            pde_norm_coe = np.mean(batch_f_loss) if batch_f_loss else 1.0
            env_norm_coe = np.mean(batch_env_loss) if batch_env_loss else 1.0
            loss_log.append(a + b)
            loss_data_log.append(1.)
            loss_pde_log.append(1.)
            loss_env_log.append(1.)
            loss_reg_log.append(np.mean(batch_r_loss) / (args.batch_size * args.batch_size_v) if batch_r_loss else 0)
            first_flag = False
        else:
            loss_log.append(avg_loss)
            loss_data_log.append(np.mean(batch_u_loss) if batch_u_loss else 0)
            loss_pde_log.append(np.mean(batch_f_loss) if batch_f_loss else 0)
            loss_env_log.append(np.mean(batch_env_loss) if batch_env_loss else 0)
            loss_reg_log.append(np.mean(batch_r_loss) if batch_r_loss else 0)

        current_lr = optimizer.param_groups[0]['lr']

        pbar.set_postfix({
            'Total': f"{avg_loss:.4e}",
            'PDE': f"{loss_pde_log[-1]:.4e}",
            'Data': f"{loss_data_log[-1]:.4e}",
            'Env': f"{loss_env_log[-1]:.4e}",
            'PDE_W': f"{b:.3f}",
            'LR': f"{current_lr:.2e}"
        })

        if use_warmup and warmup_scheduler is not None and i <= args.warmup_epochs:
            warmup_scheduler.step(i)
        elif scheduler_type == 'cosine':
            scheduler.step()
        else:
            scheduler.step(avg_loss)

        # ---- 验证 ----
        if i % args.validate_every == 0:
            model.eval()
            u_loss, f_loss = evaluate_valid_loader(
                model, dataloader['valid'], dataloader['valid_y'], device, has_freq,
                a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
                use_sobol, valid_sobol_points if use_sobol else None,
            )
            valid_u_loss.append(u_loss)
            valid_f_loss.append(f_loss)

            for category_name, category_loader in dataloader.get('valid_by_category', {}).items():
                cat_u_loss, cat_f_loss = evaluate_valid_loader(
                    model, category_loader, dataloader['valid_y'], device, has_freq,
                    a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
                    use_sobol, valid_sobol_points if use_sobol else None,
                )
                category_valid_u_loss.setdefault(category_name, []).append(cat_u_loss)
                category_valid_f_loss.setdefault(category_name, []).append(cat_f_loss)

        # ---- 可视化 ----
        if i % args.save_fig_every == 0:
            vel_pred, UU0_pred, labels_pred = plot_data["vel_pred"], plot_data["UU0_pred"], plot_data["labels_pred"]
            vel_test, UU0_test, labels_test = plot_data["vel_test"], plot_data["UU0_test"], plot_data["labels_test"]
            has_freq = plot_data.get("has_freq", False)
            freq_pred = plot_data["freq_valid"] if has_freq else None
            freq_test = plot_data["freq_train"] if has_freq else None

            plot_loss(i, args.save_doc, loss_log, loss_data_log, loss_pde_log, valid_u_loss, valid_f_loss)
            plot_category_valid_loss(i, args.save_doc, category_valid_u_loss, category_valid_f_loss)

            test_plot(args, model, fno, i, dataloader["pred"], vel_pred, UU0_pred, labels_pred, 'valid_without_fine_tune', if_fine_tune=False, freq=freq_pred)
            test_plot(args, model, fno, i, dataloader["test"], vel_test, UU0_test, labels_test, 'train', if_fine_tune=False, freq=freq_test)
            plot_sinlge(model, args, 6, vel_test, UU0_test, labels_test, freq=freq_test)

        # ---- 模型保存 ----
        if i % args.save_model_every == 0:
            pbar.write(f'>>> Epoch {i} | 保存 Checkpoint: Total Loss {loss_log[-1]:.4e} | PDE Loss {loss_pde_log[-1]:.4e}')

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }

            torch.save(checkpoint, os.path.join(args.save_doc, f'{args.filename}_PI_model_{i}epoch_weights_{args.nz}.pth'))
            np.save(os.path.join(args.save_doc, 'loss_log.npy'), loss_log)
            np.save(os.path.join(args.save_doc, 'loss_data_log.npy'), loss_data_log)
            np.save(os.path.join(args.save_doc, 'loss_pde_log.npy'), loss_pde_log)
            np.save(os.path.join(args.save_doc, 'loss_env_log.npy'), loss_env_log)
            np.save(os.path.join(args.save_doc, 'valid_category_data_loss.npy'), category_valid_u_loss)
            np.save(os.path.join(args.save_doc, 'valid_category_pde_loss.npy'), category_valid_f_loss)

    # 最终保存
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, os.path.join(args.save_doc, f'{args.filename}_PI_model_{i}epoch_weights_{args.nz}.pth'))
    np.save(os.path.join(args.save_doc, 'loss_log.npy'), loss_log)
    np.save(os.path.join(args.save_doc, 'loss_data_log.npy'), loss_data_log)
    np.save(os.path.join(args.save_doc, 'loss_pde_log.npy'), loss_pde_log)
    np.save(os.path.join(args.save_doc, 'loss_env_log.npy'), loss_env_log)
    np.save(os.path.join(args.save_doc, 'valid_category_data_loss.npy'), category_valid_u_loss)
    np.save(os.path.join(args.save_doc, 'valid_category_pde_loss.npy'), category_valid_f_loss)


def train(args):
    try:
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

        if getattr(args, 'staged_training', False):
            train_staged(args, device)
        else:
            train_single(args, device)

    except Exception as e:
        print(f"训练过程中断出错: {e}")
        raise
