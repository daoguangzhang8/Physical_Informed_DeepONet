import os
import gc
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
    args.freq_filename = base_freq_filename.replace('freq_used', f'freq{freq_range}_used')

    print(f'\n[*] Stage {stage_idx} [{stage_name}] 数据文件:')
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

    # ---- 4. 初始化优化器与调度器 ----
    optimizer = optim.Adam(model.parameters(), lr=stage_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.factor, patience=args.patience, min_lr=args.min_lr
    )
    warmup_scheduler = WarmupScheduler(
        optimizer, warmup_epochs=stage_warmup, base_lr=stage_lr,
        warmup_start_lr=stage_lr / 10., warmup_strategy="linear"
    )

    # ---- 5. 训练状态 ----
    loss_log, loss_pde_log, loss_data_log, loss_reg_log, loss_env_log = [], [], [], [], []
    valid_u_loss, valid_f_loss = [], []
    first_flag = True
    pde_norm_coe, data_norm_coe, env_norm_coe = 1., 1., 1.

    # ---- 6. Sobol 引擎 ----
    use_sobol = getattr(args, 'sampling_strategy', 'original') == 'sobol'
    if use_sobol:
        sobol_engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
        valid_sobol_engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
        sobol_scale = torch.tensor([args.nz * args.dh, args.nx * args.dh], dtype=torch.float32)
        sobol_pts = getattr(args, 'sobol_points_per_epoch', 800)
        valid_sobol_pts = getattr(args, 'valid_sobol_points', 800)
        print(f"Sobol 模式: 每 epoch {sobol_pts} 点, 验证 {valid_sobol_pts} 点")

    # ---- 7. 主训练循环 ----
    optimizer.zero_grad()
    step_counter = 0
    pbar = tqdm(range(stage_niter), desc=f"Stage {stage_idx} [{stage_name}]", dynamic_ncols=True)

    for i in pbar:
        if args.if_adjust and i > args.adjust_from and (i - args.adjust_from) % args.adjust_every == 0:
            decay_times = i // args.adjust_every
            a = max(a * (args.adjust_speed ** (-decay_times)), 2e-1)
            b, c = 1, 0

        model.train()
        batch_loss, batch_u_loss, batch_f_loss, batch_r_loss, batch_env_loss = [], [], [], [], []

        if use_sobol:
            y_sobol_base = sobol_engine.draw(sobol_pts).to(device)
            y_sobol_base = y_sobol_base * sobol_scale.to(device)

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

            if use_sobol:
                y_sobol = y_sobol_base.unsqueeze(0).expand(vel_batch.shape[0], -1, -1).clone()
                y_sobol.requires_grad_(True)

                loss, loss_f, loss_u, loss_r, loss_env = model.loss(
                    vel_batch, y_sobol, UU0_batch, labels_batch,
                    a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe, freq_batch=freq_batch,
                    y_ran=None
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
            'LR': f"{current_lr:.2e}"
        })

        if i <= stage_warmup:
            warmup_scheduler.step(i)
        else:
            scheduler.step(avg_loss)

        # ---- 验证 ----
        if i % args.validate_every == 0:
            model.eval()
            batch_u_loss, batch_f_loss = [], []

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

                if use_sobol:
                    y_valid = valid_sobol_engine.draw(valid_sobol_pts).to(device)
                    y_valid = y_valid * sobol_scale.to(device)
                    y_valid = y_valid.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                    _, loss_f_valid, loss_u_valid, _, _ = model.loss(
                        vel_batch, y_valid, UU0_batch, labels_batch,
                        a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe, freq_batch=freq_batch
                    )
                    batch_u_loss.append(loss_u_valid.item())
                    batch_f_loss.append(loss_f_valid.item())

                    del loss_f_valid, loss_u_valid, y_valid
                else:
                    for batch in dataloader['valid_y']:
                        y_batch = batch[0].to(device)
                        y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                        _, loss_f_valid, loss_u_valid, _, _ = model.loss(
                            vel_batch, y_batch, UU0_batch, labels_batch,
                            a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe, freq_batch=freq_batch
                        )
                        batch_u_loss.append(loss_u_valid.item())
                        batch_f_loss.append(loss_f_valid.item())

            valid_u_loss.append(np.mean(batch_u_loss) if batch_u_loss else 0.0)
            valid_f_loss.append(np.mean(batch_f_loss) if batch_f_loss else 1.0)

        # ---- 可视化 ----
        if i % args.save_fig_every == 0:
            vel_pred = plot_data["vel_pred"]
            UU0_pred = plot_data["UU0_pred"]
            labels_pred = plot_data["labels_pred"]
            vel_test = plot_data["vel_test"]
            UU0_test = plot_data["UU0_test"]
            labels_test = plot_data["labels_test"]
            has_freq_plot = plot_data.get("has_freq", False)
            freq_pred = plot_data["freq_valid"][0:1] if has_freq_plot else None
            freq_test = plot_data["freq_train"][0:1] if has_freq_plot else None

            plot_loss(i, save_doc, loss_log, loss_data_log, loss_pde_log, valid_u_loss, valid_f_loss,
                      suffix=f'_stage{stage_idx}')

            test_plot(args, model, fno, i, dataloader["pred"], vel_pred, UU0_pred, labels_pred,
                      f'valid_stage{stage_idx}', if_fine_tune=False, freq=freq_pred)
            test_plot(args, model, fno, i, dataloader["test"], vel_test, UU0_test, labels_test,
                      f'train_stage{stage_idx}', if_fine_tune=False, freq=freq_test)
            plot_sinlge(model, args, 6, vel_test, UU0_test, labels_test, freq=freq_test)

            torch.cuda.empty_cache()
            gc.collect()

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

    fno = FNO(args).to(device)
    if args.use_fno_as_label and args.fno_weights_path:
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

    fno = FNO(args).to(device)
    if args.use_fno_as_label and args.fno_weights_path:
        fno.load_state_dict(torch.load(args.fno_weights_path, map_location=device)['model_state_dict'])
        print(f"已加载 FNO 权重: {args.fno_weights_path}")
    fno.eval()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.factor, patience=args.patience, min_lr=args.min_lr
    )
    warmup_scheduler = WarmupScheduler(
        optimizer, warmup_epochs=args.warmup_epochs,
        base_lr=args.lr, warmup_start_lr=args.lr / 10.,
        warmup_strategy="linear"
    )

    loss_log, loss_pde_log, loss_data_log, loss_reg_log, loss_env_log = [], [], [], [], []
    valid_u_loss, valid_f_loss = [], []

    a, b, c, d = args.a, args.b, args.c, args.d
    first_flag = True
    pde_norm_coe, data_norm_coe, env_norm_coe = 1., 1., 1.

    use_sobol = getattr(args, 'sampling_strategy', 'original') == 'sobol'
    if use_sobol:
        sobol_engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
        valid_sobol_engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
        sobol_scale = torch.tensor([args.nz * args.dh, args.nx * args.dh], dtype=torch.float32)
        sobol_pts = getattr(args, 'sobol_points_per_epoch', 800)
        valid_sobol_pts = getattr(args, 'valid_sobol_points', 800)
        print(f"Sobol 模式: 每 epoch {sobol_pts} 点, 验证 {valid_sobol_pts} 点")

    optimizer.zero_grad()
    pbar = tqdm(range(args.NIter), desc="Training Progress", dynamic_ncols=True)
    step_counter = 0

    # epoch-level 共享 y_ran 采样状态
    epoch_prob = None
    epoch_score = None

    for i in pbar:
        if args.if_adjust and i > args.adjust_from and (i - args.adjust_from) % args.adjust_every == 0:
            decay_times = i // args.adjust_every
            a = max(a * (args.adjust_speed ** (-decay_times)), 2e-1)
            b, c = 1, 0

        model.train()
        batch_loss, batch_u_loss, batch_f_loss, batch_r_loss, batch_env_loss = [], [], [], [], []

        if use_sobol:
            y_sobol_base = sobol_engine.draw(sobol_pts).to(device)
            y_sobol_base = y_sobol_base * sobol_scale.to(device)

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

            if use_sobol:
                y_sobol = y_sobol_base.unsqueeze(0).expand(vel_batch.shape[0], -1, -1).clone()
                y_sobol.requires_grad_(True)

                loss, loss_f, loss_u, loss_r, loss_env = model.loss(
                    vel_batch, y_sobol, UU0_batch, labels_batch,
                    a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe, freq_batch=freq_batch,
                    y_ran=None
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
            'LR': f"{current_lr:.2e}"
        })

        if i <= args.warmup_epochs:
            warmup_scheduler.step(i)
        else:
            scheduler.step(avg_loss)

        # ---- 验证 ----
        if i % args.validate_every == 0:
            model.eval()
            batch_u_loss, batch_f_loss = [], []

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

                if use_sobol:
                    y_valid = valid_sobol_engine.draw(valid_sobol_pts).to(device)
                    y_valid = y_valid * sobol_scale.to(device)
                    y_valid = y_valid.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                    _, loss_f_valid, loss_u_valid, _, _ = model.loss(
                        vel_batch, y_valid, UU0_batch, labels_batch,
                        a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe, freq_batch=freq_batch
                    )
                    batch_u_loss.append(loss_u_valid.item())
                    batch_f_loss.append(loss_f_valid.item())

                    del loss_f_valid, loss_u_valid, y_valid
                else:
                    for batch in dataloader['valid_y']:
                        y_batch = batch[0].to(device)
                        y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                        _, loss_f_valid, loss_u_valid, _, _ = model.loss(
                            vel_batch, y_batch, UU0_batch, labels_batch,
                            a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe, freq_batch=freq_batch
                        )
                        batch_u_loss.append(loss_u_valid.item())
                        batch_f_loss.append(loss_f_valid.item())

            valid_u_loss.append(np.mean(batch_u_loss) if batch_u_loss else 0.0)
            valid_f_loss.append(np.mean(batch_f_loss) if batch_f_loss else 1.0)

        # ---- 可视化 ----
        if i % args.save_fig_every == 0:
            vel_pred, UU0_pred, labels_pred = plot_data["vel_pred"], plot_data["UU0_pred"], plot_data["labels_pred"]
            vel_test, UU0_test, labels_test = plot_data["vel_test"], plot_data["UU0_test"], plot_data["labels_test"]
            has_freq = plot_data.get("has_freq", False)
            freq_pred = plot_data["freq_valid"][0:1] if has_freq else None
            freq_test = plot_data["freq_train"][0:1] if has_freq else None

            plot_loss(i, args.save_doc, loss_log, loss_data_log, loss_pde_log, valid_u_loss, valid_f_loss)

            test_plot(args, model, fno, i, dataloader["pred"], vel_pred, UU0_pred, labels_pred, 'valid_without_fine_tune', if_fine_tune=False, freq=freq_pred)
            test_plot(args, model, fno, i, dataloader["test"], vel_test, UU0_test, labels_test, 'train', if_fine_tune=False, freq=freq_test)
            plot_sinlge(model, args, 6, vel_test, UU0_test, labels_test, freq=freq_test)

            torch.cuda.empty_cache()
            gc.collect()

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
    finally:
        torch.cuda.empty_cache()
