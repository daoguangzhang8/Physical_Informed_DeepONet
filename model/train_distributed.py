"""
单机多卡分布式训练模块 (Single-Machine Multi-GPU Distributed Training)

使用方法:
    python main2.py  (根据 config.py 中的 use_parallel 自动选择单卡/多卡)
"""

import os
import copy
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from Labconfig import *
from model.utils import (
    setup_distributed,
    cleanup_distributed,
    wrap_model_for_distributed,
    is_main_process,
    reduce_tensor,
    build_epoch_velocity_gradient_prob,
    sample_shared_y_ran_from_epoch_prob,
)
from model.dataloader import prepare_training_dataloaders, prepare_external_val_dataset
from model.PI_DeepOnet import Pi_DeepONet
from model.FNO import FNO
from model.plotting import plot_loss, test_plot, plot_sinlge, fine_tuning
from model.utils import count_parameters, WarmupScheduler


def _train_worker(rank, world_size, args):
    """
    分布式训练工作进程 (每个 GPU 运行一个)

    Args:
        rank: 当前进程的 rank (0, 1, 2, ...)
        world_size: 总进程数 (等于 GPU 数量)
        args: 配置参数对象
    """
    # 初始化分布式环境
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    args.device = rank  # 确保绘图等函数使用正确的 GPU

    if is_main_process(rank):
        print("=" * 60)
        print(f"单机多卡分布式训练模式")
        print(f"GPU 数量: {world_size}")
        print(f"=" * 60)

    # ==========================================
    # 加载数据
    # ==========================================
    if is_main_process(rank):
        print("正在加载数据...")

    dataloader, plot_data = prepare_training_dataloaders(args, device)

    # 加载外部验证集 (只在主进程)
    ext_val_sets = {}
    if hasattr(args, 'ext_val_datasets') and is_main_process(rank):
        for name, config in args.ext_val_datasets.items():
            loader, p_data = prepare_external_val_dataset(
                args,
                prefix=config['prefix'],
                loc_target=config['loc_target'],
                y_pred_grid=plot_data["y_pred"]
            )
            ext_val_sets[name] = {"loader": loader, "plot_data": p_data}

    # 替换为 DistributedSampler
    train_sampler = DistributedSampler(
        dataloader['train'].dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    dataloader['train'] = DataLoader(
        dataloader['train'].dataset,
        batch_size=args.batch_size_v,
        sampler=train_sampler,
        num_workers=max(1, 4),
        pin_memory=True
    )

    if is_main_process(rank):
        print(f"已启用 DistributedSampler (world_size={world_size})")

    # 检测数据集是否包含频率信息
    has_freq = len(dataloader['train'].dataset.tensors) >= 4

    # ==========================================
    # 创建模型
    # ==========================================
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

    # 加载预训练的 FNO 模型
    fno = FNO(args).to(device)
    if args.use_fno_as_label and args.fno_weights_path:
        fno.load_state_dict(torch.load(args.fno_weights_path, map_location=device)['model_state_dict'])
        if is_main_process(rank):
            print(f"已加载 FNO 权重: {args.fno_weights_path}")
    fno.eval()

    # ==========================================
    # 优化器与调度器
    # ==========================================
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * world_size, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.factor, patience=args.patience, min_lr=args.min_lr  * world_size
    )
    warmup_scheduler = WarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        base_lr=args.lr * world_size,
        warmup_start_lr=args.lr  * world_size/ 10.,
        warmup_strategy="linear",
        after_scheduler=None
    )

    # ==========================================
    # 训练状态初始化
    # ==========================================
    loss_log, loss_pde_log, loss_data_log, loss_reg_log, loss_env_log = [], [], [], [], []
    valid_u_loss, valid_f_loss = [], []

    # epoch-level 共享 y_ran 采样状态
    epoch_prob = None
    epoch_score = None

    a, b, c, d = args.a, args.b, args.c, args.d
    first_flag = True
    pde_norm_coe = 1.
    data_norm_coe = 1.
    env_norm_coe = 1.

    # ==========================================
    # 主训练循环
    # ==========================================
    optimizer.zero_grad()

    # 只在主进程显示进度条
    if is_main_process(rank):
        pbar = tqdm(range(args.NIter), desc="Training Progress", dynamic_ncols=True)
    else:
        pbar = range(args.NIter)

    for i in pbar:
        # 动态调整损失权重
        if args.if_adjust and i > args.adjust_from and (i - args.adjust_from) % args.adjust_every == 0:
            decay_times = i // args.adjust_every
            a = max(a * (args.adjust_speed ** (-decay_times)), 2e-1)
            b, c = 1, 0

        model.train()
        batch_loss, batch_u_loss, batch_f_loss, batch_r_loss, batch_env_loss = [], [], [], [], []

        # 设置 epoch 以确保每个 epoch shuffle 不同
        dataloader['train'].sampler.set_epoch(i)

        # 预收集 coordinate batches（每 epoch 一次，避免在 velocity 循环内重复物化）
        coord_batches = list(dataloader['train_y'])
        n_coord = len(coord_batches)

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

            # 每个 velocity batch 结束后更新参数
            optimizer.step()
            optimizer.zero_grad()

        # 跨进程平均损失
        avg_loss = np.mean(batch_loss) if batch_loss else 0
        avg_u_loss = np.mean(batch_u_loss) if batch_u_loss else 0
        avg_f_loss = np.mean(batch_f_loss) if batch_f_loss else 0

        loss_tensor = torch.tensor([avg_loss, avg_u_loss, avg_f_loss], device=device)
        loss_tensor = reduce_tensor(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss, avg_u_loss, avg_f_loss = loss_tensor.cpu().numpy()

        avg_env_loss = np.mean(batch_env_loss) if batch_env_loss else 0

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

        # 更新进度条 (只在主进程)
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
        if i <= args.warmup_epochs:
            warmup_scheduler.step(i)
        else:
            scheduler.step(avg_loss)

        # ==========================================
        # 验证环节 (只在主进程)
        # ==========================================
        if i % args.validate_every == 0 and is_main_process(rank):
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

                for batch in dataloader['valid_y']:
                    y_batch = batch[0].to(device)
                    y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                    _, loss_f_valid, loss_u_valid, _, _ = model.module.loss(
                        vel_batch, y_batch, UU0_batch, labels_batch,
                        a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe, freq_batch=freq_batch
                    )
                    batch_u_loss.append(loss_u_valid.item())
                    batch_f_loss.append(loss_f_valid.item())

            valid_u_loss.append(np.mean(batch_u_loss) if batch_u_loss else 0.0)
            valid_f_loss.append(np.mean(batch_f_loss) if batch_f_loss else 1.0)

        # ==========================================
        # 可视化与绘图 (只在主进程)
        # ==========================================
        if i % args.save_fig_every == 0 and is_main_process(rank):
            vel_pred = plot_data["vel_pred"]
            UU0_pred = plot_data["UU0_pred"]
            labels_pred = plot_data["labels_pred"]
            vel_test = plot_data["vel_test"]
            UU0_test = plot_data["UU0_test"]
            labels_test = plot_data["labels_test"]
            freq_pred = plot_data["freq_valid"][0:1] if has_freq else None
            freq_test = plot_data["freq_train"][0:1] if has_freq else None

            plot_loss(i, args.save_doc, loss_log, loss_data_log, loss_pde_log, valid_u_loss, valid_f_loss)

            if i % (args.save_fig_every * 20) == 0 and i > 0 and args.if_finetune:
                if ext_val_sets:
                    marmousi_data = ext_val_sets['Marmousi']
                    v_m_test = marmousi_data["plot_data"]["v_test"]
                    u0_m_test = marmousi_data["plot_data"]["u0_test"]
                    lab_m_test = marmousi_data["plot_data"]["lab_test"]
                    dataloader_m_y_full = marmousi_data["loader"]

                    test_plot(args, model.module, fno, i, dataloader_m_y_full,
                              v_m_test, u0_m_test, lab_m_test, 'FT_Marmousi', if_fine_tune=True)

            test_plot(args, model.module, fno, i, dataloader["pred"],
                      vel_pred, UU0_pred, labels_pred, 'valid_without_fine_tune', if_fine_tune=False, freq=freq_pred)
            test_plot(args, model.module, fno, i, dataloader["test"],
                      vel_test, UU0_test, labels_test, 'train', if_fine_tune=False, freq=freq_test)
            plot_sinlge(model.module, args, 6, vel_test, UU0_test, labels_test, freq=freq_test)

        # ==========================================
        # 模型保存 (只在主进程)
        # ==========================================
        if i % args.save_model_every == 0 and is_main_process(rank):
            if isinstance(pbar, tqdm):
                pbar.write(f'>>> Epoch {i} | 保存 Checkpoint: Total Loss {loss_log[-1]:.4e} | PDE Loss {loss_pde_log[-1]:.4e}')

            checkpoint = {
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }

            torch.save(checkpoint, os.path.join(args.save_doc, f'{args.filename}_PI_model_{i}epoch_weights_{args.nz}.pth'))
            np.save(os.path.join(args.save_doc, 'loss_log.npy'), loss_log)
            np.save(os.path.join(args.save_doc, 'loss_data_log.npy'), loss_data_log)
            np.save(os.path.join(args.save_doc, 'loss_pde_log.npy'), loss_pde_log)
            np.save(os.path.join(args.save_doc, 'loss_env_log.npy'), loss_env_log)

    # ==========================================
    # 最终保存与清理
    # ==========================================
    if is_main_process(rank):
        checkpoint = {
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.save_doc, f'{args.filename}_PI_model_final_weights_{args.nz}.pth'))
        print(f"训练完成! 模型已保存到 {args.save_doc}")

    cleanup_distributed()


def train_distributed(args):
    """
    单机多卡分布式训练入口函数

    使用 mp.spawn 内部启动多进程，无需 torchrun

    Args:
        args: 配置参数对象 (config.Args)
    """
    world_size = getattr(args, 'num_gpus', 1)

    print("=" * 60)
    print(f"启动单机多卡分布式训练")
    print(f"GPU 数量: {world_size}")
    print("=" * 60)

    # 使用 mp.spawn 启动多进程
    mp.spawn(
        _train_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


# ==============================================================================
# DDP 课程学习 (Staged Curriculum Training) 相关函数
# ==============================================================================

def _run_stage_training_loop(args, model, fno, device, rank, world_size,
                             dataloader, plot_data, has_freq,
                             optimizer, scheduler, warmup_scheduler,
                             stage_idx, stage_name, stage_niter, stage_warmup,
                             a, b, c, d, save_doc):
    """DDP 单阶段训练循环"""
    loss_log, loss_pde_log, loss_data_log, loss_reg_log, loss_env_log = [], [], [], [], []
    valid_u_loss, valid_f_loss = [], []

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
        if args.if_adjust and i > args.adjust_from and (i - args.adjust_from) % args.adjust_every == 0:
            decay_times = i // args.adjust_every
            a = max(a * (args.adjust_speed ** (-decay_times)), 2e-1)
            b, c = 1, 0

        model.train()
        batch_loss, batch_u_loss, batch_f_loss, batch_r_loss, batch_env_loss = [], [], [], [], []

        dataloader['train'].sampler.set_epoch(i)

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

                Delta_U = model(vel_batch, y_combined, UU0_batch, freq_batch=freq_batch)

                loss, loss_f, loss_u, loss_r, loss_env = model.module.compute_loss(
                    Delta_U, vel_batch, y_batch, UU0_batch, labels_batch,
                    y_combined, a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
                    freq_batch=freq_batch
                )

                loss = loss / n_coord

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


def _train_stage_ddp(args, model, fno, device, rank, world_size,
                     stage_idx, stage_config, save_doc,
                     base_vel_filename, base_bg_filename, base_wf_filename, base_freq_filename):
    """DDP 单阶段训练：数据加载、replay 合并、优化器初始化、训练循环"""
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
        dist.barrier()
        for param in model.module.parameters():
            dist.broadcast(param.data, src=0)
    else:
        if is_main_process(rank):
            print(f'[*] Stage 0: 权重已在 _train_worker_staged 中初始化')

    # ---- 3. 加载数据 ----
    # 固定 seed 确保所有 rank 得到相同的 train/valid 划分
    np.random.seed(1)
    torch.manual_seed(0)
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

            # 固定 seed 确保所有 rank 的 replay 数据划分一致
            np.random.seed(1)
            torch.manual_seed(0)
            replay_dl, _ = prepare_training_dataloaders(args, device)
            replay_ds = replay_dl['train'].dataset
            replay_tensors = replay_ds.tensors

            replay_ratio = replay_config.get('replay_ratio', 1.0)
            n_replay = replay_tensors[0].shape[0]
            if replay_ratio < 1.0:
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
        train_sampler = DistributedSampler(
            dataloader['train'].dataset,
            num_replicas=world_size, rank=rank, shuffle=True
        )
        dataloader['train'] = DataLoader(
            dataloader['train'].dataset,
            batch_size=args.batch_size_v,
            sampler=train_sampler,
            drop_last=True,
            num_workers=4, pin_memory=True, prefetch_factor=2
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

    # ---- 5. 运行训练循环 ----
    model = _run_stage_training_loop(
        args, model, fno, device, rank, world_size,
        dataloader, plot_data, has_freq,
        optimizer, scheduler, warmup_scheduler,
        stage_idx, stage_name, stage_niter, stage_warmup,
        a, b, c, d, save_doc,
    )

    # 恢复原始 load_path，防止跨阶段污染
    args.load_path = original_load_path

    return model, save_doc


def _train_worker_staged(rank, world_size, args):
    """分布式课程学习训练工作进程，在单个 mp.spawn 生命周期内依次执行所有阶段"""
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    args.device = rank

    # 创建模型（全局一次，跨阶段复用）
    model = Pi_DeepONet(args).to(device)

    if is_main_process(rank):
        model._init_weights()
        print(f"[Stage DDP] PI_DeepONet 模型总参数数量: {count_parameters(model)}")

    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    model = wrap_model_for_distributed(model, rank)

    fno = FNO(args).to(device)
    if args.use_fno_as_label and args.fno_weights_path:
        fno.load_state_dict(torch.load(args.fno_weights_path, map_location=device)['model_state_dict'])
        if is_main_process(rank):
            print(f"已加载 FNO 权重: {args.fno_weights_path}")
    fno.eval()

    save_doc = args.save_doc
    if is_main_process(rank):
        os.makedirs(save_doc, exist_ok=True)

    base_vel_filename = args.vel_filename
    base_bg_filename = args.backgroundfield_filename
    base_wf_filename = args.wavefield_filename
    base_freq_filename = args.freq_filename

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


def train_distributed_staged(args):
    """单机多卡分布式 + 课程学习训练入口函数"""
    world_size = getattr(args, 'num_gpus', 1)

    stages = getattr(args, 'stages', [])
    print("=" * 60)
    print(f"启动单机多卡分布式课程学习训练")
    print(f"GPU 数量: {world_size}")
    print(f"训练阶段数: {len(stages)}")
    for si, s in enumerate(stages):
        print(f'  Stage {si}: {s["name"]} | freq [{s["freq_min"]}-{s["freq_max"]}] Hz | '
              f'{s.get("NIter", "?")} epochs | lr={s.get("lr", "?")}')
    print("=" * 60)

    mp.spawn(
        _train_worker_staged,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

