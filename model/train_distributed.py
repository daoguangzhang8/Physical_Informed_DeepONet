"""
单机多卡分布式训练模块 (Single-Machine Multi-GPU Distributed Training)

使用方法:
    python main2.py  (根据 config.py 中的 use_parallel 自动选择单卡/多卡)
"""

import os
import gc
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
    reduce_tensor
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
    loss_log, loss_pde_log, loss_data_log, loss_reg_log = [], [], [], []
    valid_u_loss, valid_f_loss = [], []

    a, b, c = args.a, args.b, args.c
    first_flag = True
    pde_norm_coe = 1.
    data_norm_coe = 1.

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

        # 跨进程平均损失
        avg_loss = np.mean(batch_loss) if batch_loss else 0
        avg_u_loss = np.mean(batch_u_loss) if batch_u_loss else 0
        avg_f_loss = np.mean(batch_f_loss) if batch_f_loss else 0

        loss_tensor = torch.tensor([avg_loss, avg_u_loss, avg_f_loss], device=device)
        loss_tensor = reduce_tensor(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss, avg_u_loss, avg_f_loss = loss_tensor.cpu().numpy()

        if first_flag:
            data_norm_coe = avg_u_loss if avg_u_loss > 0 else 1.0
            pde_norm_coe = avg_f_loss if avg_f_loss > 0 else 1.0
            loss_log.append(a + b)
            loss_data_log.append(1.)
            loss_pde_log.append(1.)
            first_flag = False
        else:
            loss_log.append(avg_loss)
            loss_data_log.append(avg_u_loss)
            loss_pde_log.append(avg_f_loss)
            loss_reg_log.append(np.mean(batch_r_loss) / (args.batch_size * args.batch_size_v) if batch_r_loss else 0)

        # 更新进度条 (只在主进程)
        if is_main_process(rank):
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Total': f"{avg_loss:.4e}",
                'PDE': f"{loss_pde_log[-1]:.4e}",
                'Data': f"{loss_data_log[-1]:.4e}",
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

                    _, loss_f_valid, loss_u_valid, _ = model.module.loss(
                        vel_batch, y_batch, UU0_batch, labels_batch,
                        a, b, c, data_norm_coe, pde_norm_coe, freq_batch=freq_batch
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
            plot_sinlge(model.module, args, 6, vel_test, UU0_test, labels_test)

            torch.cuda.empty_cache()
            gc.collect()

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
    torch.cuda.empty_cache()
    gc.collect()


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

