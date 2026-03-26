import os
import gc
import numpy as np
import torch
from tqdm import tqdm

from Labconfig import *
from model.utils import *
from model.dataloader import *
from model.PI_DeepOnet import Pi_DeepONet
from model.FNO import FNO
from model.ploting import *

def train(args):
    try:
        # ==========================================
        # 1. 环境与数据准备
        # ==========================================
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
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
        
        # ==========================================
        # 2. 模型与优化器初始化
        # ==========================================
        model = Pi_DeepONet(args).to(device)
        model._init_weights()  
        print(f"PI_DeepONet 模型总参数数量：{count_parameters(model)}")
        
        # 加载预训练的 FNO 模型用于生成 labels
        fno = FNO(args).to(device)
        fno.load_state_dict(torch.load('FNO_PI_model_8000epoch_weights.pth', map_location=device)['model_state_dict'])
        fno.eval() # FNO 仅作推断生成 labels，设为评估模式
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.factor, patience=args.patience, min_lr=args.min_lr
        )
        warmup_scheduler = WarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=args.warmup_epochs,  
            base_lr=args.lr,
            warmup_start_lr=args.lr / 10.,  
            warmup_strategy="linear",  
            after_scheduler=None  
        )

        # ==========================================
        # 3. 训练状态与日志记录初始化
        # ==========================================
        loss_log, loss_pde_log, loss_data_log, loss_reg_log = [], [], [], []
        valid_u_loss, valid_f_loss = [], []
        
        a, b, c = args.a, args.b, args.c
        first_flag = True
        pde_norm_coe = 1.
        data_norm_coe = 1.

        # ==========================================
        # 4. 主训练循环 (引入 tqdm 进度条与显存优化)
        # ==========================================
        optimizer.zero_grad()
        pbar = tqdm(range(args.NIter), desc="Training Progress", dynamic_ncols=True)
        
        # 新增：用于全局记录 Batch 步数，确保梯度累加逻辑正确
        step_counter = 0  
        
        for i in pbar:
            # 动态调整损失权重 a
            if args.if_adjust and i > args.adjust_from and (i - args.adjust_from) % args.adjust_every == 0:
                decay_times = i // args.adjust_every 
                a = max(a * (args.adjust_speed ** (-decay_times)), 2e-1)
                b, c = 1, 0

            model.train()
            batch_loss, batch_u_loss, batch_f_loss, batch_r_loss = [], [], [], []
            
            # 遍历训练数据
            for vel_batch, UU0_batch, _ in dataloader['train']:
                vel_batch, UU0_batch = vel_batch.to(device), UU0_batch.to(device)
                
                # 使用 FNO 动态生成 labels
                # labels_batch = labels.to(device)
                with torch.no_grad():
                    labels_batch = fno(vel_batch, UU0_batch).to(device)
                
                # 针对每个空间坐标点集计算损失
                for batch in dataloader['train_y']:
                    y_batch = batch[0].to(device)
                    y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)
                    
                    # 前向传播与计算损失
                    loss, loss_f, loss_u, loss_r = model.loss(
                        vel_batch, y_batch, UU0_batch, labels_batch, 
                        a, b, c, data_norm_coe, pde_norm_coe
                    )
                    
                    # 梯度累加与反向传播
                    loss = loss / args.accumulation_steps
                    loss.backward()
                    
                    # 修复：使用 step_counter 而不是 epoch i 来判断是否更新梯度
                    step_counter += 1  
                    if step_counter % args.accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad() 
                        
                    # 记录真实损失值
                    batch_loss.append(loss.item() * args.accumulation_steps) 
                    batch_u_loss.append(loss_u.item())
                    batch_f_loss.append(loss_f.item())
                    batch_r_loss.append(loss_r.item() if isinstance(loss_r, torch.Tensor) else loss_r)
                    
                    # 新增：手动切断引用，立即释放该 sub-batch 的巨大计算图，极大降低显存占用
                    del loss, loss_f, loss_u, loss_r, y_batch

            # ------------------------------------------
            # 记录当前 Epoch 损失并更新进度条显示
            # ------------------------------------------
            avg_loss = np.mean(batch_loss) if batch_loss else 0
            
            if first_flag:
                data_norm_coe = np.mean(batch_u_loss)
                pde_norm_coe = np.mean(batch_f_loss)
                loss_log.append(a + b)
                loss_data_log.append(1.)
                loss_pde_log.append(1.)
                loss_reg_log.append(np.mean(batch_r_loss) / (args.batch_size * args.batch_size_v))
                first_flag = False
            else:
                loss_log.append(avg_loss)
                loss_data_log.append(np.mean(batch_u_loss))
                loss_pde_log.append(np.mean(batch_f_loss))
                loss_reg_log.append(np.mean(batch_r_loss))
                
            current_lr = optimizer.param_groups[0]['lr']
            
            # 使用 pbar.set_postfix 将信息追加到进度条后方
            pbar.set_postfix({
                'Total': f"{avg_loss:.4e}",
                'PDE': f"{loss_pde_log[-1]:.4e}",
                'Data': f"{loss_data_log[-1]:.4e}",
                'LR': f"{current_lr:.2e}"
            })

            # 学习率调度
            if i <= args.warmup_epochs:
                warmup_scheduler.step(i)
            else:
                scheduler.step(avg_loss)
                
            # ==========================================
            # 5. 验证环节
            # ==========================================
            if i % args.validate_every == 0:
                model.eval()
                batch_u_loss, batch_f_loss = [], [] 
                
                with torch.no_grad(): 
                    for vel_batch, UU0_batch, labels_batch in dataloader['valid']:
                        vel_batch = vel_batch.to(device)
                        UU0_batch = UU0_batch.to(device)
                        labels_batch = labels_batch.to(device)
                        
                        for batch in dataloader['valid_y']:
                            y_batch = batch[0].to(device)
                            y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)
                            
                            _, loss_f_valid, loss_u_valid, _ = model.loss(
                                vel_batch, y_batch, UU0_batch, labels_batch, 
                                a, b, c, data_norm_coe, pde_norm_coe
                            )
                            batch_u_loss.append(loss_u_valid.item())
                            batch_f_loss.append(loss_f_valid.item())
                            
                valid_u_loss.append(np.mean(batch_u_loss))
                valid_f_loss.append(np.mean(batch_f_loss)) 

            # ==========================================
            # 6. 可视化与绘图
            # ==========================================
            if i % args.save_fig_every == 0:
                vel_pred, UU0_pred, labels_pred = plot_data["vel_pred"], plot_data["UU0_pred"], plot_data["labels_pred"]
                vel_test, UU0_test, labels_test = plot_data["vel_test"], plot_data["UU0_test"], plot_data["labels_test"]

                marmousi_data = ext_val_sets['Marmousi']
                v_m_test, u0_m_test, lab_m_test = marmousi_data["plot_data"]["v_test"], marmousi_data["plot_data"]["u0_test"], marmousi_data["plot_data"]["lab_test"]
                dataloader_m_y_full = marmousi_data["loader"]

                plot_loss(i, args.save_doc, loss_log, loss_data_log, loss_pde_log, valid_u_loss, valid_f_loss)
                
                # 新增：防止绘图函数内部构建冗余计算图导致 OOM
                with torch.no_grad(): 
                    if i % (args.save_fig_every * 20) == 0 and i > 0 and args.if_finetune:
                        test_plot(args, model, fno, i, dataloader_m_y_full, v_m_test, u0_m_test, lab_m_test, 'FT_Marmousi', if_fine_tune=True)
                    
                    test_plot(args, model, fno, i, dataloader["pred"], vel_pred, UU0_pred, labels_pred, 'valid_without_fine_tune', if_fine_tune=False)
                    test_plot(args, model, fno, i, dataloader["test"], vel_test, UU0_test, labels_test, 'train', if_fine_tune=False)
                    test_plot(args, model, fno, i, dataloader_m_y_full, v_m_test, u0_m_test, lab_m_test, 'Marmousi', if_fine_tune=False)
                    plot_sinlge(model, args, 6, vel_test, UU0_test, labels_test)

                torch.cuda.empty_cache()
                gc.collect()

            # ==========================================
            # 7. 模型保存
            # ==========================================
            if i % args.save_model_every == 0 and i > 0:
                pbar.write(f'>>> Epoch {i} | 保存 Checkpoint: Total Loss {loss_log[-1]:.4e} | PDE Loss {loss_pde_log[-1]:.4e}')
                
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }
                
                torch.save(checkpoint, os.path.join(args.save_doc, f'{args.filename}_PI_model_{i}epoch_weights.pth'))
                np.save(os.path.join(args.save_doc, 'loss_log.npy'), loss_log)
                np.save(os.path.join(args.save_doc, 'loss_data_log.npy'), loss_data_log)
                np.save(os.path.join(args.save_doc, 'loss_pde_log.npy'), loss_pde_log)

    except Exception as e:
        print(f"训练过程中断出错: {e}")
        raise
    finally:
        # ==========================================
        # 8. 资源释放与清理
        # ==========================================
        if 'dataloader' in locals() and hasattr(dataloader, '_workers'):
            for worker in dataloader._workers:
                if worker.is_alive():
                    worker.terminate()
                    
        if 'dataloader_y' in locals() and hasattr(dataloader_y, '_workers'):
            for worker in dataloader_y._workers:
                if worker.is_alive():
                    worker.terminate()
                    
        torch.cuda.empty_cache()