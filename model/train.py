
from Labconfig import *
from model.utils import *
from model.dataloader import *
from model.PI_DeepOnet import Pi_DeepONet
from model.FNO import FNO
from model.ploting import *


def train(args):
    try:
        # 单卡训练：直接使用当前设备（默认GPU，无GPU则用CPU）
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        
        dataloader, plot_data = prepare_training_dataloaders(args, device)
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
        
        print(device)
        model = Pi_DeepONet(args).to(device)
        print(model.device)
        
        fno = FNO(args).to(device)
        fno.load_state_dict(torch.load('FNO_bad_PI_model_200epoch_weights.pth')['model_state_dict'])
        
        model._init_weights()  
        
        print(f"模型总参数数量：{count_parameters(model)}")
        
        # 优化器和调度器（与原逻辑一致）
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.factor, patience=args.patience, min_lr=args.min_lr
        )

        warmup_scheduler = WarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=args.warmup_epochs,  # 前50轮热身
            base_lr=args.lr,
            warmup_start_lr=args.lr/10.,  # 起始学习率（base_lr的1/10）
            warmup_strategy="linear",  # 线性增长
            after_scheduler=None  # 衔接StepLR
        )
        # 损失记录（与原逻辑一致）
        loss_log = []
        loss_pde_log = []
        loss_data_log = []
        loss_reg_log = []
        
        valid_u_loss = []
        valid_f_loss = []
        is_lr_changed = False # 状态标志，防止重复更改

        a = args.a
        b = args.b
        c = args.c
        nz = args.nz
        nx = args.nx
        Lpml = args.Lpml
        
        first_flag = True
        pde_norm_coe = 1.
        data_norm_coe = 1.
        # 训练循环
        optimizer.zero_grad()
        for i in range(args.NIter):
            if args.if_adjust and i > args.adjust_from and (i - args.adjust_from) % args.adjust_every == 0:
                # b = 1
                decay_times = i // args.adjust_every 
                a = a * args.adjust_speed ** (- decay_times)
                a = max(a, 2e-1)
                b = 1
                c = 0
            # print(f"迭代次数: {i}")
            model.train()
            batch_loss = []
            batch_u_loss = []
            batch_f_loss = []
            batch_r_loss = []
            # 单卡训练：直接迭代数据加载器，无需分布式采样器同步
            optimizer.zero_grad()
            for vel_batch, UU0_batch, labels_batch in dataloader['train']:
                # 数据移到设备
                vel_batch = vel_batch.to(device)
                UU0_batch = UU0_batch.to(device)
                # labels_batch = labels_batch.to(device)
                with torch.no_grad():
                    labels_batch = fno(vel_batch, UU0_batch).to(device)
                for batch in dataloader['train_y']:
                    y_batch = batch[0].to(device)
                    y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)
                    # 计算损失：单卡直接调用model.loss（无需.module）
                    loss, loss_f, loss_u, loss_r = model.loss(
                        vel_batch, y_batch, UU0_batch, labels_batch, 
                        a, b, c,
                        data_norm_coe, pde_norm_coe
                    )
                    loss = loss / args.accumulation_steps
                    # 反向传播
                    # optimizer.zero_grad()
                    loss.backward()
                    
                    if (i + 1) % args.accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad() # 更新完后清空梯度
                        
                        # 记录损失
                    batch_loss.append(loss.item())
                    batch_u_loss.append(loss_u.item())
                    batch_f_loss.append(loss_f.item())
            if first_flag:
                data_norm_coe = np.mean(batch_u_loss)
                pde_norm_coe = np.mean(batch_f_loss)
                avg_loss = np.mean(batch_loss) if batch_loss else 0
                loss_log.append(a+b)
                loss_data_log.append(1.)
                loss_pde_log.append(1.)
                loss_reg_log.append(np.mean(batch_r_loss)/(args.batch_size * args.batch_size_v))
                first_flag = False
            else:
                avg_loss = np.mean(batch_loss) if batch_loss else 0
                # avg_loss/(args.batch_size * args.batch_size_v * 2)
                loss_log.append(avg_loss)
                loss_data_log.append(np.mean(batch_u_loss))
                loss_pde_log.append(np.mean(batch_f_loss))
                loss_reg_log.append(np.mean(batch_r_loss))
            # 计算平均损失
            # avg_loss = np.mean(batch_loss) if batch_loss else 0
            # loss_log.append(avg_loss)
            # loss_data_log.append(np.mean(loss_u_log)/(args.batch_size * args.batch_size_v * 2))
            # loss_pde_log.append(np.mean(loss_f_log)/(args.batch_size * args.batch_size_v * 2))
            # loss_reg_log.append(np.mean(loss_r_log)/(args.batch_size * args.batch_size_v))
            current_lr = optimizer.param_groups[0]['lr']
            print(f"迭代 {i}/{args.NIter}, 总损失: {avg_loss:.6f}, PDE损失: {loss_pde_log[-1]}, 数据损失: {loss_data_log[-1]}, LR: {current_lr}")

            # 清空临时损失记录
            batch_u_loss = []
            batch_f_loss = []
            batch_r_loss = []

            # 学习率调度
            if i <= args.warmup_epochs:
                warmup_scheduler.step(i)
            else:
                scheduler.step(avg_loss)
            if i % args.validate_every == 0:
                
                model.eval()
                for vel_batch, UU0_batch, labels_batch in dataloader['valid']:
                    # 数据移到设备
                    vel_batch = vel_batch.to(device)
                    UU0_batch = UU0_batch.to(device)
                    labels_batch = labels_batch.to(device)
                    for batch in dataloader['valid_y']:
                        y_batch = batch[0].to(device)
                        y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)
                        loss_valid, loss_f_valid, loss_u_valid, loss_r_valid = model.loss(
                            vel_batch, y_batch, UU0_batch, labels_batch, 
                            a, b, c,
                            data_norm_coe, pde_norm_coe
                        )
                        batch_loss.append(loss_valid.item())
                        batch_u_loss.append(loss_u_valid.item())
                        batch_f_loss.append(loss_f_valid.item())
                # (args.valid_batch_size * args.valid_batch_size_v * 2)
                valid_u_loss.append(np.mean(batch_u_loss))
                valid_f_loss.append(np.mean(batch_u_loss))
                
                batch_u_loss = []
                batch_f_loss = []

            L = args.LD
            # 每10次迭代保存结果（与原逻辑一致，移除分布式相关）
            if i % args.save_fig_every == 0:
                
                vel_pred = plot_data["vel_pred"]
                UU0_pred = plot_data["UU0_pred"]
                labels_pred = plot_data["labels_pred"]
                
                vel_test = plot_data["vel_test"]
                UU0_test = plot_data["UU0_test"]
                labels_test = plot_data["labels_test"]

                # 2. 从外部验证集字典中显式提取 Marmousi 数据
                # (假设您在前面使用了 ext_val_sets['Marmousi'] 保存了加载的数据)
                marmousi_data = ext_val_sets['Marmousi']
                v_m_test = marmousi_data["plot_data"]["v_test"]
                u0_m_test = marmousi_data["plot_data"]["u0_test"]
                lab_m_test = marmousi_data["plot_data"]["lab_test"]
                dataloader_m_y_full = marmousi_data["loader"]
                
                train_loss_lenth = len(loss_data_log)
                valid_loss_lenth = len(valid_u_loss)
                train_x_axis = np.linspace(0, i, num=train_loss_lenth, endpoint=True)
                valid_x_axis = np.linspace(0, i, num=valid_loss_lenth, endpoint=True)
                # 绘制并保存损失曲线（与原逻辑一致）
                plt.figure()
                plt.plot(train_x_axis,loss_data_log, label="Training Data Loss")
                plt.plot(valid_x_axis,valid_u_loss, label="Valid Data Loss")
                plt.yscale('log')
                plt.legend()
                plt.title(f'epoch {i} Data Loss')
                plt.savefig(args.save_doc + '/Dataloss_curve.png')
                plt.close()

                plt.figure()
                plt.plot(train_x_axis, loss_pde_log, label="Training PDE Loss")
                plt.plot(valid_x_axis, valid_f_loss, label="Valid PDE Loss")
                plt.yscale('log')
                plt.legend()
                plt.title(f'epoch {i} PDE Loss')
                plt.savefig(args.save_doc + '/PDEloss_curve.png')
                plt.close()

                plt.figure()
                plt.plot(loss_log, label="Total Loss")
                plt.yscale('log')
                plt.legend()
                plt.savefig(args.save_doc + '/loss_curve.png')
                plt.close()
                if i % (args.save_fig_every * 20) == 0 and i > 0 and args.if_finetune:
                    test_plot(args, model, fno, i, dataloader_m_y_full, v_m_test, u0_m_test, lab_m_test,'FT_Marmousi', True)
                
                test_plot(args, model, fno, i, dataloader["pred"], vel_pred, UU0_pred, labels_pred,'valid_without_fine_tune', False)
                
                test_plot(args, model, fno, i, dataloader["test"], vel_test, UU0_test, labels_test,'train', False)
                
                test_plot(args, model, fno, i, dataloader_m_y_full, v_m_test, u0_m_test, lab_m_test,'Marmousi', False)
                # v_m_test, u_m_test, u0_m_test, y_m_test, lab_m_test
                plot_sinlge(model, args, 6, vel_test, UU0_test, labels_test)

                # 清理内存
                torch.cuda.empty_cache()
                gc.collect()

            # 训练结束：保存最终结果（与原逻辑一致，移除分布式相关）
            if i % args.save_model_every == 0:

                print(f'总损失: {loss_log[-1]}')
                print(f'PDE损失: {loss_pde_log[-1]}')
                print(f'数据损失: {loss_data_log[-1]}')
                
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }
                
                torch.save(checkpoint,f'{args.filename}_PI_model_{i}epoch_weights.pth')
                
                np.save(args.save_doc + '/loss_log.npy', loss_log)
                np.save(args.save_doc + '/loss_data_log.npy', loss_data_log)
                np.save(args.save_doc + '/loss_pde_log.npy', loss_pde_log)

    except Exception as e:
        print(f"训练出错: {e}")
        raise
    finally:
        # 清理数据加载器子进程
        if 'dataloader' in locals() and hasattr(dataloader, '_workers'):
            for worker in dataloader._workers:
                if worker.is_alive():
                    worker.terminate()
        if 'dataloader_y' in locals() and hasattr(dataloader_y, '_workers'):
            for worker in dataloader_y._workers:
                if worker.is_alive():
                    worker.terminate()
        # 清理GPU缓存
        torch.cuda.empty_cache()