from config import *
from Labconfig import *
from model.utils import *

def plot_sinlge(model, args, times, vel_pred, UU0_pred, labels_pred):
    L = args.LD
    L1 = args.LD * times
    Nz = args.nz * times
    Nx = args.nx * times
    
    nx = args.nx
    nz = args.nz
    spatial_step = 40.
    device = args.device
    tag_num = args.nz * times - L1 * 2


    x_coords = torch.arange(0, nx * times)
    z_coords = torch.arange(0, nz * times)
    NN = x_coords.shape[0] * z_coords.shape[0]

    grid_z, grid_x = torch.meshgrid(z_coords, x_coords)
    points = torch.stack([grid_z.flatten(), grid_x.flatten()], dim=1)
    y_pred = points.float() * spatial_step / times
    
    dataset_test = TensorDataset(y_pred)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
    
    model.eval()
    u_test = []
    with torch.no_grad():
        for batch in dataloader_test:
            y_pred_batch = batch[0].to(device)
            y_batch = y_pred_batch.unsqueeze(0)
            # print('y', y_batch.shape)
            u_pred_batch = model(vel_pred.to(device), y_batch, UU0_pred.to(device)).squeeze(0)
            u_test.append(u_pred_batch.detach().cpu().numpy())

    # 保存测试结果（与原逻辑一致）
    U_pred_test = np.vstack(u_test)
    U_pred_test = U_pred_test.reshape(Nz, Nx, 2)
    U_pred_real_test = U_pred_test[L1:-L1, L1:-L1, 0]
    U_pred_imag_test = U_pred_test[L1:-L1, L1:-L1, 1]
    
    y_pred_np = y_pred.detach().cpu().numpy()
    labels_pred_np = labels_pred.detach().cpu().numpy()
    U_test = labels_pred_np[0,:,:,:]
    U_test_real = U_test[0, L:-L, L:-L][:, :]
    U_test_imag = U_test[1, L:-L, L:-L][:, :]
    
    x_test = np.linspace(0, 2800, num=70, endpoint=True)  # 70个点，对应实际距离0-2800
    # 2. 为280长度的数据（U_pred_real_test）生成横坐标：0-2800，间隔10
    x_pred = np.linspace(0, 2800, num=tag_num, endpoint=True)  # 280个点，对应实际距离0-2800
    
    # ===================== 绘图逻辑修改 =====================
    figure1, ax1 = plt.subplots(figsize=(10, 6))  # 注意：plt.figure() 返回单个对象，plt.subplots() 返回 (fig, ax)
    # 绘制70点的参考数据（U_test_real）
    ax1.plot(x_test, U_test_real[:, 70//2], label='Reference Real (70 points)', linewidth=2, color='#ff7f0e')
    # 绘制280点的预测数据（U_pred_real_test）
    ax1.plot(x_pred, U_pred_real_test[:, tag_num//2], label=f'Predicted Real ({tag_num} points)', linewidth=1.5, color='#1f77b4')
    
    # 图表美化（强制统一x轴尺度为0-2800）
    ax1.set_title('Real Part Comparison at Mid X-Line (0-2800 Distance)', fontsize=12)
    ax1.set_xlabel('Distance (m)', fontsize=10)  # x轴改为实际距离（米）
    ax1.set_ylabel('Amplitude', fontsize=10)
    ax1.set_xlim(0, 2800)  # 强制x轴范围为0-2800，保证尺度统一
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # 保存图片
    plt.tight_layout()  # 防止标签被截断
    plt.savefig(args.save_doc + '/singleline.png', bbox_inches='tight')
    plt.close(figure1)  # 关闭画布，释放内存
    
def test_plot(args, model, fno, i, dataloader_y, vel, UU0, labels, filename, if_fine_tune, loc=2):
    if if_fine_tune:
        model = fine_tuning(args, model, fno, dataloader_y, vel, UU0, labels)
    model.eval()
    device = args.device
    L = args.LD
    # filename = args.filename
    u_pred = []
    
    with torch.no_grad():
        for batch in dataloader_y:
            y_batch = batch[0].to(device)
            y_batch = y_batch.unsqueeze(0)
            # 单卡直接调用model（无需.module）
            u_batch = model(vel.to(device), y_batch, UU0.to(device)).squeeze(0)
            u_pred.append(u_batch.detach().cpu().numpy())

    # 处理预测结果（与原逻辑一致）
    U_pred = np.vstack(u_pred)
    U_pred = U_pred.reshape(args.nz, args.nx, 2)
    U_pred_real = U_pred[L:-L, L:-L, 0]
    U_pred_imag = U_pred[L:-L, L:-L, 1]
    
    # 处理标签数据（与原逻辑一致）
    labels_np = labels.detach().cpu().numpy()
    U_ref = labels_np[0,:,:,:]
    U_ref_real = U_ref[0, L:-L, L:-L][:, :]
    U_ref_imag = U_ref[1, L:-L, L:-L][:, :]

    # 计算误差（与原逻辑一致）
    Umaxr, Uminr = np.max(U_ref_real), np.min(U_ref_real)
    Umaxi, Umini = np.max(U_ref_imag), np.min(U_ref_imag)
    Rr = np.maximum(np.abs(Umaxr), np.abs(Uminr))
    Ri = np.maximum(np.abs(Umaxi), np.abs(Umini))
    err_abs_real = U_ref_real - U_pred_real
    err_abs_imag = U_ref_imag - U_pred_imag
    eRr = np.max(np.abs(err_abs_real))
    eRi = np.max(np.abs(err_abs_imag))
    metrics_real = calculate_regression_metrics(U_pred_real, U_ref_real)

    # 2. 虚部指标
    metrics_imag = calculate_regression_metrics(U_pred_imag, U_ref_imag)
    print("="*60)
    print(f"模型预测性能指标汇总：{filename}")
    print("="*60)
    # 实部
    print(f"\n【实部】")
    print(f"MSE: {metrics_real['mse']:.6f}")
    print(f"MAE: {metrics_real['mae']:.6f}")
    print(f"R²:  {metrics_real['r2']:.6f}")
    # 虚部
    print(f"\n【虚部】")
    print(f"MSE: {metrics_imag['mse']:.6f}")
    print(f"MAE: {metrics_imag['mae']:.6f}")
    print(f"R²:  {metrics_imag['r2']:.6f}")
    # 绘制预测结果（与原逻辑一致）
    fig1 = plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(U_ref_real, aspect='auto', cmap='seismic')
    plt.title('ref real')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Z')

    plt.subplot(2, 2, 2)
    plt.imshow(U_pred_real, aspect='auto', cmap='seismic')
    plt.title(f'pred real epoch {i}')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Z')

    plt.subplot(2, 2, 3)
    plt.imshow(U_ref_imag, aspect='auto', cmap='seismic')
    plt.title('ref imag')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Z')

    plt.subplot(2, 2, 4)
    plt.imshow(U_pred_imag, aspect='auto', cmap='seismic')
    plt.title(f'pred imag epoch {i}')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.savefig(args.save_doc + '/epoch_plot_' + f'{filename}.png')
    plt.close()

    # 绘制误差图（与原逻辑一致）
    fig_e = plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(err_abs_real, aspect='auto', cmap='bwr')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title(f'epoch {i} valid real error')
    plt.colorbar()
    plt.clim(-eRr, eRr)

    plt.subplot(1, 2, 2)
    plt.imshow(err_abs_imag, aspect='auto', cmap='bwr')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title(f'epoch {i} valid imag error')
    plt.colorbar()
    plt.clim(-eRi, eRi)
    plt.savefig(args.save_doc + '/error_' + f'{filename}.png')
    plt.close()
    
def fine_tuning(args, model0, fno, dataloader_y, vel, UU0, labels):
    device = args.device
    model_ft = copy.deepcopy(model0).to(device)
    NIter = args.ft_NIter
    lr = args.ft_lr
    a = args.ft_a
    b = args.ft_b
    c = args.ft_c
    nz = args.nz
    nx = args.nx
    Lpml = args.Lpml
    
    first_flag = True
    pde_norm_coe = 1.
    data_norm_coe = 1.
    
    optimizer = optim.Adam(model_ft.parameters(), lr=lr, weight_decay=args.weight_decay)
    
    loss_f_log = []
    loss_u_log = []
    loss_r_log = []
    loss_op_batch = []
    
    # --- 用于追踪最佳模型的变量 ---
    best_loss = float('inf')
    best_model_state = None
    # -----------------------------------

    # --- 新增：记录微调开始时间 ---
    start_time = time.time()
    print(f" 开始在目标速度场上执行域适应微调 (总迭代次数: {NIter})...")
    # ------------------------------

    model_ft.train()
    with torch.no_grad():
        labels_fno = fno(vel.to(device), UU0.to(device)).to(device)
        
    for i in range(NIter):
        batch_loss = []
        for batch in dataloader_y:
            y_batch = batch[0].to(device)
            y_batch = y_batch.unsqueeze(0)
            
            # 计算损失
            loss, loss_f, loss_u, loss_r = model_ft.loss(
                vel.to(device), y_batch, UU0.to(device), labels_fno.to(device), 
                a, b, c, data_norm_coe, pde_norm_coe
            )
            loss_op = c * model_ft.loss_op(model0, vel.to(device), y_batch, UU0.to(device))
            loss = loss + loss_op
            
            # 修复：将除以累加步数后的结果重新赋值给 loss，否则梯度会按原比例回传
            loss = loss / args.accumulation_steps 
            
            # 反向传播
            loss.backward()
            
            if (i + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad() # 更新完后清空梯度
                
            batch_loss.append(loss.item())
            loss_u_log.append(loss_u.item())
            loss_f_log.append(loss_f.item())
            loss_op_batch.append(loss_op.item())
            
        # 动态权重更新
        if first_flag:
            data_norm_coe = np.mean(loss_u_log)/(args.batch_size * args.batch_size_v * 2)
            pde_norm_coe = np.mean(loss_f_log)/(args.batch_size * args.batch_size_v * 2)
            first_flag = False
            
        # 计算当前 Epoch 的平均总损失
        current_epoch_loss = np.mean(batch_loss)
        
        # 评估并保存最佳模型
        if i > 0 and current_epoch_loss < best_loss:
            best_loss = current_epoch_loss
            # 仅保存 state_dict，避免显存泄漏和深拷贝带来的开销
            best_model_state = copy.deepcopy(model_ft.state_dict())
            print(f"微调 {i}/{NIter} --> 发现最佳模型! 当前总损失: {best_loss:.6f}")

        print(f"微调 {i}/{NIter}, PDE损失: {np.mean(loss_f_log)/(args.batch_size * args.batch_size_v * 2):.6f}, 数据损失: {np.mean(loss_u_log)/(args.batch_size * args.batch_size_v * 2):.6f}, 锚定损失:{np.mean(loss_op_batch)/(args.batch_size * args.batch_size_v * 2):.6f}")
        
        loss_f_log = []
        loss_u_log = []
        loss_r_log = []
        loss_op_batch = []

    # --- 新增：计算并格式化总耗时 ---
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    # ------------------------------

    if best_model_state is not None:
        print(f"\n 微调结束。总耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
        print(f"正在加载历史最佳模型权重 (最低损失: {best_loss:.6f}) 用于后续测试与绘图...")
        model_ft.load_state_dict(best_model_state)
    else:
        print("\n 警告：未找到最佳模型（可能是迭代次数过少），将返回最终模型。")
        print(f"总耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    
    return model_ft