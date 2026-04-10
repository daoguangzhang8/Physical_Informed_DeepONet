class Args:
    # ==========================================
    # 1. 路径与文件配置 (Paths & I/O Configuration)
    # ==========================================
    load_path = '/home/sharedata/zdg'         # 数据集加载根目录
    weights_save_path = '/home/sharedata/zdg' # 模型权重保存根目录
    save_doc = 'output'                       # 结果输出文件夹名称
    filename = 'PI_DeepONet_pde'              # 保存的模型前缀名称

    # 训练数据文件名
    # vel_filename = 'velocity_data_70_70_n1.npy'
    # backgroundfield_filename = 'backgroundfield_data_freq5_1source_70_70_n1.npy'
    # wavefield_filename = 'wavefield_data_freq5_5sources_70_70_n1.npy'

    # 自由表面训练数据
    vel_filename = 'freesurface_velocity_freq3to20_5sources_160_180_pml20_n1.npy'
    backgroundfield_filename = 'freesurface_backgroundfield_freq3to20_5sources_160_180_pml20_n1.npy'
    wavefield_filename = 'freesurface_wavefield_freq3to20_5sources_160_180_pml20_n1.npy'
    freq_filename = 'freesurface_freq_used_5sources_160_180_pml20_n1.npy'               # 频率数据 [N_vel]，每个速度模型对应一个频率
    default_freq = 5.0                                                 # 默认频率 (Hz)，当 freq 文件不存在或 freq_batch=None 时使用
    
    # 外部泛化测试集配置 (支持动态扩展)
    ext_val_datasets = {
        # 'Marmousi': {'prefix': 'marmousi_', 'loc_target': 2},
        # 'BP': {'prefix': 'bp_', 'loc_target': 0},  
    }
    
    # ==========================================
    # 2. 硬件与设备配置 (Hardware & Device)
    # ==========================================
    device = 0                                # 单 GPU 模式: 指定使用的 GPU 设备编号

    # ==========================================
    # 2.1 单机多卡并行训练配置 (Single-Machine Multi-GPU)
    # ==========================================
    use_parallel = False                      # 是否启用多 GPU 并行训练 (True: 多GPU | False: 单GPU)
    num_gpus = 2                              # 使用的 GPU 数量
    min_gpu_memory = 23 * 1024                # GPU 最小可用内存 (MB)，低于此值的 GPU 不会被使用

    # ==========================================
    # 3. 学习率调度器参数 (Learning Rate Scheduler)
    # ==========================================
    factor = 0.9                              # 学习率衰减因子 (ReduceLROnPlateau)
    patience = 30                             # 触发衰减的容忍 epoch 数量
    min_lr = 1e-7                             # 允许的最小学习率
    
    # ==========================================
    # 4. 训练控制与状态保存 (Training Control & Checkpoints)
    # ==========================================
    if_load_model = False                     # 是否加载预训练模型权重继续训练
    if_adjust = True                          # 是否在训练过程中动态调整 Loss 权重
    adjust_from = 2000                        # 从第几个 epoch 开始动态调整
    adjust_every = 1000                       # 每隔多少个 epoch 调整一次权重
    adjust_speed = 1.1                        # 权重衰减/增长的速度因子
    save_fig_every = 50                       # 每隔多少个 epoch 保存一次验证/测试可视化图片
    save_model_every = 500                    # 每隔多少个 epoch 保存一次模型权重文件

    # ==========================================
    # 5. 标签来源配置 (Label Source Configuration)
    # ==========================================
    use_fno_as_label = False                  # True: 使用 FNO 预测作为软标签 | False: 使用真实标签
    fno_weights_path = ''                     # FNO 预训练权重路径 (当 use_fno_as_label=True 时需要指定)
    
    # ==========================================
    # 5. 空间点采样模式配置 (Spatial Sampling Mode)
    # ==========================================
    sampling_mode = 'halton'                 # 'full_grid': 全网格采样 | 'halton': Halton 准随机采样
    halton_sample_ratio = 0.5                   # Halton 采样比例（仅在 sampling_mode='halton' 时生效，如 0.2 表示采样 20% 的网格点）

    # ==========================================
    # 5.1 数据集与批处理配置 (Dataset & Dataloader)
    # ==========================================
    nvel_train = 1500                            # 训练所用的速度模型数量
    ny_train = 4900                           # 训练集空间采样点总数
    batch_size = 700                          # Trunk Net 坐标采样批次大小 (num_sample)
    batch_size_v = 30                          # Branch Net 速度场/背景场批次大小 (Batch_v)
    
    valid_rate = 0.1                          # 验证集划分比例
    validate_every = 100                      # 每隔多少个 epoch 执行一次模型验证
    valid_batch_size = 350                    # 验证集坐标采样批次大小
    valid_batch_size_v = 6                    # 验证集速度场批次大小
    accumulation_steps = 1                    # 梯度累加步数 (用于等效增大 batch size，节约显存)

    source_list = [0, 1, 2, 3, 4]                 # 训练数据中包含的震源编号列表 (0-4 共5个震源)  
    # ==========================================
    # 5.2 物理网格间距 (Spatial Grid Spacing)
    # ==========================================
    dh = 40                                     # 空间网格间距 (m)，物理坐标 = 网格索引 * dh

    # ==========================================
    # 6. 物理网格与边界条件 (Physical Grid & PML Boundaries)
    # ==========================================
    nx = 140                                   # 物理模型 x 方向网格数 (不含外延 PML)
    nz = 140                                   # 物理模型 z 方向网格数 (不含外延 PML)
    pml = True                                # 是否启用 PML (Perfectly Matched Layer, 完美匹配层) 吸收边界
    pml_total = 20                            # PML 吸收层的总网格厚度
    pml_crop = 15                              # 训练时裁剪/忽略的 PML 网格数
    pml_active = pml_total - pml_crop         # 剩余参与训练的 PML 网格数

    # 边界类型配置
    # 'full_pml': 四边 PML 吸收边界，原始数据 90×90 → 网络输入 72×72
    # 'free_surface': 顶部自由表面 + 其他三边 PML，原始数据 80×90 → 网络输入 71×72
    boundary_type = 'free_surface'            # 根据实际数据选择
    
    # ==========================================
    # 7. 微调与域适应配置 (Fine-Tuning for Out-of-Distribution)
    # ==========================================
    if_finetune = False                       # 是否在外部复杂地层 (如 Marmousi) 上进行微调评估
    ft_NIter = 1000                             # 微调阶段的迭代步数
    ft_lr = 2e-5                              # 微调阶段的专属学习率
    ft_a = 0.2                                # 微调阶段的数据 Loss 权重
    ft_b = 1                                  # 微调阶段的 PDE Loss 权重
    ft_c = 1                                  # 微调阶段的正则化 Loss 权重
    
    # ==========================================
    # 8. 主训练循环超参数 (Main Training Hyperparameters)
    # ==========================================
    NIter = 2000 + 1                         # 总训练 epoch 数 (+1 确保最后一步记录和保存生效)
    warmup_epochs = 100                       # 学习率热身 (Warmup) 的 epoch 数
    lr = 1 * 1e-4                             # 初始基础学习率
    weight_decay = 1e-4                       # 优化器权重衰减 (L2 正则化)系数
    
    # ==========================================
    # 9. 物理信息损失函数权重 (PINN Loss Coefficients)
    # ==========================================
    a = 1                                     # coefficient of dataloss (数据拟合项权重)
    b = 1                                     # coefficient of pdeloss (PDE 物理残差项权重)
    c = 0                                     # coefficient of regularization loss (额外正则化项权重)
    
    # ==========================================
    # 10. 网络架构与张量形状占位 (Network Architecture Inputs)
    # ==========================================
    in_channels = 2                           # 波场相关输入通道数 (如复数波场的实部、虚部)
    in_channels_vel = 1                       # 速度模型输入通道数 (1个通道代表速度 v)
    input_shape_trunk = (batch_size, in_channels, 1, 2)       # Trunk Net (评估坐标) 的输入形状占位
    input_shape_branch1 = (batch_size, in_channels_vel, nz, nx) # Branch Net 1 (速度场) 输入形状占位
    input_shape_branch2 = (batch_size, in_channels, nz, nx)     # Branch Net 2 (背景场/震源) 输入形状占位





    