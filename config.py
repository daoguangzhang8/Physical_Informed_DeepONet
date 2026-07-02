class Args:
    # ==========================================
    # 1. 路径与文件配置 (Paths & I/O)
    # ==========================================
    load_path = '/home/sharedata/zdg'         # 数据集加载根目录
    weights_save_path = '/home/sharedata/zdg' # 模型权重保存根目录
    save_doc = 'output_single_source_multivel_multifreq'  # 单震源、多速度模型、多频训练输出
    filename = 'PI_DeepONet_pde'              # 保存的模型前缀名称

    # 训练数据文件名
    # vel_filename = 'velocity_data_70_70_n1.npy'
    # backgroundfield_filename = 'backgroundfield_data_freq5_1source_70_70_n1.npy'
    # wavefield_filename = 'wavefield_data_freq5_5sources_70_70_n1.npy'

    # 自由表面训练数据
    # vel_filename = 'freesurface_velocity_freq3to20_5sources_160_180_pml20_n1.npy'
    # backgroundfield_filename = 'freesurface_backgroundfield_freq3to20_5sources_160_180_pml20_n1.npy'
    # wavefield_filename = 'freesurface_wavefield_freq3to20_5sources_160_180_pml20_n1.npy'
    # freq_filename = 'freesurface_freq_used_5sources_160_180_pml20_n1.npy'

    # 合并训练数据 (freq3to11 + freq12to18 + freq18to25)
    vel_filename = 'multifreq_merged1_ds2/freesurface_full_5sources_velocity.npy'
    backgroundfield_filename = 'multifreq_merged1_ds2/freesurface_full_5sources_background.npy'
    wavefield_filename = 'multifreq_merged1_ds2/freesurface_full_5sources_wavefield.npy'
    freq_filename = 'multifreq_merged1_ds2/freesurface_full_5sources_freq_used.npy'  # [N_vel, N_freq]
    category_filter = ['flat_layers', 'flat_layers_2', 'fold', 'fault', 'salt']  # 单震源多速度模型训练使用全部已有类别
    frequency_filter = None                    # None=使用每个速度模型包含的全部频率
    velocity_mode = 'multi'                    # 'single': 使用下方固定速度模型索引 | 'multi': 使用各类别数量随机划分
    train_velocity_indices = [0]               # velocity_mode='single' 时使用的训练速度模型索引
    valid_velocity_indices = [27]              # velocity_mode='single' 时使用的验证速度模型索引
    default_freq = 5.0                                                 # 默认频率 (Hz)，当 freq 文件不存在或 freq_batch=None 时使用

    # 外部泛化测试集配置 (支持动态扩展)
    ext_val_datasets = {
        # 'Marmousi': {'prefix': 'marmousi_', 'loc_target': 2},
        # 'BP': {'prefix': 'bp_', 'loc_target': 0},
    }

    # 标签来源配置
    use_fno_as_label = False                  # True: 使用 FNO 预测作为软标签 | False: 使用真实标签
    fno_weights_path = ''                     # FNO 预训练权重路径 (当 use_fno_as_label=True 时需要指定)

    # ==========================================
    # 2. 硬件与设备配置 (Hardware & Device)
    # ==========================================
    device = 2                                 # 单 GPU 模式时的默认设备；DDP 会自动映射可用 GPU
    use_parallel = True                        # 多速度模型训练默认启用多 GPU
    num_gpus = 2                               # 当前默认 2 卡；8/16 卡运行时改为对应数量
    min_gpu_memory = 20 * 1024                # GPU 最小可用内存 (MB)，低于此值的 GPU 不会被使用
    ddp_num_workers = 2                       # DDP 每个 rank 的 DataLoader worker 数；16 卡时避免进程过多

    # ==========================================
    # 3. 物理网格与边界条件 (Physical Grid & PML)
    # ==========================================
    dh = 20                                     # 空间网格间距 (m)，物理坐标 = 网格索引 * dh
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
    n_freq_ranges = 3                          # 合并数据来源的频段数量 (单频数据设为1)

    # ==========================================
    # 4. 数据集与采样 (Dataset & Sampling)
    # ==========================================
    # 每类抽取的基础速度模型数；dataloader 会自动展开为 n_freq × n_source 训练样本。
    nvel_train_per_category = {
        'flat_layers': 400,
        'flat_layers_2': 200,
        'fold': 450,
        'fault': 450,
        'salt': 0,
    }
    nvel_valid_per_category = {
        'flat_layers': 10,
        'flat_layers_2': 10,
        'fold': 10,
        'fault': 10,
        'salt': 0,
    }
    nvel_train = 1500                           # 多速度模型训练所用基础速度模型数量
    source_list = [2]                           # 单震源训练：震源位置 2
    valid_source_list = [2]                     # 单震源验证：震源位置 2
    test_source_list = [2]                      # 单震源测试/绘图：震源位置 2
    valid_random_one_freq = False               # False: valid 覆盖每个速度模型的全部频段，分类曲线更稳定
    valid_random_one_source = True              # 验证模型随机抽取配置中的一个震源

    # 空间点采样模式
    sampling_mode = 'halton'                  # 'full_grid': 全网格采样 | 'halton': Halton 准随机采样
    halton_sample_ratio = 0.5                   # Halton 采样比例（仅在 sampling_mode='halton' 时生效，如 0.2 表示采样 20% 的网格点）

    # 批处理配置
    batch_size = 1600                          # Trunk Net 坐标采样批次大小 (num_sample)
    batch_size_v = 64                          # 多速度模型训练的 Branch Net 批次大小
    ny_train = int(nz * nx * halton_sample_ratio)  # 训练集空间采样点总数 (由网格尺寸和采样比例自动计算)
    accumulation_steps = 2                    # 梯度累加步数 (用于等效增大 batch size，节约显存)

    # 验证集
    valid_rate = 0.1                          # 验证集划分比例
    valid_batch_size = 350                    # 验证集坐标采样批次大小
    valid_batch_size_v = 6                    # 验证集速度场批次大小

    # ==========================================
    # 5. 训练超参数 (Training Hyperparameters)
    # ==========================================
    NIter = 5000 + 1                          # 简单 smoke test，先快速检查新结构可训练
    lr = 1 * 1e-4                             # 初始基础学习率
    weight_decay = 1e-4                       # 优化器权重衰减

    # DDP 与单卡结果对齐配置
    ddp_scale_lr = False                       # True: DDP 学习率按 GPU 数线性放大 | False: 保持与单卡相同
    ddp_split_batch_size_v = True              # True: DDP 每卡 batch_size_v=batch_size_v/num_gpus，使全局 batch 接近单卡

    # 学习率调度器
    scheduler_type = 'cosine'               # 'plateau': ReduceLROnPlateau | 'cosine': CosineAnnealingWarmRestarts
    
    use_warmup = True                        # 是否启用 warmup 预热
    warmup_epochs = 100                       # 学习率热身 (Warmup) 的 epoch 数

    # ReduceLROnPlateau 参数
    factor = 0.9                              # 学习率衰减因子
    patience = 50                             # 触发衰减的容忍 epoch 数量
    min_lr = 1e-5                             # 允许的最小学习率

    # CosineAnnealingWarmRestarts 参数
    cosine_T_0 = 701                        # 首个周期的 epoch 长度
    cosine_T_mult = 2                         # 后续周期倍增系数 (T_0, T_0*2, T_0*4, ...)
    cosine_eta_min = 1e-5                     # 余弦退火最低学习率

    # ==========================================
    # 6. 损失函数 (Loss Function)
    # ==========================================
    a = 1                                     # 数据拟合项 (Data Loss) 权重
    b = 1                                     # PDE 物理残差项 (PDE Loss) 权重，保持正常值
    c = 0                                     # 正则化项 (Regularization Loss) 权重
    d = 1                                   # 包络损失项 (Envelope Loss) 权重（MSE 形式，仅在标签点上计算）

    # PDE 权重平滑增长: a 保持不变，b 从 pde_start_weight 余弦增长到配置值 b
    use_pde_weight_ramp = False
    pde_start_weight = 0.0
    pde_ramp_epochs = 5000

    # Data Loss 固定频率加权: 高频权重更大，从训练开始保持不变；验证/外部测试不加权。
    use_frequency_data_weight = False
    frequency_data_weight_start_hz = 12.0
    frequency_data_weight_end_hz = 25.0
    frequency_data_weight_max = 1.8

    # Data Loss 难点学习: 在每个 Sobol/坐标 batch 内提高高误差空间点权重。
    # 权重按每个速度模型归一化到均值 1，只改变梯度关注位置，不放大整体 DataLoss 尺度。
    use_hard_data_weight = False
    hard_data_start_epoch = 300
    hard_data_ramp_epochs = 2000
    hard_data_gamma = 0.5
    hard_data_max_weight = 2.0

    # LPIPS 感知损失: 额外使用规则图像网格计算结构相似性。
    # lpips_grid_size=0 时使用完整 nz×nx 网格；>0 时使用 grid_size×grid_size 低分辨率网格。
    use_lpips_loss = False
    lpips_weight = 1
    lpips_net = 'alex'
    lpips_grid_size = 0
    lpips_start_epoch = 0
    lpips_interval = 1

    # 旧版动态权重调整，仅在 use_pde_weight_ramp=False 时生效
    if_adjust = False                          # 是否在训练过程中动态调整 Loss 权重
    adjust_from = 2000                        # 从第几个 epoch 开始动态调整
    adjust_every = 1000                       # 每隔多少个 epoch 调整一次权重
    adjust_speed = 1.1                        # 权重衰减/增长的速度因子

    # ==========================================
    # 7. 训练控制与保存 (Training Control & Checkpoints)
    # ==========================================
    if_load_model = False                     # 是否加载预训练模型权重继续训练
    validate_every = 200                      # 每隔多少个 epoch 执行一次模型验证
    save_fig_every = 250                       # 每隔多少个 epoch 保存一次验证/测试可视化图片
    save_model_every = 500                    # 每隔多少个 epoch 保存一次模型权重文件

    # ==========================================
    # 8. 微调与域适应 (Fine-Tuning)
    # ==========================================
    if_finetune = False                       # 是否在外部复杂地层 (如 Marmousi) 上进行微调评估
    ft_NIter = 1000                             # 微调阶段的迭代步数
    ft_lr = 2e-5                              # 微调阶段的专属学习率
    ft_a = 0.2                                # 微调阶段的数据 Loss 权重
    ft_b = 1                                  # 微调阶段的 PDE Loss 权重
    ft_c = 1                                  # 微调阶段的正则化 Loss 权重

    # ==========================================
    # 9. Positional Encoding (位置编码)
    # ==========================================
    pe_max_scale = 12.0                          # PE 最高频率尺度 (原值=6.0; 建议扫描: 8.0, 10.0, 12.0)
    use_kpe = False                              # 关闭残差波数 PE；保留兼容旧权重的 kpe_alpha 参数
    use_trunk_freq_encoding = True               # 将频率显式编码后拼接进 Trunk 坐标输入
    trunk_freq_embed_dim = 8                     # 频率编码输出维度；Trunk 输入维度=16+该值
    trunk_freq_num_bands = 3                     # 频率 Fourier band 数: [f, sin/cos(1f), sin/cos(2f), ...]
    trunk_freq_norm_hz = 25.0                    # 频率归一化参考值，覆盖当前 3-25Hz 训练范围

    # ==========================================
    # 10. 网络架构 (Network Architecture)
    # ==========================================
    in_channels = 2                           # 波场相关输入通道数 (如复数波场的实部、虚部)
    in_channels_vel = 1                       # 速度模型输入通道数 (1个通道代表速度 v)
    branch1_modes = 32                       # 速度模型 FNO modes；原值 12，增强速度模型泛化
    branch1_width = 32                       # 速度模型 FNO width；原值 32
    branch2_type = 'fno'                  # 'hybrid': FNO(UU0)+CNN(UU0) 频率门控融合 | 'fno' | 'resnet' | 'conv'
    branch2_modes = 32                       # branch2_type='fno' 时 UU0 FNO modes
    branch2_width = 32                       # branch2_type='fno' 时 UU0 FNO width
    branch2_global_modes = 20                # UU0 全局 FNO 分支 modes
    branch2_global_width = 32                # UU0 全局 FNO 分支 width
    branch2_local_type = 'conv'              # hybrid 的局部分支: 'conv' | 'resnet'
    branch2_freq_gate_norm_hz = 25.0         # 频率 gate 的归一化参考频率
    input_shape_trunk = (batch_size, in_channels, 1, 2)       # Trunk Net (评估坐标) 的输入形状占位
    input_shape_branch1 = (batch_size, in_channels_vel, nz, nx) # Branch Net 1 (速度场) 输入形状占位
    input_shape_branch2 = (batch_size, in_channels, nz, nx)     # Branch Net 2 (背景场/震源) 输入形状占位

    # ==========================================
    # 11. Sobol 采样配置 (Sobol Sampling)
    # ==========================================
    sampling_strategy = 'sobol'                 # 'original': 双层循环+Halton | 'sobol': 分块连续Sobol
    sobol_points_per_step = 2250                # Sobol 模式: 每次参数更新使用的数据监督点数
    sobol_steps_per_velocity_batch = 3          # Sobol 模式: 每个 velocity batch 的参数更新次数
    sobol_points_per_epoch = 1600               # 兼容旧配置；未设置 sobol_points_per_step 时作为兜底
    valid_sobol_points = 4096                   # 固定验证网格点数；所有类别复用同一组 Sobol 点
    sobol_seed = 2026                           # 训练序列种子；DDP 每个 rank 自动使用不同偏移

    # ==========================================
    # 12. 三阶段渐进训练 (Staged Curriculum Training)
    # ==========================================
    staged_training = False                   # 总开关，False 则使用原始单阶段训练

    # 每阶段使用独立数据集，文件名通过 freq_range 替换基础文件名中的 'freq3to20' 得到
    # 例如基础文件名含 'freq3to20' → Stage 0 替换为 'freq3to11'
    stages = [
        {
            'name': 'low_freq',
            'freq_range': '3to11',               # 替换基础文件名中的 'freq3to20'
            'freq_min': 3.0, 'freq_max': 11.0,   # 信息标签，用于日志打印
            'NIter': 6001,
            'lr': 2e-4,                           # 从头训练，完整 LR
            'warmup_epochs': 100,
            'a': 1, 'b': 1, 'c': 0,
            'replay_stages': [],
            'replay_ratio': 0.2,                  # replay 数据保留比例 (1.0=全部, 0.5=随机抽取50%)
            'data_dir': '/home/sharedata/zdg/multifreq_selected/freq_3to11',
        },
        {
            'name': 'mid_freq',
            'freq_range': '12to18',
            'freq_min': 12.0, 'freq_max': 18.0,
            'NIter': 2001,
            'lr': 1e-4,                           # 课程学习，适度降低
            'warmup_epochs': 50,
            'a': 1, 'b': 1, 'c': 0,
            'replay_stages': [0],
            'replay_ratio': 0.2,
            'data_dir': '/home/sharedata/zdg/multifreq_selected/freq_12to18',
        },
        {
            'name': 'high_freq',
            'freq_range': '18to25',
            'freq_min': 18.0, 'freq_max': 25.0,
            'NIter': 1001,
            'lr': 5e-5,                           # 高频更难，进一步降低
            'warmup_epochs': 50,
            'a': 1, 'b': 1, 'c': 0,
            'replay_stages': [0, 1],
            'replay_ratio': 0.2,
            'data_dir': '/home/sharedata/zdg/multifreq_selected/freq_18to25',
        },
    ]

    # ==========================================
    # 13. y_ran Epoch-Level 共享采样 (Epoch Shared Sampling)
    # ==========================================
    use_y_ran = False                           # 使用 y_ran 自由点参与 PDE 计算

    use_epoch_shared_y_ran = True              # True: 使用 epoch 级共享采样 | False: 使用原始 per-model 采样

    y_ran_num_pts = 400                        # y_ran 采样点总数
    y_ran_structure_ratio = 0.40               # epoch-structure 采样点比例
    y_ran_surface_ratio = 0.50                 # 表层采样点比例
    y_ran_uniform_ratio = 0.10                 # 均匀采样点比例
    y_ran_source_ratio = 0.0                   # 震源附近采样点比例 (实验A不使用)

    y_ran_surface_depth_grids = 5              # 表层深度（网格点数）
    y_ran_use_max_mix = False                  # True: score = mean_weight*mean + max_weight*max | False: 纯 mean
    y_ran_mean_weight = 0.7                    # mean+max 混合时的 mean 权重
    y_ran_max_weight = 0.3                     # mean+max 混合时的 max 权重

    # 概率图更新频率: 1=每epoch, >1=每N个epoch, 0=只计算一次并缓存
    y_ran_prob_update_every = 0
