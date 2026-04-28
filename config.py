class Args:
    # ==========================================
    # 1. 路径与文件配置 (Paths & I/O)
    # ==========================================
    load_path = '/home/sharedata/zdg'         # 数据集加载根目录
    weights_save_path = '/home/sharedata/zdg' # 模型权重保存根目录
    save_doc = 'output_all'                       # 结果输出文件夹名称
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

    # 标签来源配置
    use_fno_as_label = False                  # True: 使用 FNO 预测作为软标签 | False: 使用真实标签
    fno_weights_path = ''                     # FNO 预训练权重路径 (当 use_fno_as_label=True 时需要指定)

    # ==========================================
    # 2. 硬件与设备配置 (Hardware & Device)
    # ==========================================
    device = 1                                # 单 GPU 模式: 指定使用的 GPU 设备编号
    use_parallel = False                      # 是否启用多 GPU 并行训练 (True: 多GPU | False: 单GPU)
    num_gpus = 2                              # 使用的 GPU 数量
    min_gpu_memory = 23 * 1024                # GPU 最小可用内存 (MB)，低于此值的 GPU 不会被使用

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

    # ==========================================
    # 4. 数据集与采样 (Dataset & Sampling)
    # ==========================================
    nvel_train = 1                           # 训练所用的速度模型数量
    source_list = [0]                         # [0, 1, 2, 3, 4] 训练数据中包含的震源编号列表 (0-4 共5个震源)

    # 空间点采样模式
    sampling_mode = 'halton'                  # 'full_grid': 全网格采样 | 'halton': Halton 准随机采样
    halton_sample_ratio = 0.5                   # Halton 采样比例（仅在 sampling_mode='halton' 时生效，如 0.2 表示采样 20% 的网格点）

    # 批处理配置
    batch_size = 800                          # Trunk Net 坐标采样批次大小 (num_sample)
    batch_size_v = 1                          # Branch Net 速度场/背景场批次大小 (Batch_v)
    ny_train = int(nz * nx * halton_sample_ratio)  # 训练集空间采样点总数 (由网格尺寸和采样比例自动计算)
    accumulation_steps = 4                    # 梯度累加步数 (用于等效增大 batch size，节约显存)

    # 验证集
    valid_rate = 0.1                          # 验证集划分比例
    valid_batch_size = 350                    # 验证集坐标采样批次大小
    valid_batch_size_v = 6                    # 验证集速度场批次大小

    # ==========================================
    # 5. 训练超参数 (Training Hyperparameters)
    # ==========================================
    NIter = 3000 + 1                         # 总训练 epoch 数 (+1 确保最后一步记录和保存生效)
    lr = 1 * 1e-4                             # 初始基础学习率
    weight_decay = 1e-4                       # 优化器权重衰减 (L2 正则化)系数

    # 学习率调度器 (ReduceLROnPlateau)
    warmup_epochs = 100                       # 学习率热身 (Warmup) 的 epoch 数
    factor = 0.9                              # 学习率衰减因子
    patience = 30                             # 触发衰减的容忍 epoch 数量
    min_lr = 1e-5                             # 允许的最小学习率

    # ==========================================
    # 6. 损失函数 (Loss Function)
    # ==========================================
    a = 1                                     # 数据拟合项 (Data Loss) 权重
    b = 1                                     # PDE 物理残差项 (PDE Loss) 权重
    c = 0                                     # 正则化项 (Regularization Loss) 权重
    d = 1                                   # 包络损失项 (Envelope Loss) 权重（MSE 形式，仅在标签点上计算）

    # 动态权重调整
    if_adjust = False                          # 是否在训练过程中动态调整 Loss 权重
    adjust_from = 2000                        # 从第几个 epoch 开始动态调整
    adjust_every = 1000                       # 每隔多少个 epoch 调整一次权重
    adjust_speed = 1.1                        # 权重衰减/增长的速度因子

    # ==========================================
    # 7. 训练控制与保存 (Training Control & Checkpoints)
    # ==========================================
    if_load_model = False                     # 是否加载预训练模型权重继续训练
    validate_every = 100                      # 每隔多少个 epoch 执行一次模型验证
    save_fig_every = 50                       # 每隔多少个 epoch 保存一次验证/测试可视化图片
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
    # 波数 PE 的 freq 和 c_ref 由训练数据动态提供:
    #   freq → freq_batch (来自 freesurface_freq_used_*.npy)
    #   c_ref → vel.mean() (每个速度模型的平均速度)

    # ==========================================
    # 10. 网络架构 (Network Architecture)
    # ==========================================
    in_channels = 2                           # 波场相关输入通道数 (如复数波场的实部、虚部)
    in_channels_vel = 1                       # 速度模型输入通道数 (1个通道代表速度 v)
    input_shape_trunk = (batch_size, in_channels, 1, 2)       # Trunk Net (评估坐标) 的输入形状占位
    input_shape_branch1 = (batch_size, in_channels_vel, nz, nx) # Branch Net 1 (速度场) 输入形状占位
    input_shape_branch2 = (batch_size, in_channels, nz, nx)     # Branch Net 2 (背景场/震源) 输入形状占位

    # ==========================================
    # 11. Sobol 采样配置 (Sobol Sampling)
    # ==========================================
    sampling_strategy = 'original'               # 'original': 双层循环+Halton | 'sobol': 单层循环+Sobol
    sobol_points_per_epoch = 10000              # Sobol 模式: 每 epoch 采样点数 (所有 velocity batch 共享)
    valid_sobol_points = 300                  # Sobol 模式: 验证集每 epoch 采样点数

    # ==========================================
    # 12. 三阶段渐进训练 (Staged Curriculum Training)
    # ==========================================
    staged_training = False                    # 总开关，False 则使用原始单阶段训练

    # 每阶段使用独立数据集，文件名通过 freq_range 替换基础文件名中的 'freq3to20' 得到
    # 例如基础文件名含 'freq3to20' → Stage 0 替换为 'freq3to11'
    stages = [
        {
            'name': 'low_freq',
            'freq_range': '3to11',               # 替换基础文件名中的 'freq3to20'
            'freq_min': 3.0, 'freq_max': 11.0,   # 信息标签，用于日志打印
            'NIter': 3500,
            'lr': 1e-4,                           # 从头训练，完整 LR
            'warmup_epochs': 100,
            'a': 1, 'b': 1, 'c': 0,
        },
        {
            'name': 'mid_freq',
            'freq_range': '12to18',
            'freq_min': 12.0, 'freq_max': 18.0,
            'NIter': 3500,
            'lr': 5e-5,                           # 课程学习，适度降低
            'warmup_epochs': 50,
            'a': 1, 'b': 1, 'c': 0,
        },
        {
            'name': 'high_freq',
            'freq_range': '18to25',
            'freq_min': 18.0, 'freq_max': 25.0,
            'NIter': 3500,
            'lr': 2e-5,                           # 高频更难，进一步降低
            'warmup_epochs': 50,
            'a': 1, 'b': 1, 'c': 0,
        },
    ]

    # ==========================================
    # 13. y_ran Epoch-Level 共享采样 (Epoch Shared Sampling)
    # ==========================================
    use_epoch_shared_y_ran = True              # True: 使用 epoch 级共享采样 | False: 使用原始 per-model 采样

    y_ran_num_pts = 900                        # y_ran 采样点总数
    y_ran_structure_ratio = 0.60               # epoch-structure 采样点比例
    y_ran_surface_ratio = 0.20                 # 表层采样点比例
    y_ran_uniform_ratio = 0.20                 # 均匀采样点比例
    y_ran_source_ratio = 0.0                   # 震源附近采样点比例 (实验A不使用)

    y_ran_surface_depth_grids = 5              # 表层深度（网格点数）
    y_ran_use_max_mix = False                  # True: score = mean_weight*mean + max_weight*max | False: 纯 mean
    y_ran_mean_weight = 0.7                    # mean+max 混合时的 mean 权重
    y_ran_max_weight = 0.3                     # mean+max 混合时的 max 权重

    # 概率图更新频率: 1=每epoch, >1=每N个epoch, 0=只计算一次并缓存
    y_ran_prob_update_every = 1
