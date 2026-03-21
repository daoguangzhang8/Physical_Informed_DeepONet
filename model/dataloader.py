from Labconfig import *
from model.utils import Halton_Sample


def Training_data(args, loc, vel, UU_loc, UU0_loc):
    """
    生成训练数据和验证数据
    保持原逻辑：使用 Halton 采样生成坐标点，计算 UU - UU0 的残差标签
    """
    # 1. 基本参数准备
    nvel, ny, Lpml = args.nvel_train, args.ny_train, args.Lpml
    spatial_step = 40
    nz, nx = vel.shape[1], vel.shape[2]
    valid_num = int(args.valid_rate * nvel) + 1
    
    # 2. 索引随机筛选
    idx = np.random.choice(vel.shape[0], nvel, replace=False)
    selected_idx_set = set(idx)
    remaining_idx = [i for i in range(len(vel)) if i not in selected_idx_set]
    
    # 定义波源坐标候选列表
    source_coords = [
        [Lpml//2 + 1, Lpml//2 + 5], [Lpml//2 + 1, Lpml//2 + 20], [Lpml//2 + 1, Lpml//2 + 35], 
        [Lpml//2 + 1, Lpml//2 + 50], [Lpml//2 + 1, Lpml//2 + 65]
    ]
    loc_list = [2] # 保持原样只取索引为2的波源
    
    # --- 核心处理逻辑封装（避免重复代码） ---
    def process_split(indices, count):
        curr_vel = vel[indices[:count], :, :].unsqueeze(1) # [N, 1, NZ, NX]
        
        # 目前 loc_list 只有一个元素 [2]，直接取第一个即可，保持原循环逻辑兼容性
        for loci in loc_list:
            # 提取对应的物理场数据
            u_current = UU_loc[loci][indices[:count], :, :, :]
            u0_current = UU0_loc[loci][indices[:count], :, :, :]
            
            # 计算标签 (UU - UU0)
            labels_current = u_current - u0_current
            
            # 计算 Source 坐标并扩展维度
            sz, sx = source_coords[loci]
            src_tensor = torch.tensor([sz, sx]).expand(count * ny, -1).float()
            
        return curr_vel, u_current, u0_current, labels_current, src_tensor * spatial_step

    # --- 3. 生成训练集数据 ---
    vel_train, UU_loc_train, UU0_train, labels, source_train = process_split(idx, nvel)
    
    # 训练集的坐标点 y_train (使用全部网格平铺逻辑)
    x_c = torch.arange(0, nx)
    z_c = torch.arange(0, nz)
    grid_z, grid_x = torch.meshgrid(z_c, x_c, indexing='ij') # 明确指定索引方式
    y_train = torch.stack([grid_z.flatten(), grid_x.flatten()], dim=1).float() * spatial_step

    # --- 4. 生成验证集数据 ---
    vel_valid, UU_loc_valid, UU0_valid, labels_valid, source_valid = process_split(remaining_idx, valid_num)
    
    # 验证集的坐标点 y_valid (使用 Halton 采样逻辑)
    # array_size = vel[0].shape
    # train_points = torch.tensor(Halton_Sample(array_size, ny)).long()
    # y_valid = torch.stack([train_points[:, 0], train_points[:, 1]], dim=1).float() * spatial_step
    y_valid = y_train
    
    # return vel_train, UU_loc_train, UU0_train, y_train, labels, vel_valid, UU_loc_valid, UU0_valid, y_valid, labels_valid
    return (
        vel_train, UU_loc_train, UU0_train, y_train, labels, 
        vel_valid, UU_loc_valid, UU0_valid, y_valid, labels_valid
    )
def Test_data_single(args, loc_idx, vel_single, UU_loc_single, UU0_loc_single):
    """
    专门用于加载单个测试模型（如 Marmousi）
    逻辑完全对齐训练脚本中的 valid 生成逻辑：
    1. 保持输入数据的维度一致
    2. 使用 Halton 采样生成坐标点 y_test
    3. 计算标签 labels = UU - UU0
    """
    # 1. 基本参数准备
    ny, Lpml = args.ny_train, args.Lpml
    spatial_step = 40
    nz, nx = vel_single.shape[-2], vel_single.shape[-1]
    
    # 确保输入是 Tensor 且维度正确 [1, NZ, NX]
    if isinstance(vel_single, np.ndarray):
        vel_single = torch.from_numpy(vel_single)
    
    # 2. 速度模型处理 (对齐 [1, 1, NZ, NX])
    # 假设输入是 [NZ, NX]，增加 Batch 和 Channel 维度
    vel_test = vel_single.view(1, 1, nz, nx).float()
    
    # 3. 提取指定波源位置的数据 (loc_idx 通常为 2)
    # 假设 UU_loc_single 传入的是该波源下的 [2, NZ, NX] 或包含波源维度的字典/列表
    u_current = UU_loc_single.view(1, 2, nz, nx).float()
    u0_current = UU0_loc_single.view(1, 2, nz, nx).float()
    
    # 计算标签 (UU - UU0) -> [1, 2, NZ, NX]
    labels_test = u_current - u0_current
    
    # 4. 波源坐标处理
    source_coords = [
        [Lpml//2 + 1, Lpml//2 + 5], [Lpml//2 + 1, Lpml//2 + 20], [Lpml//2 + 1, Lpml//2 + 35], 
        [Lpml//2 + 1, Lpml//2 + 50], [Lpml//2 + 1, Lpml//2 + 65]
    ]
    sz, sx = source_coords[loc_idx]
    # 对齐 valid 逻辑，扩展为 [1 * ny, 2]
    source_test = torch.tensor([sz, sx]).expand(1 * ny, -1).float() * spatial_step
    
    # 5. 坐标点采样 (严格对齐 valid 的 Halton 采样逻辑)
    array_size = (nz, nx)
    
    # Halton_Sample 需在外部已定义
    test_points = torch.tensor(Halton_Sample(array_size, ny)).long()
    x_c = torch.arange(0, nx)
    z_c = torch.arange(0, nz)
    grid_z, grid_x = torch.meshgrid(z_c, x_c, indexing='ij') # 明确指定索引方式
    y_train = torch.stack([grid_z.flatten(), grid_x.flatten()], dim=1).float() * spatial_step
    
    # 生成 y_test [ny, 2]
    y_test = y_train
    # 返回参数顺序与原函数保持一致
    # 这里的 _train 位置用 _test 代替，确保单模型输出
    return vel_test, u_current, u0_current, y_test, labels_test

def load_tensor_from_npy(base_path, filename):
    """通用的数据读取鲁棒接口"""
    path = os.path.join(base_path, filename)
    if not os.path.exists(path):
        path = filename  # Fallback
    return torch.tensor(np.load(path), dtype=torch.float32)

def prepare_training_dataloaders(args, device):
    """
    仅处理用于模型训练和内部验证的数据流
    """
    # 1. 基础训练数据读取
    vel_original = load_tensor_from_npy(args.load_path, 'velocity_data_70_70_n1.npy')
    UU0_original = load_tensor_from_npy(args.load_path, 'backgroundfield_data_freq5_1source_70_70_n1.npy')
    UU_original = load_tensor_from_npy(args.load_path, 'wavefield_data_freq5_5sources_70_70_n1.npy')
    
    # 2. PML 边界处理
    args.nx = args.nx + args.LD * 2
    args.nz = args.nz + args.LD * 2
    
    if args.pml:
        Lpml = args.Lpml
        vel = vel_original[:, Lpml:-Lpml, Lpml:-Lpml]
        UU0 = UU0_original[:, :, Lpml:-Lpml, Lpml:-Lpml]
        UU = UU_original[:, :, Lpml:-Lpml, Lpml:-Lpml]
    else:
        vel, UU0, UU = vel_original, UU0_original, UU_original

    # 3. 震源拆分与训练集生成
    UU_loc = [UU[loc * len(vel) : (loc + 1) * len(vel), ...] for loc in range(5)]
    UU0_loc = [UU0[loc * len(vel) : (loc + 1) * len(vel), ...] for loc in range(5)]
    
    np.random.seed(1)
    vel_train, UU_loc_train, UU0_train, y_train, labels_train, \
    vel_valid, UU_loc_valid, UU0_valid, y_valid, labels_valid = Training_data(args, 2, vel, UU_loc, UU0_loc)

    # 4. 物理场归一化
    vel_train = vel_train / 1000.
    vel_valid = vel_valid / 1000.

    vel_pred, UU0_pred, labels_pred = vel_valid[0:1], UU0_valid[0:1], labels_valid[0:1]
    vel_test, UU0_test, labels_test = vel_train[0:1], UU0_train[0:1], labels_train[0:1]

    # 5. 生成坐标网格点 (dx=dz=40)
    x_coords, z_coords = torch.arange(0, args.nx), torch.arange(0, args.nz)
    grid_z, grid_x = torch.meshgrid(z_coords, x_coords, indexing='ij')
    points = torch.stack([grid_z.flatten(), grid_x.flatten()], dim=1)
    y_pred = points.float() * 40
    y_test = y_pred

    # 6. 构建 DataLoader
    pin_mem = device.type == 'cuda'
    
    train_loaders = {
        "train": DataLoader(TensorDataset(vel_train, UU0_train, labels_train), 
                            batch_size=args.batch_size_v, shuffle=True, drop_last=True, pin_memory=pin_mem),
        "train_y": DataLoader(TensorDataset(y_train), 
                              batch_size=args.batch_size, shuffle=True, pin_memory=pin_mem),
        "valid": DataLoader(TensorDataset(vel_valid, UU0_valid, labels_valid), 
                            batch_size=args.valid_batch_size_v, shuffle=True, drop_last=True, pin_memory=pin_mem),
        "valid_y": DataLoader(TensorDataset(y_valid), 
                              batch_size=args.valid_batch_size, shuffle=True, pin_memory=pin_mem),
        "pred": DataLoader(TensorDataset(y_pred), batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(TensorDataset(y_test), batch_size=args.batch_size, shuffle=False)
    }

    plot_data = {
        "vel_pred": vel_pred, "UU0_pred": UU0_pred, "labels_pred": labels_pred,
        "vel_test": vel_test, "UU0_test": UU0_test, "labels_test": labels_test,
        "y_pred": y_pred  # 供外部验证集复用坐标
    }
    
    return train_loaders, plot_data

def prepare_external_val_dataset(args, prefix, loc_target, y_pred_grid):
    """
    通用接口：用于动态加载和处理单个外部验证集（如 Marmousi, BP 等）
    """
    # 1. 读取特定前缀的数据
    vel_ext = load_tensor_from_npy(args.load_path, f'{prefix}velocity_data_70_70_n1.npy')
    UU0_ext = load_tensor_from_npy(args.load_path, f'{prefix}backgroundfield_data_freq5_1source_70_70_n1.npy')
    UU_ext = load_tensor_from_npy(args.load_path, f'{prefix}wavefield_data_freq5_5sources_70_70_n1.npy')

    # 2. PML 边界处理
    if args.pml:
        Lpml = args.Lpml
        vel_ext = vel_ext.unsqueeze(0)[:, Lpml:-Lpml, Lpml:-Lpml]
        UU0_ext = UU0_ext[:, :, Lpml:-Lpml, Lpml:-Lpml]
        UU_ext = UU_ext[:, :, Lpml:-Lpml, Lpml:-Lpml]
    else:
        vel_ext = vel_ext.unsqueeze(0)

    # 3. 截取目标震源位置
    num_samples = len(vel_ext) 
    m_uu_single = UU_ext[loc_target * num_samples : (loc_target + 1) * num_samples]
    m_uu0_single = UU0_ext[loc_target * num_samples : (loc_target + 1) * num_samples]

    # 4. 生成测试格式数据
    v_test, u_test, u0_test, y_test, lab_test = Test_data_single(
        args, loc_target, vel_ext, m_uu_single, m_uu0_single
    )

    # 5. 归一化对齐训练逻辑
    v_test = v_test / 1000.0

    # 6. 生成专用的 DataLoader 和绘图数据字典
    ext_loader = DataLoader(TensorDataset(y_pred_grid), batch_size=args.batch_size, shuffle=False)
    
    ext_plot_data = {
        "v_test": v_test, 
        "u0_test": u0_test, 
        "lab_test": lab_test
    }
    
    print(f'External dataset [{prefix}] ready: vel_shape {v_test.shape}')
    return ext_loader, ext_plot_data