from Labconfig import *
from model.utils import Halton_Sample



def Training_data(args, vel, UU_loc, UU0_loc):
    """
    生成训练数据和验证数据（支持多震源并发训练）
    """
    # 1. 基本参数准备
    nvel, ny, Lpml = args.nvel_train, args.ny_train, args.Lpml
    spatial_step = 40
    nz, nx = vel.shape[1], vel.shape[2]
    valid_num = int(args.valid_rate * nvel) + 1
    
    # 2. 索引随机筛选 (划分训练集和验证集的速度模型)
    idx = np.random.choice(vel.shape[0], nvel, replace=False)
    selected_idx_set = set(idx)
    remaining_idx = [i for i in range(len(vel)) if i not in selected_idx_set]
    
    # 定义波源坐标候选列表
    source_coords = [
        [Lpml//2 + 1, Lpml//2 + 5], [Lpml//2 + 1, Lpml//2 + 20], [Lpml//2 + 1, Lpml//2 + 35], 
        [Lpml//2 + 1, Lpml//2 + 50], [Lpml//2 + 1, Lpml//2 + 65]
    ]
    loc_list = args.source_list  # 现在可以包含多个震源，例如 [1, 2, 3]
    
    # --- 核心处理逻辑封装（支持多震源数据拼接） ---
    def process_split(indices, count):
        # 提取当前划分的速度模型基础张量 [count, 1, NZ, NX]
        base_vel = vel[indices[:count], :, :].unsqueeze(1) 
        
        vel_list, u_list, u0_list, labels_list, src_list = [], [], [], [], []
        
        # 遍历所有被激活的震源
        for loci in loc_list:
            # 1. 速度模型对每个震源都是一样的，直接复制加入列表
            vel_list.append(base_vel)
            
            # 2. 提取对应震源的物理场数据 [count, 2, NZ, NX]
            u_current = UU_loc[loci][indices[:count], :, :, :]
            u0_current = UU0_loc[loci][indices[:count], :, :, :]
            
            # 3. 计算标签残差
            labels_current = u_current - u0_current
            
            u_list.append(u_current)
            u0_list.append(u0_current)
            labels_list.append(labels_current)
            
            # 4. 计算当前 Source 坐标并扩展维度 (适配当前 batch 大小 count)
            sz, sx = source_coords[loci]
            # 这里将维度扩展为 [count, 2]，代表这 count 个样本对应同一个震源坐标
            src_tensor = torch.tensor([sz, sx]).expand(count, -1).float()
            src_list.append(src_tensor * spatial_step)

        # 沿 Batch 维度 (dim=0) 拼接所有震源的数据
        # 最终的 Batch Size = count * len(loc_list)
        vel_out = torch.cat(vel_list, dim=0)
        u_out = torch.cat(u_list, dim=0)
        u0_out = torch.cat(u0_list, dim=0)
        labels_out = torch.cat(labels_list, dim=0)
        src_out = torch.cat(src_list, dim=0)
            
        return vel_out, u_out, u0_out, labels_out, src_out

    # --- 3. 生成训练集数据 ---
    vel_train, UU_loc_train, UU0_train, labels, source_train = process_split(idx, nvel)
    
    # 训练集的坐标点 y_train (使用全部网格平铺逻辑，所有样本共享一份即可节省内存)
    x_c = torch.arange(0, nx)
    z_c = torch.arange(0, nz)
    grid_z, grid_x = torch.meshgrid(z_c, x_c, indexing='ij') 
    y_train = torch.stack([grid_z.flatten(), grid_x.flatten()], dim=1).float() * spatial_step

    # --- 4. 生成验证集数据 ---
    vel_valid, UU_loc_valid, UU0_valid, labels_valid, source_valid = process_split(remaining_idx, valid_num)
    y_valid = y_train  # 验证集坐标点与训练集保持一致
    
    # 注意：原代码的 return 中没有包含 source_train 和 source_valid。
    # 如果您的 DeepONet 损失函数（或网络输入）需要动态震源坐标，请考虑在返回值中加入它们！
    return (
        vel_train, UU_loc_train, UU0_train, y_train, labels, 
        vel_valid, UU_loc_valid, UU0_valid, y_valid, labels_valid
    )

def Test_data_single(args, loc_idx, vel_single, UU_loc_single, UU0_loc_single):
    """
    专门用于加载测试模型（如 Marmousi），支持自适应单震源或多震源并发输入。
    """
    # 1. 基本参数准备
    spatial_step = 40
    nz, nx = vel_single.shape[-2], vel_single.shape[-1]
    
    # 确保输入是 Tensor
    if isinstance(vel_single, np.ndarray):
        vel_single = torch.from_numpy(vel_single)
    if isinstance(UU_loc_single, np.ndarray):
        UU_loc_single = torch.from_numpy(UU_loc_single)
    if isinstance(UU0_loc_single, np.ndarray):
        UU0_loc_single = torch.from_numpy(UU0_loc_single)
        
    # ==========================================
    # 核心修改点：自适应维度推导
    # ==========================================
    # 2. 提取波场数据，利用 -1 自动推导包含的震源数量 (num_sources)
    u_current = UU_loc_single.view(-1, 2, nz, nx).float()
    u0_current = UU0_loc_single.view(-1, 2, nz, nx).float()
    
    num_sources = u_current.shape[0]  # 获取实际传进来的震源数量 (例如 1 或 5)
    
    # 3. 速度模型处理
    # 速度模型本身只有 1 个，为了能放进 Dataloader，必须复制成 num_sources 份与波场对齐
    vel_test = vel_single.view(1, 1, nz, nx).expand(num_sources, -1, -1, -1).float()
    
    # 4. 计算标签 (UU - UU0) -> [num_sources, 2, NZ, NX]
    labels_test = u_current - u0_current
    
    # ==========================================
    # 坐标点采样
    # ==========================================
    # 生成全空间网格
    x_c = torch.arange(0, nx)
    z_c = torch.arange(0, nz)
    grid_z, grid_x = torch.meshgrid(z_c, x_c, indexing='ij') 
    
    # 展平成 [NZ*NX, 2]
    y_grid = torch.stack([grid_z.flatten(), grid_x.flatten()], dim=1).float() * spatial_step
    
    # 同样地，网格坐标也需要扩展成 num_sources 份，变成 [num_sources, NZ*NX, 2]
    y_test = y_grid.unsqueeze(0).expand(num_sources, -1, -1)
    
    # 返回参数顺序与原函数保持一致
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
    vel_valid, UU_loc_valid, UU0_valid, y_valid, labels_valid = Training_data(args, vel, UU_loc, UU0_loc)
    print('vel_train', vel_train.shape)
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
    # m_uu_single = UU_ext[loc_target * num_samples : (loc_target + 1) * num_samples]
    # m_uu0_single = UU0_ext[loc_target * num_samples : (loc_target + 1) * num_samples]
    # 兼容 loc_target 是列表（多震源）或整数（单震源）的情况
    if isinstance(loc_target, list):
        m_uu_single = torch.cat([UU_ext[loc * num_samples : (loc + 1) * num_samples] for loc in loc_target], dim=0)
        m_uu0_single = torch.cat([UU0_ext[loc * num_samples : (loc + 1) * num_samples] for loc in loc_target], dim=0)
        
        # 注意：如果你的速度场 v_ext 只有一份（形状如 [1, 1, Z, X]），
        # 在拼接成 Dataloader 之前，可能需要将其按震源数量复制对齐：
        # v_ext = v_ext.repeat(len(loc_target), 1, 1, 1) 
    else:
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

# def extract_single_model_multi_source(args, vel_set, UU0_set, labels_set, target_model_idx=0):
#     """
#     从按震源顺序拼接的数据集中，提取出【指定索引】的一个速度模型及其对应的多震源波场数据。
    
#     Args:
#         args: 全局参数，需包含 args.source_list (例如 [0, 1, 2, 3, 4])
#         vel_set: 训练或验证集的速度场 Tensor [base_count * num_sources, 1, Z, X]
#         UU0_set: 背景波场 Tensor [base_count * num_sources, 2, Z, X]
#         labels_set: 真实标签 Tensor [base_count * num_sources, 2, Z, X]
#         base_count: 该集合基础速度模型的数量 (train集为 nvel_train, valid集为 valid_num)
#         target_model_idx: 指定要提取第几个速度模型 (0 <= target_model_idx < base_count)
        
#     Returns:
#         model_data_pack (dict): 包含画图所需的 vel, UU0_list, labels_list
#     """
#     num_sources = len(args.source_list)
#     base_count = num_sources // 5
#     # 防止索引越界
#     if target_model_idx >= base_count or target_model_idx < 0:
#         raise ValueError(f"指定的索引 {target_model_idx} 超出范围，该集合只有 {base_count} 个基础模型。")
    
#     # 1. 提取指定索引的速度模型 (扩展出 batch=1 的维度 [1, 1, Z, X])
#     vel_single = vel_set[target_model_idx].unsqueeze(0)
    
#     UU0_list = []
#     labels_list = []
    
#     # 2. 跨块跳跃提取该模型在所有震源下的波场数据
#     for s in range(num_sources):
#         # 核心索引公式：指定模型索引 + 震源索引 * 基础模型数量
#         target_idx = target_model_idx + s * base_count
        
#         UU0_list.append(UU0_set[target_idx].unsqueeze(0))      # [1, 2, Z, X]
#         labels_list.append(labels_set[target_idx].unsqueeze(0)) # [1, Z, X, 2] 或其它对应维度
        
#     # 3. 组装返回
#     model_data_pack = {
#         "vel": vel_single,
#         "UU0_list": UU0_list,
#         "labels_list": labels_list
#     }
    
#     return model_data_pack