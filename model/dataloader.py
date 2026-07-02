from Labconfig import *
from model.utils import Halton_Sample
import json



def Training_data(args, vel, UU_loc, UU0_loc, freq=None):
    """
    生成训练数据和验证数据（支持多震源并发训练）

    Args:
        freq: 频率数据 [N_vel]，每个速度模型对应一个频率值。若为 None 则使用默认值。
    """
    # 1. 基本参数准备
    nvel, ny, pml_crop = args.nvel_train, args.ny_train, args.pml_crop
    spatial_step = args.dh
    nz, nx = vel.shape[1], vel.shape[2]
    valid_num = int(args.valid_rate * nvel) + 1
    
    # 2. 索引随机筛选 (划分训练集和验证集的速度模型)
    idx = np.random.choice(vel.shape[0], nvel, replace=False)
    selected_idx_set = set(idx)
    remaining_idx = [i for i in range(len(vel)) if i not in selected_idx_set]
    
    # source_coords 已注释 — 硬编码坐标与实际数据不匹配，待后续从数据中自动检测
    # source_coords = [
    #     [pml_crop//2 + 1, pml_crop//2 + 5], [pml_crop//2 + 1, pml_crop//2 + 20], [pml_crop//2 + 1, pml_crop//2 + 35],
    #     [pml_crop//2 + 1, pml_crop//2 + 50], [pml_crop//2 + 1, pml_crop//2 + 65]
    # ]
    loc_list = args.source_list

    # --- 核心处理逻辑封装（支持多震源数据拼接） ---
    def process_split(indices, count):
        # 提取当前划分的速度模型基础张量 [count, 1, NZ, NX]
        base_vel = vel[indices[:count], :, :].unsqueeze(1)

        vel_list, u_list, u0_list, labels_list, freq_list = [], [], [], [], []

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

            # 4. freq 按震源复制（每个速度模型的 freq 对所有震源相同）
            if freq is not None:
                freq_list.append(freq[indices[:count]])

        # 沿 Batch 维度 (dim=0) 拼接所有震源的数据
        # 最终的 Batch Size = count * len(loc_list)
        vel_out = torch.cat(vel_list, dim=0)
        u_out = torch.cat(u_list, dim=0)
        u0_out = torch.cat(u0_list, dim=0)
        labels_out = torch.cat(labels_list, dim=0)
        freq_out = torch.cat(freq_list, dim=0) if freq_list else None

        return vel_out, u_out, u0_out, labels_out, freq_out

    # --- 3. 生成训练集数据 ---
    vel_train, UU_loc_train, UU0_train, labels, freq_train = process_split(idx, nvel)
    
    # 训练集的坐标点 y_train（所有样本共享一份以节省内存）
    if getattr(args, 'sampling_mode', 'full_grid') == 'halton':
        # Halton 准随机采样
        total_pts = nz * nx
        ratio = getattr(args, 'halton_sample_ratio', 0.2)
        num_pts = max(1, int(total_pts * ratio))
        halton_indices = Halton_Sample((nz, nx), num_pts)
        z_idx = torch.tensor([p[0] for p in halton_indices], dtype=torch.float32)
        x_idx = torch.tensor([p[1] for p in halton_indices], dtype=torch.float32)
        y_train = torch.stack([z_idx, x_idx], dim=1) * spatial_step
        print(f'[Halton] 采样点数: {num_pts}, y_train shape: {y_train.shape}')
    else:
        # 全网格采样（默认）
        x_c = torch.arange(0, nx)
        z_c = torch.arange(0, nz)
        grid_z, grid_x = torch.meshgrid(z_c, x_c, indexing='ij')
        y_train = torch.stack([grid_z.flatten(), grid_x.flatten()], dim=1).float() * spatial_step

    # --- 4. 生成验证集数据 ---
    vel_valid, UU_loc_valid, UU0_valid, labels_valid, freq_valid = process_split(remaining_idx, valid_num)
    y_valid = y_train  # 验证集坐标点与训练集保持一致

    return (
        vel_train, UU_loc_train, UU0_train, y_train, labels, freq_train,
        vel_valid, UU_loc_valid, UU0_valid, y_valid, labels_valid, freq_valid
    )

def Test_data_single(args, loc_idx, vel_single, UU_loc_single, UU0_loc_single):
    """
    专门用于加载测试模型（如 Marmousi），支持自适应单震源或多震源并发输入。
    """
    # 1. 基本参数准备
    spatial_step = args.dh
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

def resolve_npy_path(base_path, filename):
    path = os.path.join(base_path, filename)
    if not os.path.exists(path):
        path = filename  # Fallback
    return path


def load_tensor_from_npy(base_path, filename):
    """通用的数据读取鲁棒接口"""
    path = resolve_npy_path(base_path, filename)
    return torch.tensor(np.load(path), dtype=torch.float32)


def load_npy_mmap(base_path, filename):
    path = resolve_npy_path(base_path, filename)
    return np.load(path, mmap_mode='r')


def make_coordinate_grid(args, nz, nx):
    spatial_step = args.dh
    if getattr(args, 'sampling_mode', 'full_grid') == 'halton':
        total_pts = nz * nx
        ratio = getattr(args, 'halton_sample_ratio', 0.2)
        num_pts = max(1, int(total_pts * ratio))
        halton_indices = Halton_Sample((nz, nx), num_pts)
        z_idx = torch.tensor([p[0] for p in halton_indices], dtype=torch.float32)
        x_idx = torch.tensor([p[1] for p in halton_indices], dtype=torch.float32)
        y_grid = torch.stack([z_idx, x_idx], dim=1) * spatial_step
        print(f'[Halton] 采样点数: {num_pts}, y shape: {y_grid.shape}')
        return y_grid

    x_c = torch.arange(0, nx)
    z_c = torch.arange(0, nz)
    grid_z, grid_x = torch.meshgrid(z_c, x_c, indexing='ij')
    return torch.stack([grid_z.flatten(), grid_x.flatten()], dim=1).float() * spatial_step


def make_full_coordinate_grid(args, nz, nx):
    x_c = torch.arange(0, nx)
    z_c = torch.arange(0, nz)
    grid_z, grid_x = torch.meshgrid(z_c, x_c, indexing='ij')
    return torch.stack([grid_z.flatten(), grid_x.flatten()], dim=1).float() * args.dh


def get_pml_slices(args):
    if not args.pml:
        return slice(None), slice(None)

    pml_crop = args.pml_crop
    if args.boundary_type == 'free_surface':
        z_slice = slice(0, -pml_crop)
    else:
        z_slice = slice(pml_crop, -pml_crop)
    x_slice = slice(pml_crop, -pml_crop)
    return z_slice, x_slice


def read_multifreq_split(vel_np, uu0_np, uu_np, freq_np, flat_indices, source_list, z_slice, x_slice):
    n_freq = vel_np.shape[1]
    vel_ids = flat_indices // n_freq
    freq_ids = flat_indices % n_freq

    base_vel = torch.from_numpy(np.asarray(
        vel_np[vel_ids, freq_ids, z_slice, x_slice], dtype=np.float32
    )).unsqueeze(1)
    base_freq = torch.from_numpy(np.asarray(freq_np[vel_ids, freq_ids], dtype=np.float32))

    u0_list, label_list = [], []
    for src in source_list:
        u0_src = torch.from_numpy(np.asarray(
            uu0_np[vel_ids, freq_ids, src, :, z_slice, x_slice], dtype=np.float32
        ))
        label_src = torch.from_numpy(np.asarray(
            uu_np[vel_ids, freq_ids, src, :, z_slice, x_slice], dtype=np.float32
        ))
        label_src.sub_(u0_src)
        u0_list.append(u0_src)
        label_list.append(label_src)

    n_src_selected = len(source_list)
    vel_out = base_vel.repeat(n_src_selected, 1, 1, 1)
    freq_out = base_freq.repeat(n_src_selected)
    uu0_out = torch.cat(u0_list, dim=0)
    labels_out = torch.cat(label_list, dim=0)
    return vel_out, uu0_out, labels_out, freq_out


def read_multifreq_samples(vel_np, uu0_np, uu_np, freq_np, vel_ids, freq_ids, source_ids, z_slice, x_slice):
    vel_ids = np.asarray(vel_ids, dtype=np.int64)
    freq_ids = np.asarray(freq_ids, dtype=np.int64)
    source_ids = np.asarray(source_ids, dtype=np.int64)
    if not (len(vel_ids) == len(freq_ids) == len(source_ids)):
        raise ValueError(
            f'采样索引长度不一致: vel={len(vel_ids)}, freq={len(freq_ids)}, source={len(source_ids)}'
        )

    vel_out = torch.from_numpy(np.asarray(
        vel_np[vel_ids, freq_ids, z_slice, x_slice], dtype=np.float32
    )).unsqueeze(1)
    freq_out = torch.from_numpy(np.asarray(freq_np[vel_ids, freq_ids], dtype=np.float32))
    uu0_out = torch.from_numpy(np.asarray(
        uu0_np[vel_ids, freq_ids, source_ids, :, z_slice, x_slice], dtype=np.float32
    ))
    labels_out = torch.from_numpy(np.asarray(
        uu_np[vel_ids, freq_ids, source_ids, :, z_slice, x_slice], dtype=np.float32
    ))
    labels_out.sub_(uu0_out)
    return vel_out, uu0_out, labels_out, freq_out


def random_one_freq_source_indices(base_indices, n_freq, source_list, rng):
    base_indices = np.asarray(base_indices, dtype=np.int64)
    freq_ids = rng.integers(0, n_freq, size=len(base_indices), dtype=np.int64)
    source_choices = np.asarray(source_list, dtype=np.int64)
    source_ids = rng.choice(source_choices, size=len(base_indices), replace=True)
    return base_indices, freq_ids, source_ids


def load_category_metadata(args, n_base):
    data_dir = os.path.dirname(resolve_npy_path(args.load_path, args.vel_filename))
    category_path = os.path.join(data_dir, 'model_category.npy')
    names_path = os.path.join(data_dir, 'category_names.json')
    if os.path.exists(category_path):
        model_category = np.load(category_path, mmap_mode='r')
    else:
        model_category = np.zeros(n_base, dtype=np.int64)

    if os.path.exists(names_path):
        with open(names_path, 'r', encoding='utf-8') as f:
            raw_names = json.load(f)
        category_names = [raw_names[str(i)] for i in range(len(raw_names))]
    else:
        category_names = ['all']

    return model_category, category_names


def selected_category_ids(args, category_names):
    category_filter = getattr(args, 'category_filter', None)
    if category_filter is None:
        return list(range(len(category_names)))

    if isinstance(category_filter, (str, int)):
        category_filter = [category_filter]

    name_to_id = {name: idx for idx, name in enumerate(category_names)}
    ids = []
    for item in category_filter:
        if isinstance(item, str):
            if item not in name_to_id:
                raise ValueError(f'未知类别 {item}, 可用类别: {sorted(name_to_id)}')
            ids.append(name_to_id[item])
        else:
            ids.append(int(item))
    return ids


def count_for_category(config_value, category_name, default_value):
    if isinstance(config_value, dict):
        if category_name in config_value:
            return int(config_value[category_name])
        return int(default_value)
    if config_value is None:
        return int(default_value)
    return int(config_value)


def base_to_flat_indices(base_indices, n_freq):
    base_indices = np.asarray(base_indices, dtype=np.int64)
    return (base_indices[:, None] * n_freq + np.arange(n_freq)[None, :]).reshape(-1).astype(np.int64)


def base_to_filtered_flat_indices(base_indices, freq_np, frequency_filter=None):
    """Expand velocity ids to flattened velocity-frequency ids, optionally filtering frequency values."""
    base_indices = np.asarray(base_indices, dtype=np.int64)
    n_freq = freq_np.shape[1]
    if frequency_filter is None:
        return base_to_flat_indices(base_indices, n_freq)

    if np.isscalar(frequency_filter):
        frequency_filter = [frequency_filter]
    mask = np.isin(np.asarray(freq_np[base_indices]), np.asarray(frequency_filter, dtype=np.float32))
    local_vel_ids, freq_ids = np.where(mask)
    if len(local_vel_ids) == 0:
        raise ValueError(
            f'速度模型 {base_indices.tolist()} 中不存在 frequency_filter={list(frequency_filter)}'
        )
    return (base_indices[local_vel_ids] * n_freq + freq_ids).astype(np.int64)


def make_field_loader(vel, uu0, labels, freq, batch_size, shuffle, drop_last):
    dataset = TensorDataset(vel, uu0, labels, freq)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=False,
        num_workers=0,
    )


def configure_frequency_data_weight(args, freq_train):
    """Precompute the target mean weight over the whole training split."""
    if freq_train is None or not getattr(args, 'use_frequency_data_weight', False):
        return

    start_hz = float(getattr(args, 'frequency_data_weight_start_hz', 12.0))
    end_hz = float(getattr(args, 'frequency_data_weight_end_hz', 25.0))
    max_weight = float(getattr(args, 'frequency_data_weight_max', 1.0))
    if max_weight <= 1.0 or end_hz <= start_hz:
        setattr(args, 'frequency_data_weight_target_mean', 1.0)
        return

    freq = freq_train.float()
    progress = ((freq - start_hz) / (end_hz - start_hz)).clamp(0.0, 1.0)
    raw_weight = 1.0 + (max_weight - 1.0) * 0.5 * (1.0 - torch.cos(progress * torch.pi))
    target_mean = float(raw_weight.mean().item())
    setattr(args, 'frequency_data_weight_target_mean', max(target_mean, 1e-8))
    print(
        f'固定频率 DataLoss 权重: start={start_hz:g}Hz, end={end_hz:g}Hz, '
        f'max={max_weight:g}, target_mean={target_mean:.4f}'
    )


def prepare_multifreq_training_dataloaders(args, vel_np, uu0_np, uu_np, freq_np):
    """
    新数据格式:
      velocity: [nvel, n_freq, Z, X]
      background/wavefield: [nvel, n_freq, n_source, 2, Z, X]
      freq: [nvel, n_freq]

    训练样本单位为 (velocity_id, freq_id, source_id)，其中 source_id 由 args.source_list 控制。
    """
    if uu0_np.shape != uu_np.shape:
        raise ValueError(f'background shape {uu0_np.shape} != wavefield shape {uu_np.shape}')
    if vel_np.ndim != 4 or uu0_np.ndim != 6 or freq_np.ndim != 2:
        raise ValueError(
            f'新格式维度不匹配: velocity={vel_np.shape}, background={uu0_np.shape}, freq={freq_np.shape}'
        )
    if vel_np.shape[:2] != uu0_np.shape[:2] or vel_np.shape[:2] != freq_np.shape:
        raise ValueError(
            f'速度/波场/频率前两维不一致: velocity={vel_np.shape}, background={uu0_np.shape}, freq={freq_np.shape}'
        )

    z_slice, x_slice = get_pml_slices(args)
    n_base, n_freq = vel_np.shape[:2]
    n_src = uu0_np.shape[2]
    source_list = list(getattr(args, 'source_list', [0]))
    for src in source_list:
        if src < 0 or src >= n_src:
            raise ValueError(f'source_list 包含非法震源 {src}, 可用范围: 0-{n_src - 1}')

    valid_source_cfg = getattr(args, 'valid_source_list', None)
    valid_source_list = source_list if valid_source_cfg is None else list(valid_source_cfg)
    test_source_list = list(getattr(args, 'test_source_list', list(range(n_src))))
    for src_group_name, src_group in [('valid_source_list', valid_source_list), ('test_source_list', test_source_list)]:
        for src in src_group:
            if src < 0 or src >= n_src:
                raise ValueError(f'{src_group_name} 包含非法震源 {src}, 可用范围: 0-{n_src - 1}')

    sample_shape = vel_np[0, 0, z_slice, x_slice].shape
    args.nz, args.nx = sample_shape

    model_category, category_names = load_category_metadata(args, n_base)
    active_category_ids = selected_category_ids(args, category_names)
    n_train_cfg = getattr(args, 'nvel_train_per_category', None)
    if n_train_cfg is None:
        n_train_cfg = max(1, int(getattr(args, 'nvel_train', n_base) / max(1, len(active_category_ids))))
    n_valid_cfg = getattr(args, 'nvel_valid_per_category', None)

    rng = np.random.default_rng(1)
    train_flat_parts = []
    valid_flat_parts = []
    category_valid_tensors = {}
    category_test_tensors = {}
    split_summary = {}
    valid_random_one_freq = getattr(args, 'valid_random_one_freq', True)
    valid_random_one_source = getattr(args, 'valid_random_one_source', True)
    frequency_filter = getattr(args, 'frequency_filter', None)
    velocity_mode = getattr(args, 'velocity_mode', 'multi')
    if velocity_mode not in ('single', 'multi'):
        raise ValueError(f"velocity_mode 必须是 'single' 或 'multi', 当前为: {velocity_mode}")
    if velocity_mode == 'single':
        explicit_train_base = getattr(args, 'train_velocity_indices', None)
        explicit_valid_base = getattr(args, 'valid_velocity_indices', None)
        if explicit_train_base is None or explicit_valid_base is None:
            raise ValueError("velocity_mode='single' 时必须设置 train_velocity_indices 和 valid_velocity_indices")
    else:
        explicit_train_base = None
        explicit_valid_base = None
    if explicit_train_base is not None:
        explicit_train_base = np.asarray(explicit_train_base, dtype=np.int64)
    if explicit_valid_base is not None:
        explicit_valid_base = np.asarray(explicit_valid_base, dtype=np.int64)

    for category_id in active_category_ids:
        category_name = category_names[category_id]
        base_indices = np.where(model_category == category_id)[0]

        if explicit_train_base is not None:
            train_base = explicit_train_base[np.isin(explicit_train_base, base_indices)]
            if len(train_base) == 0:
                continue
            invalid = explicit_train_base[~np.isin(explicit_train_base, np.arange(n_base))]
            if len(invalid) > 0:
                raise ValueError(f'train_velocity_indices 越界: {invalid.tolist()}')
            if explicit_valid_base is None:
                raise ValueError('设置 train_velocity_indices 时必须同时设置 valid_velocity_indices')
            valid_base = explicit_valid_base[np.isin(explicit_valid_base, base_indices)]
            if len(valid_base) == 0:
                raise ValueError(f'类别 {category_name} 中没有有效的 valid_velocity_indices')
            if np.intersect1d(train_base, valid_base).size > 0:
                raise ValueError('train_velocity_indices 与 valid_velocity_indices 不能重叠')
        else:
            n_train_cat = count_for_category(n_train_cfg, category_name, 0)
            if n_train_cat <= 0:
                continue
            if n_train_cat >= len(base_indices):
                raise ValueError(
                    f'类别 {category_name} 只有 {len(base_indices)} 个速度模型, '
                    f'nvel_train_per_category={n_train_cat} 不应全部用完'
                )
            n_valid_default = int(getattr(args, 'valid_rate', 0.1) * n_train_cat) + 1
            n_valid_cat = count_for_category(n_valid_cfg, category_name, n_valid_default)
            if n_train_cat + n_valid_cat > len(base_indices):
                raise ValueError(
                    f'类别 {category_name} 训练+验证需要 {n_train_cat + n_valid_cat} 个速度模型, '
                    f'但只有 {len(base_indices)} 个'
                )
            shuffled = rng.permutation(base_indices)
            train_base = shuffled[:n_train_cat]
            valid_base = shuffled[n_train_cat:n_train_cat + n_valid_cat]

        train_flat_cat = base_to_filtered_flat_indices(train_base, freq_np, frequency_filter)
        train_flat_parts.append(train_flat_cat)

        if valid_random_one_freq and valid_random_one_source:
            if frequency_filter is not None:
                raise ValueError('设置 frequency_filter 时请将 valid_random_one_freq=False')
            valid_vel_ids, valid_freq_ids, valid_source_ids = random_one_freq_source_indices(
                valid_base, n_freq, valid_source_list, rng
            )
            category_valid_tensors[category_name] = read_multifreq_samples(
                vel_np, uu0_np, uu_np, freq_np,
                valid_vel_ids, valid_freq_ids, valid_source_ids,
                z_slice, x_slice,
            )
            valid_flat_cat = valid_vel_ids * n_freq + valid_freq_ids
            valid_flat_parts.append((valid_vel_ids, valid_freq_ids, valid_source_ids))
            valid_vf_count = len(valid_vel_ids)
        else:
            valid_flat_cat = base_to_filtered_flat_indices(valid_base, freq_np, frequency_filter)
            category_valid_tensors[category_name] = read_multifreq_split(
                vel_np, uu0_np, uu_np, freq_np, valid_flat_cat, valid_source_list, z_slice, x_slice
            )
            valid_flat_parts.append(valid_flat_cat)
            valid_vf_count = len(valid_flat_cat)

        category_test_tensors[category_name] = read_multifreq_split(
            vel_np, uu0_np, uu_np, freq_np, valid_flat_cat, test_source_list, z_slice, x_slice
        )
        split_summary[category_name] = {
            'train_velocity': int(len(train_base)),
            'valid_velocity': int(len(valid_base)),
            'train_velocity_frequency': int(len(train_flat_cat)),
            'valid_velocity_frequency': int(valid_vf_count),
        }

    if not train_flat_parts:
        raise ValueError('没有可用训练类别，请检查 category_filter 和 nvel_train_per_category')

    train_flat = rng.permutation(np.concatenate(train_flat_parts))
    if valid_random_one_freq and valid_random_one_source:
        valid_vel_ids = np.concatenate([part[0] for part in valid_flat_parts])
        valid_freq_ids = np.concatenate([part[1] for part in valid_flat_parts])
        valid_source_ids = np.concatenate([part[2] for part in valid_flat_parts])
    else:
        valid_flat = np.concatenate(valid_flat_parts)

    vel_train, UU0_train, labels_train, freq_train = read_multifreq_split(
        vel_np, uu0_np, uu_np, freq_np, train_flat, source_list, z_slice, x_slice
    )
    if valid_random_one_freq and valid_random_one_source:
        vel_valid, UU0_valid, labels_valid, freq_valid = read_multifreq_samples(
            vel_np, uu0_np, uu_np, freq_np,
            valid_vel_ids, valid_freq_ids, valid_source_ids,
            z_slice, x_slice,
        )
    else:
        vel_valid, UU0_valid, labels_valid, freq_valid = read_multifreq_split(
            vel_np, uu0_np, uu_np, freq_np, valid_flat, valid_source_list, z_slice, x_slice
        )

    vel_train = vel_train / 1000.
    vel_valid = vel_valid / 1000.
    configure_frequency_data_weight(args, freq_train)
    y_train = make_coordinate_grid(args, args.nz, args.nx)
    y_valid = y_train

    print('新多频数据格式:')
    print(f'  velocity: {vel_np.shape}, background/wavefield: {uu0_np.shape}, freq: {freq_np.shape}')
    print(f'  velocity_mode: {velocity_mode}')
    print(f'  train source_list: {source_list}, valid_source_list: {valid_source_list}, test_source_list: {test_source_list}')
    print(f'  frequency_filter: {frequency_filter}')
    if explicit_train_base is not None:
        print(f'  fixed velocity split: train={explicit_train_base.tolist()}, valid={explicit_valid_base.tolist()}')
    print(f'  valid random one freq/source: {valid_random_one_freq and valid_random_one_source}')
    for category_name, info in split_summary.items():
        print(
            f'  {category_name}: train_vel={info["train_velocity"]}, '
            f'valid_vel={info["valid_velocity"]}, '
            f'train_vf={info["train_velocity_frequency"]}, valid_vf={info["valid_velocity_frequency"]}'
        )
    print(f'  train: vel={vel_train.shape}, UU0={UU0_train.shape}, labels={labels_train.shape}, freq={freq_train.shape}')
    print(f'  valid: vel={vel_valid.shape}, UU0={UU0_valid.shape}, labels={labels_valid.shape}, freq={freq_valid.shape}')

    hf_test_idx = freq_train.argmax().item()
    hf_pred_idx = freq_valid.argmax().item()
    vel_pred = vel_valid[hf_pred_idx:hf_pred_idx + 1]
    UU0_pred = UU0_valid[hf_pred_idx:hf_pred_idx + 1]
    labels_pred = labels_valid[hf_pred_idx:hf_pred_idx + 1]
    vel_test = vel_train[hf_test_idx:hf_test_idx + 1]
    UU0_test = UU0_train[hf_test_idx:hf_test_idx + 1]
    labels_test = labels_train[hf_test_idx:hf_test_idx + 1]

    y_pred = make_full_coordinate_grid(args, args.nz, args.nx)
    y_test = y_pred

    train_loaders = {
        "train": make_field_loader(
            vel_train, UU0_train, labels_train, freq_train,
            args.batch_size_v, shuffle=True, drop_last=True,
        ),
        "train_y": DataLoader(TensorDataset(y_train),
                              batch_size=args.batch_size, shuffle=True, pin_memory=False,
                              num_workers=0),
        "valid": make_field_loader(
            vel_valid, UU0_valid, labels_valid, freq_valid,
            args.valid_batch_size_v, shuffle=False, drop_last=False,
        ),
        "valid_y": DataLoader(TensorDataset(y_valid),
                              batch_size=args.valid_batch_size, shuffle=True, pin_memory=False,
                              num_workers=0),
        "pred": DataLoader(TensorDataset(y_pred), batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(TensorDataset(y_test), batch_size=args.batch_size, shuffle=False)
    }
    train_loaders["valid_by_category"] = {}
    for category_name, tensors in category_valid_tensors.items():
        v_cat, u0_cat, lab_cat, f_cat = tensors
        v_cat = v_cat / 1000.
        train_loaders["valid_by_category"][category_name] = make_field_loader(
            v_cat, u0_cat, lab_cat, f_cat,
            args.valid_batch_size_v, shuffle=False, drop_last=False,
        )
        train_loaders[f"valid_{category_name}"] = train_loaders["valid_by_category"][category_name]

    plot_data = {
        "vel_pred": vel_pred, "UU0_pred": UU0_pred, "labels_pred": labels_pred,
        "vel_test": vel_test, "UU0_test": UU0_test, "labels_test": labels_test,
        "y_pred": y_pred,
        "freq_train": freq_train[hf_test_idx:hf_test_idx + 1],
        "freq_valid": freq_valid[hf_pred_idx:hf_pred_idx + 1],
        "valid_by_category": train_loaders["valid_by_category"],
        "test_by_category": {},
        "split_summary": split_summary,
        "has_freq": True,
    }
    for category_name, tensors in category_test_tensors.items():
        v_cat, u0_cat, lab_cat, f_cat = tensors
        plot_data["test_by_category"][category_name] = {
            "vel": v_cat / 1000.,
            "UU0": u0_cat,
            "labels": lab_cat,
            "freq": f_cat,
            "source_list": test_source_list,
        }
    return train_loaders, plot_data

def prepare_training_dataloaders(args, device):
    """
    仅处理用于模型训练和内部验证的数据流
    """
    # 1. 基础训练数据读取。先用 mmap 探测 shape，避免新格式大数组一次性读入内存。
    vel_np = load_npy_mmap(args.load_path, args.vel_filename)
    UU0_np = load_npy_mmap(args.load_path, args.backgroundfield_filename)
    UU_np = load_npy_mmap(args.load_path, args.wavefield_filename)

    # 加载频率数据（若文件存在）
    freq_filename = getattr(args, 'freq_filename', None)
    if freq_filename:
        freq_path = resolve_npy_path(args.load_path, freq_filename)
        if os.path.exists(freq_path):
            freq_np = np.load(freq_path, mmap_mode='r')
            print(f'已加载频率数据: {freq_filename}, shape: {freq_np.shape}, 唯一值: {np.unique(freq_np).tolist()}')
        else:
            print(f'⚠️ 频率文件不存在: {freq_path}，将使用默认频率')
            freq_np = None
    else:
        freq_np = None

    if vel_np.ndim == 4 and UU0_np.ndim == 6 and UU_np.ndim == 6 and freq_np is not None:
        return prepare_multifreq_training_dataloaders(args, vel_np, UU0_np, UU_np, freq_np)

    # 旧格式兜底：保持原有逻辑
    vel_original = torch.tensor(np.asarray(vel_np), dtype=torch.float32)
    UU0_original = torch.tensor(np.asarray(UU0_np), dtype=torch.float32)
    UU_original = torch.tensor(np.asarray(UU_np), dtype=torch.float32)
    freq = torch.tensor(np.asarray(freq_np), dtype=torch.float32) if freq_np is not None else None

    # 2. PML 边界处理
    if args.pml:
        pml_crop = args.pml_crop
        # 根据边界类型确定切片范围
        if args.boundary_type == 'free_surface':
            z_slice = slice(0, -pml_crop)          # 顶部不切，底部切 pml_crop
        else:  # 'full_pml'
            z_slice = slice(pml_crop, -pml_crop)   # 上下都切 pml_crop

        x_slice = slice(pml_crop, -pml_crop)       # 左右都切 pml_crop

        vel = vel_original[:, z_slice, x_slice]
        UU0 = UU0_original[:, :, z_slice, x_slice]
        UU = UU_original[:, :, z_slice, x_slice]

        # 更新 args.nz 和 args.nx 为切片后的实际尺寸
        args.nz = vel.shape[1]  # 实际的 z 维度
        args.nx = vel.shape[2]  # 实际的 x 维度
    else:
        vel, UU0, UU = vel_original, UU0_original, UU_original
        # 无 PML 时，使用原始数据尺寸
        args.nz = vel.shape[1]
        args.nx = vel.shape[2]

    # 3. 多频段数据重排: [freq0_all, freq1_all, ...] → [src0_all, src1_all, ...]
    n_freq = getattr(args, 'n_freq_ranges', 1)
    if n_freq > 1:
        n_src = UU0.shape[0] // vel.shape[0]  # 5
        n_vel_per_freq = vel.shape[0] // n_freq
        # reshape: (n_freq * n_src * n_vel_per_freq, C, H, W) → (n_freq, n_src, n_vel_per_freq, C, H, W)
        UU0 = UU0.reshape(n_freq, n_src, n_vel_per_freq, *UU0.shape[1:])
        UU0 = UU0.permute(1, 0, 2, *range(3, UU0.dim())).contiguous().reshape(n_src * vel.shape[0], *UU0.shape[3:])
        UU = UU.reshape(n_freq, n_src, n_vel_per_freq, *UU.shape[1:])
        UU = UU.permute(1, 0, 2, *range(3, UU.dim())).contiguous().reshape(n_src * vel.shape[0], *UU.shape[3:])

    # 4. 震源拆分与训练集生成
    UU_loc = [UU[loc * len(vel) : (loc + 1) * len(vel), ...] for loc in range(5)]
    UU0_loc = [UU0[loc * len(vel) : (loc + 1) * len(vel), ...] for loc in range(5)]
    
    np.random.seed(1)
    vel_train, UU_loc_train, UU0_train, y_train, labels_train, freq_train, \
    vel_valid, UU_loc_valid, UU0_valid, y_valid, labels_valid, freq_valid = Training_data(args, vel, UU_loc, UU0_loc, freq)
    print('vel_train', vel_train.shape)
    # 4. 物理场归一化
    vel_train = vel_train / 1000.
    vel_valid = vel_valid / 1000.
    configure_frequency_data_weight(args, freq_train)

    # 绘图示例: 选取高频样本 (freq 值最大的样本)
    if freq_train is not None:
        hf_test_idx = freq_train.argmax().item()
        hf_pred_idx = freq_valid.argmax().item()
    else:
        hf_test_idx = len(vel_train) - 1
        hf_pred_idx = len(vel_valid) - 1

    vel_pred, UU0_pred, labels_pred = vel_valid[hf_pred_idx:hf_pred_idx+1], UU0_valid[hf_pred_idx:hf_pred_idx+1], labels_valid[hf_pred_idx:hf_pred_idx+1]
    vel_test, UU0_test, labels_test = vel_train[hf_test_idx:hf_test_idx+1], UU0_train[hf_test_idx:hf_test_idx+1], labels_train[hf_test_idx:hf_test_idx+1]

    # 5. 生成坐标网格点
    x_coords, z_coords = torch.arange(0, args.nx), torch.arange(0, args.nz)
    grid_z, grid_x = torch.meshgrid(z_coords, x_coords, indexing='ij')
    points = torch.stack([grid_z.flatten(), grid_x.flatten()], dim=1)
    y_pred = points.float() * args.dh
    y_test = y_pred

    # 6. 构建 DataLoader
    pin_mem = False
    num_workers = 0

    # 构建 DataLoader 时根据 freq 是否存在决定 TensorDataset 内容
    train_ds = (TensorDataset(vel_train, UU0_train, labels_train, freq_train)
                if freq_train is not None
                else TensorDataset(vel_train, UU0_train, labels_train))
    valid_ds = (TensorDataset(vel_valid, UU0_valid, labels_valid, freq_valid)
                if freq_valid is not None
                else TensorDataset(vel_valid, UU0_valid, labels_valid))

    train_loaders = {
        "train": DataLoader(train_ds,
                            batch_size=args.batch_size_v, shuffle=True, drop_last=True,
                            pin_memory=pin_mem, num_workers=num_workers),
        "train_y": DataLoader(TensorDataset(y_train),
                              batch_size=args.batch_size, shuffle=True, pin_memory=pin_mem,
                              num_workers=num_workers),
        "valid": DataLoader(valid_ds,
                            batch_size=args.valid_batch_size_v, shuffle=True, drop_last=True,
                            pin_memory=pin_mem, num_workers=num_workers),
        "valid_y": DataLoader(TensorDataset(y_valid),
                              batch_size=args.valid_batch_size, shuffle=True, pin_memory=pin_mem,
                              num_workers=num_workers),
        "pred": DataLoader(TensorDataset(y_pred), batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(TensorDataset(y_test), batch_size=args.batch_size, shuffle=False)
    }

    plot_data = {
        "vel_pred": vel_pred, "UU0_pred": UU0_pred, "labels_pred": labels_pred,
        "vel_test": vel_test, "UU0_test": UU0_test, "labels_test": labels_test,
        "y_pred": y_pred,
        "freq_train": freq_train[hf_test_idx:hf_test_idx+1] if freq_train is not None else None,
        "freq_valid": freq_valid[hf_pred_idx:hf_pred_idx+1] if freq_valid is not None else None,
        "has_freq": freq_train is not None,
    }
    
    return train_loaders, plot_data

def prepare_external_val_dataset(args, prefix, loc_target, y_pred_grid):
    """
    通用接口：用于动态加载和处理单个外部验证集（如 Marmousi, BP 等）
    """
    # 1. 读取特定前缀的数据 (文件名中尺寸需与实际数据匹配)
    grid_suffix = f'{args.nz}_{args.nx}_n1.npy'
    vel_ext = load_tensor_from_npy(args.load_path, f'{prefix}velocity_data_{grid_suffix}')
    UU0_ext = load_tensor_from_npy(args.load_path, f'{prefix}backgroundfield_data_freq5_1source_{grid_suffix}')
    UU_ext = load_tensor_from_npy(args.load_path, f'{prefix}wavefield_data_freq5_5sources_{grid_suffix}')

    # 2. PML 边界处理
    if args.pml:
        pml_crop = args.pml_crop
        # 根据边界类型确定切片范围
        if args.boundary_type == 'free_surface':
            z_slice = slice(0, -pml_crop)
        else:  # 'full_pml'
            z_slice = slice(pml_crop, -pml_crop)
        x_slice = slice(pml_crop, -pml_crop)

        vel_ext = vel_ext.unsqueeze(0)[:, z_slice, x_slice]
        UU0_ext = UU0_ext[:, :, z_slice, x_slice]
        UU_ext = UU_ext[:, :, z_slice, x_slice]
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
