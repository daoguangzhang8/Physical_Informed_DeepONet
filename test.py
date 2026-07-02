"""
PI-DeepONet 外部测试集评估脚本

加载 external_test/ 中的测试数据，支持通过命令行指定震源位置和测试频率。

Usage:
    python test.py --dataset marmousi --sources 2 --freqs 5,10,15,20,25
    python test.py --dataset overthrust --sources 0,2,4 --freqs 10 --weights output2/xxx.pth
    python test.py --dataset random --sources 2 --freqs 5,15,25 --finetune
"""

import argparse
import os
import re
import glob
import numpy as np
import torch
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from model.utils import *
from model.PI_DeepOnet import Pi_DeepONet
from model.FNO import FNO
from model.plotting import calculate_regression_metrics, fine_tuning

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.environ.get('PIDEEPONET_DATA_ROOT', '/home/sharedata/zdg')
DEFAULT_TEST_DATA_DIR = os.path.join(DEFAULT_DATA_ROOT, 'external_test')
DEFAULT_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'output_cnn1')


# =====================================================================
# 命令行参数
# =====================================================================
def parse_cli():
    p = argparse.ArgumentParser(description='PI-DeepONet 外部测试集评估')
    p.add_argument('--dataset', type=str, default='marmousi',
                   choices=['marmousi', 'overthrust', 'random', 'marmousi_alt'],
                   help='测试数据集名称 (默认 marmousi)')
    p.add_argument('--sources', type=str, default='2',
                   help='震源索引, 逗号分隔 (默认 "2")')
    p.add_argument('--freqs', type=str, default='5,10,15,20,25',
                   help='测试频率, 逗号分隔 (默认 "5,10,15,20,25")')
    p.add_argument('--weights', type=str, default=None,
                   help='模型权重路径')
    p.add_argument('--data-dir', type=str, default=None,
                   help='外部测试数据目录 (默认 /home/sharedata/zdg/external_test)')
    p.add_argument('--weights-dir', type=str, default=None,
                   help='未指定 --weights 时自动查找权重的目录 (默认 output_cnn1)')
    p.add_argument('--device', type=int, default=None,
                   help='GPU 编号')
    p.add_argument('--finetune', action='store_true',
                   help='启用域适应微调')
    p.add_argument('--output', type=str, default=None,
                   help='输出目录 (默认 output_test_{dataset})')
    return p.parse_args()


# =====================================================================
# 测试配置 (与 config.py 格式一致，可直接修改此处的默认值)
# =====================================================================
class ArgsTest:
    # ==========================================
    # 1. 路径与文件配置 (Paths & I/O)
    # ==========================================
    load_path = DEFAULT_DATA_ROOT             # 测试数据根目录
    weights_save_path = PROJECT_ROOT          # 模型权重保存根目录
    save_doc = 'output_test'                  # 结果输出文件夹名称 (会被 CLI --output 覆盖)
    filename = 'PI_DeepONet_pde'             # 保存的模型前缀名称

    # 测试数据路径
    test_data_dir = DEFAULT_TEST_DATA_DIR     # 外部测试数据目录

    # ==========================================
    # 2. 硬件与设备配置 (Hardware & Device)
    # ==========================================
    device = 2                                # GPU 设备编号 (会被 CLI --device 覆盖)
    use_parallel = False

    # ==========================================
    # 3. 物理网格与边界条件 (Physical Grid & PML)
    # ==========================================
    dh = 20                                   # 空间网格间距 (m)，物理坐标 = 网格索引 * dh
    nx = 140                                  # 物理模型 x 方向网格数 (不含外延 PML)
    nz = 140                                  # 物理模型 z 方向网格数 (不含外延 PML)
    pml = True                                # 是否启用 PML 吸收边界
    pml_total = 20                            # PML 吸收层的总网格厚度
    pml_crop = 15                             # 裁剪/忽略的 PML 网格数
    pml_active = pml_total - pml_crop         # 剩余参与评估的 PML 网格数

    # 边界类型配置
    boundary_type = 'free_surface'            # 'free_surface' | 'full_pml'

    # ==========================================
    # 4. 测试数据筛选 (Test Data Selection)
    # ==========================================
    source_list = [2]                         # 默认震源列表 (会被 CLI --sources 覆盖)
    freq_list = [5, 10, 15, 20, 25]          # 默认频率列表 (会被 CLI --freqs 覆盖)
    n_freq_ranges = 3                         # 合并数据来源的频段数量

    # ==========================================
    # 5. 模型权重配置 (Model Weights)
    # ==========================================
    model_weights_path = ''                   # 模型权重路径 (会被 CLI --weights 覆盖)
    trained_weights_dir = DEFAULT_WEIGHTS_DIR # 未指定权重时，从该目录自动选择最新权重
    # ==========================================
    # 5.5 网络架构 (Architecture)
    # ==========================================
    branch1_modes = 12                        # 会根据 checkpoint 自动覆盖
    branch1_width = 32                        # 会根据 checkpoint 自动覆盖
    branch2_type = 'conv'                     # Branch2 架构: 'hybrid' | 'fno' | 'resnet' | 'conv'
    branch2_modes = 32                        # 会根据 checkpoint 自动覆盖
    branch2_width = 32                        # 会根据 checkpoint 自动覆盖
    branch2_global_modes = 32
    branch2_global_width = 32
    branch2_local_type = 'conv'
    branch2_freq_gate_norm_hz = 25.0
    use_kpe = False
    use_trunk_freq_encoding = False           # 会根据 checkpoint 自动覆盖
    trunk_freq_embed_dim = 8
    trunk_freq_num_bands = 3
    trunk_freq_norm_hz = 25.0

    # ==========================================
    # 6. 评估与批处理配置 (Evaluation & Batch)
    # ==========================================
    batch_size = 1600                         # 推理时坐标采样批次大小
    in_channels = 2                           # 波场输入通道数 (实部 + 虚部)
    in_channels_vel = 1                       # 速度模型输入通道数
    input_shape_trunk = (batch_size, in_channels, 1, 2)
    input_shape_branch1 = (batch_size, in_channels_vel, nz, nx)
    input_shape_branch2 = (batch_size, in_channels, nz, nx)

    # ==========================================
    # 7. 微调与域适应 (Fine-Tuning)
    # ==========================================
    if_finetune = False                       # 是否启用域适应微调 (会被 CLI --finetune 覆盖)
    ft_NIter = 1000                           # 微调迭代步数
    ft_lr = 2e-5                              # 微调学习率
    ft_a = 0.                                 # 微调数据 Loss 权重
    ft_b = 1                                  # 微调 PDE Loss 权重
    ft_c = 0.00001                            # 微调正则化 Loss 权重

    # ==========================================
    # 8. 损失函数权重 (Loss Weights)
    # ==========================================
    a = 1                                     # 数据拟合项权重
    b = 1                                     # PDE 物理残差项权重
    c = 0                                     # 正则化项权重
    d = 1                                     # 包络损失项权重

    # ==========================================
    # 9. Positional Encoding
    # ==========================================
    pe_max_scale = 12.0                       # PE 最高频率尺度

    # ==========================================
    # 10. 训练相关占位 (网络构建需要，测试中不使用)
    # ==========================================
    nvel_train = 100
    ny_train = 100
    sampling_mode = 'full_grid'
    halton_sample_ratio = 0.5
    sampling_strategy = 'original'
    use_y_ran = False
    use_epoch_shared_y_ran = True

    def __init__(self, cli_args):
        # CLI 覆盖默认值
        if cli_args.device is not None:
            self.device = cli_args.device
        if cli_args.finetune:
            self.if_finetune = True
        if cli_args.weights:
            self.model_weights_path = resolve_path(cli_args.weights, PROJECT_ROOT)
        if cli_args.data_dir:
            self.test_data_dir = resolve_path(cli_args.data_dir, PROJECT_ROOT)
        if cli_args.weights_dir:
            self.trained_weights_dir = resolve_path(cli_args.weights_dir, PROJECT_ROOT)
        if cli_args.output:
            self.save_doc = resolve_path(cli_args.output, PROJECT_ROOT)
        else:
            self.save_doc = os.path.join(PROJECT_ROOT, f'output_test_{cli_args.dataset}')

        self._cli = cli_args
        self._source_list = [int(s) for s in cli_args.sources.split(',')]
        self._freq_list = [float(f) for f in cli_args.freqs.split(',')]
        self.source_list = self._source_list
        self.freq_list = self._freq_list


def resolve_path(path, base_dir):
    """把用户传入的相对路径解析为项目内路径，绝对路径保持不变。"""
    if not path:
        return path
    path = os.path.expanduser(path)
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def refresh_model_shapes(args):
    """外部数据 PML 裁切后，刷新网络构建所需的 shape 占位。"""
    args.input_shape_trunk = (args.batch_size, args.in_channels, 1, 2)
    args.input_shape_branch1 = (args.batch_size, args.in_channels_vel, args.nz, args.nx)
    args.input_shape_branch2 = (args.batch_size, args.in_channels, args.nz, args.nx)


def find_latest_weight(args):
    """未显式指定 --weights 时，从默认训练输出目录选择 epoch 最大的权重。"""
    weights_dir = resolve_path(getattr(args, 'trained_weights_dir', ''), PROJECT_ROOT)
    pattern = os.path.join(weights_dir, f'{args.filename}_PI_model_*epoch_weights_*.pth')
    candidates = glob.glob(pattern)
    if not candidates:
        return ''

    def epoch_of(path):
        match = re.search(r'(\d+)epoch', os.path.basename(path))
        return int(match.group(1)) if match else -1

    return max(candidates, key=epoch_of)


def resolve_model_path(args):
    """解析显式权重路径；未指定时自动查找最新权重。"""
    model_path = getattr(args, 'model_weights_path', None)
    if model_path:
        return model_path

    model_path = find_latest_weight(args)
    if model_path:
        args.model_weights_path = model_path
        print(f'[*] 未指定 --weights，自动使用最新权重: {model_path}')
        return model_path

    raise ValueError(
        '未指定权重路径，且自动查找失败。请使用 --weights 指定权重，'
        f'或把权重放到 {args.trained_weights_dir}'
    )


def checkpoint_state_dict(model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    return ckpt.get('model_state_dict', ckpt)


def infer_branch2_type_from_checkpoint(model_path, device):
    """根据 checkpoint key 自动识别 branch2 架构，兼容旧实验输出目录。"""
    state_dict = checkpoint_state_dict(model_path, device)
    keys = [key.replace('module.', '', 1) for key in state_dict.keys()]
    if any(key.startswith('branch2_global.') for key in keys):
        return 'hybrid'
    if any(key.startswith('branch2.0.fc0.') for key in keys):
        return 'fno'
    if any(key.startswith('branch2.net.12.') for key in keys):
        return 'conv_deep'
    if any(key.startswith('branch2.net.') for key in keys):
        return 'conv'
    if any(key.startswith('branch2.stem.') or key.startswith('branch2.blocks.') for key in keys):
        return 'resnet'
    return None


def infer_branch1_config_from_checkpoint(model_path, device):
    """根据 checkpoint tensor shape 自动识别 branch1 FNO 的 modes/width。"""
    state_dict = checkpoint_state_dict(model_path, device)
    normalized = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
    config = {}

    fc0_weight = normalized.get('branch1.0.fc0.weight')
    if fc0_weight is not None:
        config['branch1_width'] = int(fc0_weight.shape[0])

    spectral_weight = normalized.get('branch1.0.conv1.weights1')
    if spectral_weight is not None and spectral_weight.ndim >= 4:
        config['branch1_modes'] = int(spectral_weight.shape[2])

    return config


def infer_branch2_fno_config_from_checkpoint(model_path, device):
    """根据 checkpoint tensor shape 自动识别 branch2 FNO 的 modes/width。"""
    state_dict = checkpoint_state_dict(model_path, device)
    normalized = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
    config = {}

    fc0_weight = normalized.get('branch2.0.fc0.weight')
    if fc0_weight is not None:
        config['branch2_width'] = int(fc0_weight.shape[0])

    spectral_weight = normalized.get('branch2.0.conv1.weights1')
    if spectral_weight is not None and spectral_weight.ndim >= 4:
        config['branch2_modes'] = int(spectral_weight.shape[2])

    return config


def infer_trunk_config_from_checkpoint(model_path, device):
    """根据 trunk.fc1 输入维度自动识别是否启用了显式频率编码。"""
    state_dict = checkpoint_state_dict(model_path, device)
    normalized = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
    config = {}

    fc1_weight = normalized.get('trunk.fc1.weight')
    if fc1_weight is None or fc1_weight.ndim != 2:
        return config

    input_dim = int(fc1_weight.shape[1])
    if input_dim > 16:
        config['use_trunk_freq_encoding'] = True
        config['trunk_freq_embed_dim'] = input_dim - 16
        freq_weight = normalized.get('trunk_freq_encoder.0.weight')
        if freq_weight is not None and freq_weight.ndim == 2:
            freq_feature_dim = int(freq_weight.shape[1])
            if freq_feature_dim >= 3 and (freq_feature_dim - 1) % 2 == 0:
                config['trunk_freq_num_bands'] = (freq_feature_dim - 1) // 2
    else:
        config['use_trunk_freq_encoding'] = False

    return config


def load_model_checkpoint(model, model_path, device):
    """兼容单卡与 DDP 保存的 checkpoint/state_dict。"""
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    if any(key.startswith('module.') for key in state_dict):
        state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    return ckpt


def select_eval_device(args):
    """选择可用的评估设备，并避免默认 GPU 编号在新机器上越界。"""
    if not torch.cuda.is_available():
        args.device = torch.device('cpu')
        return args.device

    requested = args.device if isinstance(args.device, int) else 0
    gpu_count = torch.cuda.device_count()
    if requested < 0 or requested >= gpu_count:
        print(f'⚠️ 请求的 GPU cuda:{requested} 不存在，自动改用 cuda:0')
        requested = 0

    args.device = torch.device(f'cuda:{requested}')
    return args.device


# =====================================================================
# 数据加载
# =====================================================================
def load_test_data(args):
    """
    加载外部测试数据，按 source/freq 筛选。

    数据格式 (gen_external_test.py 生成):
        {name}_velocity.npy   (n_total, nz_ext, nx_ext)
        {name}_wavefield.npy  (n_total * N_SRC, 2, nz_ext, nx_ext)  source-major
        {name}_background.npy 同 wavefield
        {name}_freq_used.npy  (n_total,)

    其中 n_total = n_models * n_freqs_all, N_SRC = 5
    wavefield 排序: [src0_sample0, src0_sample1, ..., src1_sample0, ...]
    """
    name = args._cli.dataset
    source_list = args._source_list
    freq_list = args._freq_list
    data_dir = args.test_data_dir
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f'外部测试数据目录不存在: {data_dir}')

    required_files = [
        os.path.join(data_dir, f'{name}_velocity.npy'),
        os.path.join(data_dir, f'{name}_wavefield.npy'),
        os.path.join(data_dir, f'{name}_background.npy'),
        os.path.join(data_dir, f'{name}_freq_used.npy'),
    ]
    missing = [path for path in required_files if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError('外部测试数据文件缺失:\n' + '\n'.join(missing))

    vel = np.load(os.path.join(data_dir, f'{name}_velocity.npy'))
    wf = np.load(os.path.join(data_dir, f'{name}_wavefield.npy'))
    bg = np.load(os.path.join(data_dir, f'{name}_background.npy'))
    freq_all = np.load(os.path.join(data_dir, f'{name}_freq_used.npy'))

    n_total = vel.shape[0]  # n_models * n_freqs_all
    n_src = 5
    nz_ext, nx_ext = vel.shape[1], vel.shape[2]
    freq_unique = np.unique(freq_all).tolist()

    print(f'[*] 数据集: {name}')
    print(f'    原始维度: vel={vel.shape}, wf={wf.shape}, freq={freq_all.shape}')
    print(f'    数据目录: {data_dir}')
    print(f'    包含频率: {freq_unique}')

    # --- PML 裁切 (与训练逻辑一致) ---
    if args.pml:
        pc = args.pml_crop
        if args.boundary_type == 'free_surface':
            z_sl = slice(0, -pc)
        else:
            z_sl = slice(pc, -pc)
        x_sl = slice(pc, -pc)

        vel = vel[:, z_sl, x_sl]
        wf = wf[:, :, z_sl, x_sl]
        bg = bg[:, :, z_sl, x_sl]

        args.nz = vel.shape[1]
        args.nx = vel.shape[2]
        print(f'    PML 裁切后: {vel.shape[1]}×{vel.shape[2]}  (pml_crop={pc})')

    # --- 按 freq 筛选 ---
    freq_mask = np.isin(freq_all, freq_list)
    sample_idx = np.where(freq_mask)[0]
    if len(sample_idx) == 0:
        raise ValueError(f'freq_list={freq_list} 在数据中不存在 (可用: {freq_unique})')

    vel_sel = vel[sample_idx]              # (n_sel, nz, nx)
    freq_sel = freq_all[sample_idx]        # (n_sel,)

    # --- 按 source 筛选 (source-major 布局) ---
    wf_sel = []
    bg_sel = []
    for s in source_list:
        if s < 0 or s >= n_src:
            raise ValueError(f'震源索引 {s} 超出范围 [0, {n_src-1}]')
        s_idx = s * n_total + sample_idx
        wf_sel.append(wf[s_idx])
        bg_sel.append(bg[s_idx])
    wf_sel = np.concatenate(wf_sel, axis=0)  # (n_sources * n_sel, 2, nz, nx)
    bg_sel = np.concatenate(bg_sel, axis=0)

    # 为每个 source 复制对应的 freq
    freq_expanded = np.tile(freq_sel, len(source_list))

    print(f'    筛选: sources={source_list}, freqs={freq_list}')
    print(f'    筛选后: vel={vel_sel.shape}, wf={wf_sel.shape}')

    # 转 tensor
    vel_t = torch.from_numpy(vel_sel).float()
    wf_t = torch.from_numpy(wf_sel).float()
    bg_t = torch.from_numpy(bg_sel).float()
    freq_t = torch.from_numpy(freq_expanded).float()

    return {
        'vel': vel_t,
        'wavefield': wf_t,
        'background': bg_t,
        'freq': freq_t,
        'n_samples': len(sample_idx),
        'n_sources': len(source_list),
    }


# =====================================================================
# 评估与绘图
# =====================================================================
def evaluate_single(args, model, vel, UU0, label, freq_val, device):
    """对单个 (vel, bg) 组合在全网格上推理，返回预测和指标。"""
    model.eval()
    nz, nx = args.nz, args.nx

    # 坐标网格
    grid_z, grid_x = torch.meshgrid(
        torch.arange(nz), torch.arange(nx), indexing='ij')
    y_grid = torch.stack([grid_z.flatten(), grid_x.flatten()], dim=1).float() * args.dh
    loader = DataLoader(TensorDataset(y_grid), batch_size=args.batch_size, shuffle=False)

    vel_dev = vel.to(device)
    UU0_dev = UU0.unsqueeze(0).to(device) if UU0.dim() == 3 else UU0.to(device)
    freq_dev = torch.tensor([freq_val], device=device, dtype=torch.float32)

    pred_parts = []
    with torch.no_grad():
        for batch in loader:
            y_b = batch[0].to(device).unsqueeze(0)
            out = model(vel_dev, y_b, UU0_dev, freq_batch=freq_dev)
            pred_parts.append(out)

    pred = torch.cat(pred_parts, dim=1)
    pred_2d = pred[0].view(nz, nx, 2).cpu().numpy()
    true_2d = label.cpu().numpy().transpose(1, 2, 0)  # (2, nz, nx) -> (nz, nx, 2)

    # PML active 区域裁切 (用于指标计算)
    L = args.pml_active
    if args.boundary_type == 'free_surface':
        z_sl = slice(0, -L) if L > 0 else slice(None)
    else:
        z_sl = slice(L, -L) if L > 0 else slice(None)
    x_sl = slice(L, -L) if L > 0 else slice(None)

    pred_crop = pred_2d[z_sl, x_sl, :]
    true_crop = true_2d[z_sl, x_sl, :] if true_2d.shape[0] == nz else true_2d

    m_r = calculate_regression_metrics(pred_crop[:, :, 0], true_crop[:, :, 0])
    m_i = calculate_regression_metrics(pred_crop[:, :, 1], true_crop[:, :, 1])

    return pred_2d, m_r, m_i


def _crop_entry(entry, args):
    """对单个 entry 做 PML active 裁切，返回 (vel_crop, true_crop, pred_crop)。"""
    L = args.pml_active
    z_sl = slice(0, -L) if args.boundary_type == 'free_surface' and L > 0 else (
        slice(L, -L) if L > 0 else slice(None))
    x_sl = slice(L, -L) if L > 0 else slice(None)

    vel_crop = entry['vel'][z_sl, x_sl] * 1000
    true_crop = entry['true'][z_sl, x_sl, :]
    pred_crop = entry['pred'][z_sl, x_sl, :]
    return vel_crop, true_crop, pred_crop


def _group_by_velocity(results):
    """按速度模型内容分组，返回 [(group_id, [entries])]。"""
    from collections import OrderedDict, defaultdict

    groups = OrderedDict()
    grouped = defaultdict(list)
    for entry in results:
        key = entry['vel'].tobytes()
        if key not in groups:
            groups[key] = len(groups)
        grouped[groups[key]].append(entry)
    return sorted(grouped.items())


def _plot_random_by_model(args, dataset_name, results, epoch_num):
    """random 数据集按速度模型分组，每个模型分别保存 REAL/IMAG 图。"""
    groups = _group_by_velocity(results)

    for model_idx, entries in groups:
        entries_sorted = sorted(entries, key=lambda e: (e['source'], e['freq']))

        for part_idx, suffix in enumerate(['REAL', 'IMAG']):
            n = len(entries_sorted)
            fig, axes = plt.subplots(4, n, figsize=(4.2 * n, 14))
            if n == 1:
                axes = axes[:, np.newaxis]

            src_list = sorted({entry['source'] for entry in entries_sorted})
            fig.suptitle(
                f'{dataset_name} Model {model_idx} src{src_list} {suffix} | Epoch {epoch_num}',
                fontsize=14)

            for col, entry in enumerate(entries_sorted):
                vel_crop, true_crop, pred_crop = _crop_entry(entry, args)
                true_part = true_crop[:, :, part_idx]
                pred_part = pred_crop[:, :, part_idx]
                err_part = true_part - pred_part
                metrics = entry['metrics_real'] if part_idx == 0 else entry['metrics_imag']

                im = axes[0, col].imshow(vel_crop, cmap='jet', aspect='equal')
                axes[0, col].set_title(
                    f'src{entry["source"]}_{entry["freq"]:.0f}Hz\nVelocity', fontsize=9)
                axes[0, col].axis('off')
                fig.colorbar(im, ax=axes[0, col], fraction=0.046)

                vm = max(np.abs(true_part).max(), 1e-12)
                em = max(np.abs(err_part).max(), 1e-12)

                im = axes[1, col].imshow(
                    true_part, cmap='seismic', vmin=-vm, vmax=vm, aspect='equal')
                axes[1, col].set_title(f'True R²={metrics["r2"]:.4f}', fontsize=9)
                axes[1, col].axis('off')
                fig.colorbar(im, ax=axes[1, col], fraction=0.046)

                im = axes[2, col].imshow(
                    pred_part, cmap='seismic', vmin=-vm, vmax=vm, aspect='equal')
                axes[2, col].set_title('Pred', fontsize=9)
                axes[2, col].axis('off')
                fig.colorbar(im, ax=axes[2, col], fraction=0.046)

                im = axes[3, col].imshow(
                    err_part, cmap='bwr', vmin=-em, vmax=em, aspect='equal')
                axes[3, col].set_title(f'Error MSE={metrics["mse"]:.2e}', fontsize=9)
                axes[3, col].axis('off')
                fig.colorbar(im, ax=axes[3, col], fraction=0.046)

            fig.tight_layout(rect=[0, 0.03, 1, 0.94])
            path = os.path.join(
                args.save_doc,
                f'{dataset_name}_model{model_idx}_{suffix}_epoch_{epoch_num}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  已保存 {path}')


def plot_results(args, dataset_name, results, epoch_num):
    """绘制评估结果图。增加速度模型显示，并支持 random 数据集多图保存。"""
    n = len(results)
    if n == 0:
        return

    os.makedirs(args.save_doc, exist_ok=True)

    if args._cli.dataset == 'random':
        _plot_random_by_model(args, dataset_name, results, epoch_num)
        return

    fig_r, ax_r = plt.subplots(4, n, figsize=(4.2 * n, 14))
    fig_i, ax_i = plt.subplots(4, n, figsize=(4.2 * n, 14))
    if n == 1:
        ax_r, ax_i = ax_r[:, np.newaxis], ax_i[:, np.newaxis]

    fig_r.suptitle(f'{dataset_name} REAL | Epoch {epoch_num}', fontsize=14)
    fig_i.suptitle(f'{dataset_name} IMAG | Epoch {epoch_num}', fontsize=14)

    for col, entry in enumerate(results):
        src = entry['source']
        freq = entry['freq']
        m_r, m_i = entry['metrics_real'], entry['metrics_imag']
        tag = f'src{src}_{freq:.0f}Hz'

        vel_crop, true_crop, pred_crop = _crop_entry(entry, args)
        t_r, p_r = true_crop[:, :, 0], pred_crop[:, :, 0]
        t_i, p_i = true_crop[:, :, 1], pred_crop[:, :, 1]
        e_r, e_i = t_r - p_r, t_i - p_i

        for (fig, axes, t, p, e, m, part) in [
            (fig_r, ax_r, t_r, p_r, e_r, m_r, 'real'),
            (fig_i, ax_i, t_i, p_i, e_i, m_i, 'imag'),
        ]:
            im = axes[0, col].imshow(vel_crop, cmap='jet', aspect='equal')
            axes[0, col].set_title(f'{tag}\nVelocity', fontsize=9)
            axes[0, col].axis('off')
            fig.colorbar(im, ax=axes[0, col], fraction=0.046)

            vm = max(np.abs(t).max(), 1e-12)
            em = max(np.abs(e).max(), 1e-12)

            im = axes[1, col].imshow(t, cmap='seismic', vmin=-vm, vmax=vm, aspect='equal')
            axes[1, col].set_title(f'True {part} R²={m["r2"]:.4f}', fontsize=9)
            axes[1, col].axis('off')
            fig.colorbar(im, ax=axes[1, col], fraction=0.046)

            im = axes[2, col].imshow(p, cmap='seismic', vmin=-vm, vmax=vm, aspect='equal')
            axes[2, col].set_title(f'Pred {part}', fontsize=9)
            axes[2, col].axis('off')
            fig.colorbar(im, ax=axes[2, col], fraction=0.046)

            im = axes[3, col].imshow(e, cmap='bwr', vmin=-em, vmax=em, aspect='equal')
            axes[3, col].set_title(f'Error MSE={m["mse"]:.2e}', fontsize=9)
            axes[3, col].axis('off')
            fig.colorbar(im, ax=axes[3, col], fraction=0.046)

    for fig, suffix in [(fig_r, 'REAL'), (fig_i, 'IMAG')]:
        fig.tight_layout(rect=[0, 0.03, 1, 0.94])
        path = os.path.join(args.save_doc, f'{dataset_name}_{suffix}_epoch_{epoch_num}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  已保存 {path}')


# =====================================================================
# 主测试流程
# =====================================================================
def test(args):
    device = select_eval_device(args)
    print(f'\n{"="*60}')
    print(f'PI-DeepONet 外部测试评估 | 设备: {device}')
    print(f'数据集: {args._cli.dataset}  震源: {args._source_list}  频率: {args._freq_list}')
    print(f'网格: nz={args.nz}, nx={args.nx}, dh={args.dh}, boundary={args.boundary_type}')
    print(f'PML: total={args.pml_total}, crop={args.pml_crop}, active={args.pml_active}')
    print(f'{"="*60}')

    # ---- 1. 加载测试数据 ----
    data = load_test_data(args)
    refresh_model_shapes(args)
    vel = data['vel'] / 1000.0          # 归一化, 与训练一致
    vel = vel.unsqueeze(1)
    wf = data['wavefield']
    bg = data['background']
    freq = data['freq']
    n_samples = data['n_samples']
    n_sources = data['n_sources']

    # ---- 2. 加载模型 ----
    model_path = resolve_model_path(args)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'权重文件不存在: {model_path}')

    inferred_branch2 = infer_branch2_type_from_checkpoint(model_path, device)
    if inferred_branch2 and inferred_branch2 != getattr(args, 'branch2_type', None):
        print(f'[*] 根据权重自动切换 branch2_type: {args.branch2_type} -> {inferred_branch2}')
        args.branch2_type = inferred_branch2
    inferred_branch1 = infer_branch1_config_from_checkpoint(model_path, device)
    for name, value in inferred_branch1.items():
        if getattr(args, name, None) != value:
            print(f'[*] 根据权重自动切换 {name}: {getattr(args, name, None)} -> {value}')
            setattr(args, name, value)
    if getattr(args, 'branch2_type', None) == 'fno':
        inferred_branch2_fno = infer_branch2_fno_config_from_checkpoint(model_path, device)
        for name, value in inferred_branch2_fno.items():
            if getattr(args, name, None) != value:
                print(f'[*] 根据权重自动切换 {name}: {getattr(args, name, None)} -> {value}')
                setattr(args, name, value)
    inferred_trunk = infer_trunk_config_from_checkpoint(model_path, device)
    for name, value in inferred_trunk.items():
        if getattr(args, name, None) != value:
            print(f'[*] 根据权重自动切换 {name}: {getattr(args, name, None)} -> {value}')
            setattr(args, name, value)

    model = Pi_DeepONet(args).to(device)
    fno = FNO(args).to(device)
    fno.eval()

    load_model_checkpoint(model, model_path, device)
    model.eval()
    epoch_match = re.search(r'(\d+)epoch', model_path)
    epoch_num = int(epoch_match.group(1)) if epoch_match else 0
    print(f'✅ 权重已加载: {model_path} (epoch={epoch_num})')

    # ---- 3. 微调 (可选) ----
    if args.if_finetune:
        print(f'\n[!] 域适应微调...')
        labels = wf - bg
        # 复制速度场以匹配 source-major 排列: [src0 samples, src1 samples, ...]
        vel_ft = vel.repeat(n_sources, 1, 1, 1).to(device)
        bg_ft = bg.to(device)
        lab_ft = labels.to(device)
        freq_ft = freq.to(device)

        # 构建坐标 dataloader
        grid_z, grid_x = torch.meshgrid(
            torch.arange(args.nz), torch.arange(args.nx), indexing='ij')
        y_grid = torch.stack([grid_z.flatten(), grid_x.flatten()], dim=1).float() * args.dh
        loader_y = DataLoader(TensorDataset(y_grid), batch_size=args.batch_size, shuffle=False)

        model = fine_tuning(args, model, fno, loader_y, vel_ft, bg_ft, lab_ft, freq=freq_ft)
        model.eval()
        print('  微调完成')

    # ---- 4. 逐 (source, freq) 评估 ----
    print(f'\n{"="*60}')
    print(f'开始评估 ({n_sources} sources × {n_samples} samples)')
    print(f'{"="*60}')

    n_total = vel.shape[0]  # n_sources * n_samples
    results = []

    for si, src in enumerate(args._source_list):
        for fi in range(n_samples):
            idx = si * n_samples + fi  # source-major 布局中的索引
            freq_val = freq[idx].item()

            bg_i = bg[idx]                                 # (2, nz, nx)
            label_i = wf[idx] - bg[idx]                    # (2, nz, nx)
            vel_single = vel[fi].unsqueeze(0)              # (1, 1, nz, nx)

            pred_2d, m_r, m_i = evaluate_single(
                args, model, vel_single, bg_i.unsqueeze(0), label_i, freq_val, device)

            results.append({
                'source': src,
                'freq': freq_val,
                'vel': vel_single[0, 0].cpu().numpy(),
                'true': label_i.numpy().transpose(1, 2, 0),  # (nz, nx, 2)
                'pred': pred_2d,
                'metrics_real': m_r,
                'metrics_imag': m_i,
            })

            print(f'  src={src} freq={freq_val:5.1f}Hz | '
                  f'REAL R²={m_r["r2"]:.4f} MSE={m_r["mse"]:.2e} | '
                  f'IMAG R²={m_i["r2"]:.4f} MSE={m_i["mse"]:.2e}')

    # ---- 5. 绘图 ----
    dataset_name = args._cli.dataset.upper()
    plot_results(args, dataset_name, results, epoch_num)

    # 汇总
    r2_r = np.mean([r['metrics_real']['r2'] for r in results])
    r2_i = np.mean([r['metrics_imag']['r2'] for r in results])
    print(f'\n{"="*60}')
    print(f'评估完成: 平均 R² real={r2_r:.4f} imag={r2_i:.4f} ({len(results)} 组)')
    print(f'结果已保存至: {args.save_doc}/')
    print(f'{"="*60}')


def main():
    cli = parse_cli()
    args = ArgsTest(cli)

    if torch.cuda.is_available():
        requested = args.device if isinstance(args.device, int) else 0
        if requested < 0 or requested >= torch.cuda.device_count():
            requested = 0
        mem = torch.cuda.get_device_properties(requested).total_memory / (1024**3)
        print(f'GPU: {torch.cuda.get_device_name(requested)} ({mem:.1f}GB)')

    test(args)


if __name__ == '__main__':
    print('*******************************************')
    print('       PI-DeepONet External Test           ')
    print('*******************************************')
    main()
