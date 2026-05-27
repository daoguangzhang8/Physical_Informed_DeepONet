import numpy as np
from scipy.sparse import spdiags, coo_matrix, csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter, zoom
from scipy.io import savemat
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time as _time
import gc

from optimal_Parameters import optimal_Parameters
from getA9_PML import getA9_PML
from getA25_PML import getA25_PML
from getFPML import getFPML
from getCPML import getCPML

# =====================================================================
# 全局参数配置
# =====================================================================
# 差分格式: '9pt' | '25pt'
STENCIL_TYPE = '9pt'

# 网格细化
TIMES = 4                # 细化倍数: 新网格数 = 70 * TIMES, h = 40 / TIMES
DOWNSAMPLE_TIMES = 2     # 降采样目标倍数; None=不降采样

N_MODELS = 2000          # 正演模型数量
F0 = 10                  # 参考频率 (Hz)
Q = 75                   # 品质因子

# 三组频率配置: (频率池, 存储目录名, 文件名中频率标签)
FREQ_CONFIGS = [
    ([25, 23, 22, 21, 19], 'freq_18to25', 'freq18to25'),
    ([12, 13, 15, 17, 18], 'freq_12to18', 'freq12to18'),
    ([11,  9,  7,  5,  3], 'freq_3to11',  'freq3to11'),
]

# 震源 x 坐标 (原始位置, 后续乘 TIMES)
SRC_X_ORIG = [15, 30, 45, 60, 75]
N_SRC = len(SRC_X_ORIG)

# 数据根路径
DATA_ROOT = '/home/sharedata/zdg'
SAVE_ROOT = os.path.join(DATA_ROOT, 'multifreq')
MERGED_SAVE_ROOT = os.path.join(DATA_ROOT, 'multifreq_merged1')
MERGE_STAGE_CONFIGS = [
    ('freq_3to11', 'freq3to11'),
    ('freq_12to18', 'freq12to18'),
    ('freq_18to25', 'freq18to25'),
]

# =====================================================================
# 辅助函数
# =====================================================================
def load_velocity_models():
    v0 = sio.loadmat(os.path.join(DATA_ROOT, 'feature5_curve.mat'))['feature']
    v1 = np.load(os.path.join(DATA_ROOT, 'merged_velocity_models.npy'))
    return np.concatenate([v0, v1], axis=0)


def build_source_terms(nz, nx, n_pml, times):
    Ls1 = [0 + n_pml[0, 1]] * N_SRC
    Ls2 = [x * times for x in SRC_X_ORIG]
    b0 = csr_matrix((nz * nx, N_SRC))
    for i in range(N_SRC):
        bb = np.zeros((nz, nx))
        bb[Ls2[i], Ls1[i]] = 1
        bb = np.reshape(bb, (nz * nx, 1), order='F')
        b0[bb.nonzero()[0], i] = 1
    return [csc_matrix(2 * 0.25 * b0), csc_matrix(2 * 0.25 * b0)]


def interp_velocity_models(vel_original, model_indices, times, sigma=3.0):
    vel_interp = []
    for i in model_indices:
        v = zoom(vel_original[i], times, order=3)
        v = gaussian_filter(v, sigma * times)
        vel_interp.append(v)
    return vel_interp


def solve_stage(freq_list, vel_interp, model_indices, n, n_pml, nz, nx,
                h, FPML, bb, downsample_times, times):
    """对一组频率进行正演，返回 (UU, UU0, vel_out, freq_used)。"""
    n_models = len(model_indices)
    vel_out = np.zeros((n_models, nx, nz))
    freq_used = []

    # 每个震源独立收集: list of [n_models] 个 [2, nz, nx]
    u_all = [[] for _ in range(N_SRC)]
    u0_all = [[] for _ in range(N_SRC)]

    for idx in range(n_models):
        freq = np.random.choice(freq_list)
        # freq = 25
        freq_used.append(freq)

        # 衰减系数
        alpha = 1 / Q
        alpha = 0.0
        rhot = (1 - alpha / np.pi * np.log(freq / 50) - 1j * alpha / 2) ** 2

        v = vel_interp[idx]
        v_min, v_max = np.min(v), np.max(v)
        Gmin = v_min / (h * freq)

        b, c, d, e = optimal_Parameters(Gmin, v_max / (h * freq))
        b, c, d, e = float(b), float(c), float(d), float(e)
        print(f'  [{idx+1}/{n_models}] freq={freq}Hz, v=[{v_min:.0f},{v_max:.0f}], Gmin={Gmin:.2f}')

        # 均匀背景
        v0 = np.ones((n[1], n[0])) * 1500
        mv0 = rhot / (np.reshape(v0, (n[0] * n[1], 1), order='F') / 1000) ** 2
        mv0 = np.reshape(FPML * mv0, (nz, nx))

        vel_out[idx] = np.reshape(
            FPML * np.reshape(v, (n[0] * n[1], 1), order='F'), (nz, nx)
        ).T

        mv = rhot / (np.reshape(v, (n[0] * n[1], 1), order='F') / 1000) ** 2
        mv = np.reshape(FPML * mv, (nz, nx))

        # 构建系数矩阵并求解
        if STENCIL_TYPE == '25pt':
            A, _, _, _ = getA25_PML(n_pml, nz, nx, freq, F0, h, mv)
            A0, _, _, _ = getA25_PML(n_pml, nz, nx, freq, F0, h, mv0)
        else:
            A, _, _, _ = getA9_PML(n_pml, nz, nx, freq, F0, h, mv, b, c, d, e)
            A0, _, _, _ = getA9_PML(n_pml, nz, nx, freq, F0, h, mv0, b, c, d, e)

        d0 = spsolve(csc_matrix(A), bb[0]).toarray()
        d00 = spsolve(csc_matrix(A0), bb[0]).toarray()

        for s in range(N_SRC):
            ds = np.reshape(d0[:, s], (nx, nz))
            u_all[s].append([np.real(ds), np.imag(ds)])
            ds0 = np.reshape(d00[:, s], (nx, nz))
            u0_all[s].append([np.real(ds0), np.imag(ds0)])

    # 合并: [N_SRC * n_models, 2, nz, nx]
    UU = np.concatenate([np.array(u) for u in u_all], axis=0)
    UU0 = np.concatenate([np.array(u) for u in u0_all], axis=0)

    # 降采样
    if downsample_times is not None and times > downsample_times:
        stride = times // downsample_times
        UU = UU[:, :, ::stride, ::stride]
        UU0 = UU0[:, :, ::stride, ::stride]
        vel_out = vel_out[:, ::stride, ::stride]
        print(f'  降采样 stride={stride}: -> {UU.shape}')

    return UU, UU0, vel_out, np.array(freq_used)


def save_stage(dir_name, freq_tag, UU, UU0, vel_out, freq_used):
    save_dir = os.path.join(SAVE_ROOT, dir_name)
    os.makedirs(save_dir, exist_ok=True)
    prefix = f'freesurface_{freq_tag}_5sources'

    np.save(os.path.join(save_dir, f'{prefix}_wavefield.npy'), UU)
    np.save(os.path.join(save_dir, f'{prefix}_background.npy'), UU0)
    np.save(os.path.join(save_dir, f'{prefix}_velocity.npy'), vel_out)
    np.save(os.path.join(save_dir, f'{prefix}_freq_used.npy'), freq_used)
    print(f'  已保存到 {save_dir}/')


def print_freq_stats(freq_tag, freq_list, freq_used):
    freq_array = np.array(freq_used)
    print(f'  频率统计 ({freq_tag}):')
    for f in freq_list:
        count = np.sum(freq_array == f)
        print(f'    {f:2d} Hz: {count:4d} ({count/len(freq_array)*100:.1f}%)')


def visualize_last_stage(dir_name, freq_tag):
    """从磁盘重新加载最后一阶段数据用于可视化，不占用主循环内存。"""
    save_dir = os.path.join(SAVE_ROOT, dir_name)
    prefix = f'freesurface_{freq_tag}_5sources'
    UU = np.load(os.path.join(save_dir, f'{prefix}_wavefield.npy'), mmap_mode='r')
    UU0 = np.load(os.path.join(save_dir, f'{prefix}_background.npy'), mmap_mode='r')
    vel = np.load(os.path.join(save_dir, f'{prefix}_velocity.npy'), mmap_mode='r')

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    idx = 0

    def adaptive_lim(data, pct=99):
        mx = np.percentile(np.abs(data), pct)
        return -mx, mx

    fields = [
        (axes[0, 0], UU[idx, 0], 'seismic', 'Wavefield Real'),
        (axes[0, 1], UU0[idx, 0], 'seismic', 'Background Real'),
        (axes[0, 2], UU[idx, 0] - UU0[idx, 0], 'seismic', 'Scattered Field'),
        (axes[1, 0], UU[idx, 1], 'seismic', 'Wavefield Imag'),
        (axes[1, 1], vel[idx], 'viridis', 'Velocity Model'),
        (axes[1, 2], np.abs(UU[idx, 0]) / (np.abs(UU0[idx, 0]) + 1e-10), 'hot', 'Scattered/BG Ratio'),
    ]
    for ax, data, cmap, title in fields:
        if cmap == 'viridis':
            vmin, vmax = data.min(), data.max()
        elif cmap == 'hot':
            vmin, vmax = 0, np.percentile(data, 99)
        else:
            vmin, vmax = adaptive_lim(data)
        im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'{title}\n[{vmin:.1f}, {vmax:.1f}]')
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    fig.suptitle(f'Stencil: {STENCIL_TYPE}, Stage: {freq_tag}, Grid: {40/TIMES}m', fontsize=12)
    plt.subplots_adjust(top=0.88)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'可视化已保存到 {out_path}')


def merge_multifreq_stages(src_root=SAVE_ROOT, dst_root=MERGED_SAVE_ROOT):
    """
    合并三个频段阶段数据，生成训练脚本直接使用的 multifreq_merged1 数据。

    输出排列与训练 dataloader 的 n_freq_ranges > 1 逻辑保持一致：
      velocity/freq: [freq0_vel..., freq1_vel..., freq2_vel...]
      background/wavefield: [freq0_src0_vel..., freq0_src1_vel..., ..., freq2_src4_vel...]
    """
    print('\n' + '=' * 60)
    print('合并三阶段多频数据')
    print('=' * 60)

    os.makedirs(dst_root, exist_ok=True)

    for dir_name, freq_tag in MERGE_STAGE_CONFIGS:
        stage_dir = os.path.join(src_root, dir_name)
        prefix = f'freesurface_{freq_tag}_5sources'
        vel = np.load(os.path.join(stage_dir, f'{prefix}_velocity.npy'), mmap_mode='r')
        bg = np.load(os.path.join(stage_dir, f'{prefix}_background.npy'), mmap_mode='r')
        wf = np.load(os.path.join(stage_dir, f'{prefix}_wavefield.npy'), mmap_mode='r')
        freq = np.load(os.path.join(stage_dir, f'{prefix}_freq_used.npy'), mmap_mode='r')
        print(f'{dir_name}: vel={vel.shape}, bg={bg.shape}, wf={wf.shape}, freq={freq.shape}')

    first_dir, first_tag = MERGE_STAGE_CONFIGS[0]
    first_prefix = f'freesurface_{first_tag}_5sources'
    vel = np.load(os.path.join(src_root, first_dir, f'{first_prefix}_velocity.npy'))
    n_vel = vel.shape[0]
    n_stages = len(MERGE_STAGE_CONFIGS)

    for dir_name, freq_tag in MERGE_STAGE_CONFIGS[1:]:
        prefix = f'freesurface_{freq_tag}_5sources'
        stage_vel = np.load(os.path.join(src_root, dir_name, f'{prefix}_velocity.npy'))
        if not np.array_equal(vel, stage_vel):
            raise ValueError(f'{dir_name} velocity 与 {first_dir} 不一致，不能直接合并')

    vel_merged = np.tile(vel, (n_stages, 1, 1))
    freq_parts = []
    for dir_name, freq_tag in MERGE_STAGE_CONFIGS:
        prefix = f'freesurface_{freq_tag}_5sources'
        freq_parts.append(np.load(os.path.join(src_root, dir_name, f'{prefix}_freq_used.npy')))
    freq_merged = np.concatenate(freq_parts, axis=0)

    for field_name in ['background', 'wavefield']:
        stage_fields = []
        for dir_name, freq_tag in MERGE_STAGE_CONFIGS:
            prefix = f'freesurface_{freq_tag}_5sources'
            stage_fields.append(np.load(os.path.join(src_root, dir_name, f'{prefix}_{field_name}.npy')))

        parts = []
        for stage_field in stage_fields:
            for src_idx in range(N_SRC):
                start = src_idx * n_vel
                parts.append(stage_field[start:start + n_vel])

        merged = np.concatenate(parts, axis=0)
        out_path = os.path.join(dst_root, f'freesurface_full_5sources_{field_name}.npy')
        np.save(out_path, merged)
        print(f'{field_name}: saved -> {out_path}, shape={merged.shape}')

        del stage_fields, parts, merged
        gc.collect()

    vel_path = os.path.join(dst_root, 'freesurface_full_5sources_velocity.npy')
    np.save(vel_path, vel_merged)
    print(f'velocity: saved -> {vel_path}, shape={vel_merged.shape}')

    freq_path = os.path.join(dst_root, 'freesurface_full_5sources_freq_used.npy')
    np.save(freq_path, freq_merged)
    print(f'freq_used: saved -> {freq_path}, shape={freq_merged.shape}')

    print('\n=== 合并结果验证 ===')
    vel_v = np.load(vel_path, mmap_mode='r')
    bg_v = np.load(os.path.join(dst_root, 'freesurface_full_5sources_background.npy'), mmap_mode='r')
    wf_v = np.load(os.path.join(dst_root, 'freesurface_full_5sources_wavefield.npy'), mmap_mode='r')
    fq_v = np.load(freq_path, mmap_mode='r')
    print(f'velocity:   {vel_v.shape}')
    print(f'background: {bg_v.shape}')
    print(f'wavefield:  {wf_v.shape}')
    print(f'freq_used:  {fq_v.shape}')

    if vel_v.shape[0] != fq_v.shape[0]:
        raise ValueError(f'vel 第一维 {vel_v.shape[0]} 与 freq 第一维 {fq_v.shape[0]} 不匹配')
    if bg_v.shape[0] != vel_v.shape[0] * N_SRC:
        raise ValueError(f'bg 第一维 {bg_v.shape[0]} != vel 第一维 {vel_v.shape[0]} * {N_SRC}')
    if wf_v.shape != bg_v.shape:
        raise ValueError(f'wavefield shape {wf_v.shape} 与 background shape {bg_v.shape} 不匹配')

    print('合并完成:', dst_root)


# =====================================================================
# 主流程
# =====================================================================
def main():
    t_total = _time.time()

    # 网格参数
    nz_base = nx_base = 70 * TIMES
    h = 40 / TIMES
    Lpml = 10 * TIMES

    n = np.array([nx_base, nz_base])
    n_pml = np.array([[Lpml, 0], [Lpml, Lpml]])
    ne = n + np.sum(n_pml, axis=0)
    nz, nx = ne[0], ne[1]

    print(f'网格: {nz}x{nx}, h={h}m, PML={Lpml}, stencil={STENCIL_TYPE}')

    # PML 吸收系数 (所有阶段共用)
    FPML = getFPML(n_pml, n)
    bb = build_source_terms(nz, nx, n_pml, TIMES)

    # 加载 & 插值速度模型
    vel_original = load_velocity_models()
    print(f'速度模型池: {vel_original.shape}')

    np.random.seed(42)
    model_indices = np.random.choice(vel_original.shape[0], size=N_MODELS, replace=False)

    print('插值速度模型...')
    vel_interp = interp_velocity_models(vel_original, model_indices, TIMES)
    del vel_original
    gc.collect()
    print(f'插值完成: {len(vel_interp)} 个模型')

    # 逐阶段正演
    for cfg_idx, (freq_list, dir_name, freq_tag) in enumerate(FREQ_CONFIGS):
        t_stage = _time.time()
        print(f'\n{"="*60}')
        print(f'阶段 [{cfg_idx+1}/{len(FREQ_CONFIGS)}]: {freq_tag} ({freq_list} Hz)')
        print(f'{"="*60}')

        UU, UU0, vel_out, freq_used = solve_stage(
            freq_list, vel_interp, model_indices, n, n_pml, nz, nx,
            h, FPML, bb, DOWNSAMPLE_TIMES, TIMES,
        )

        print_freq_stats(freq_tag, freq_list, freq_used)
        save_stage(dir_name, freq_tag, UU, UU0, vel_out, freq_used)

        # 释放本阶段大数组
        del UU, UU0, vel_out, freq_used
        gc.collect()

        print(f'本阶段耗时: {_time.time()-t_stage:.1f}s')

    # 释放插值模型 & PML
    del vel_interp, FPML, bb
    gc.collect()

    # 可视化 (从磁盘加载最后一阶段)
    _, last_dir, last_tag = FREQ_CONFIGS[-1]
    visualize_last_stage(last_dir, last_tag)

    # 三阶段数据全部生成后，直接合并为训练使用的数据集
    merge_multifreq_stages()

    print(f'\n全部完成, 总耗时: {_time.time()-t_total:.1f}s')


if __name__ == '__main__':
    main()
