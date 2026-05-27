"""
外部测试数据生成脚本

基于 modeling.py 框架，为 Marmousi、Overthrust 和随机抽取的速度模型
在 5 个频率 (5, 10, 15, 20, 25 Hz) 下生成 FDFD 正演数据。

参数与 config.py (Physical_Informed_DeepONet) 一致:
  TIMES=4, DOWNSAMPLE_TIMES=2 → 最终网格 160×180, h=20m
  free_surface 边界, PML=20

输出格式与训练数据一致:
  {prefix}_velocity.npy   (n_samples, nx_ds, nz_ds)
  {prefix}_background.npy (n_samples * N_SRC, 2, nx_ds, nz_ds)
  {prefix}_wavefield.npy  (n_samples * N_SRC, 2, nx_ds, nz_ds)
  {prefix}_freq_used.npy  (n_samples,)

  其中 n_samples = n_models * len(TEST_FREQS), 每个 (模型, 频率) 为一个样本。
  波场排序: source-major → [src0_m0_f0, src0_m0_f1, ..., src1_m0_f0, ...]
"""

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter, zoom
import scipy.io as sio
import os
import time as _time
import gc

from optimal_Parameters import optimal_Parameters
from getA9_PML import getA9_PML
from getFPML import getFPML
from getCPML import getCPML

# =====================================================================
# 参数配置 (与训练 modeling.py / config.py 一致)
# =====================================================================
STENCIL_TYPE = '9pt'
TIMES = 4                 # 细化倍数: 70*4=280, h=10m
DOWNSAMPLE_TIMES = 2      # 降采样到 140×140, h=20m

F0 = 10                   # 参考频率 (Hz)
Q = 75                    # 品质因子 (实际 alpha=0, 不衰减)

TEST_FREQS = [5, 10, 15, 20, 25]
N_SRC = 5
SRC_X_ORIG = [15, 30, 45, 60, 75]
SRC_X_ORIG_ALT = [17, 32, 47, 62, 73]   # 近似震源位置 (偏移+2)

N_RANDOM_MODELS = 10
RANDOM_SEED = 42

DATA_ROOT = '/home/sharedata/zdg'
SAVE_ROOT = os.path.join(DATA_ROOT, 'external_test')


# =====================================================================
# 辅助函数 (复用 modeling.py 逻辑)
# =====================================================================
def build_source_terms(nz, nx, n_pml, times, src_x_orig=None):
    if src_x_orig is None:
        src_x_orig = SRC_X_ORIG
    Ls1 = [0 + n_pml[0, 1]] * N_SRC
    Ls2 = [x * times for x in src_x_orig]
    b0 = csr_matrix((nz * nx, N_SRC))
    for i in range(N_SRC):
        bb = np.zeros((nz, nx))
        bb[Ls2[i], Ls1[i]] = 1
        bb = np.reshape(bb, (nz * nx, 1), order='F')
        b0[bb.nonzero()[0], i] = 1
    return [csc_matrix(2 * 0.25 * b0), csc_matrix(2 * 0.25 * b0)]


def interp_velocity(vel_model, times, sigma=3.0):
    """单个速度模型插值到高分辨率网格。"""
    v = zoom(vel_model, times, order=3)
    v = gaussian_filter(v, sigma * times)
    return v


def solve_all_freqs(vel_interp, freq_list, n, n_pml, nz, nx,
                    h, FPML, bb, downsample_times, times):
    """
    对每个速度模型在所有指定频率下正演。
    每个 (模型, 频率) 组合生成一条记录，格式与训练数据一致。

    Returns:
        UU:      (n_total * N_SRC, 2, nz_ds, nx_ds)  全波场
        UU0:     同上                                背景波场
        vel_out: (n_total, nx_ds, nz_ds)             速度场
        freq_used: (n_total,)                         频率标签
    """
    n_models = len(vel_interp)
    n_freqs = len(freq_list)
    n_total = n_models * n_freqs

    vel_out = np.zeros((n_total, nx, nz))
    freq_used = []

    u_all = [[] for _ in range(N_SRC)]
    u0_all = [[] for _ in range(N_SRC)]

    sample_idx = 0
    for model_idx in range(n_models):
        v = vel_interp[model_idx]

        for freq in freq_list:
            alpha = 0.0  # 无衰减, 与训练数据一致
            rhot = (1 - alpha / np.pi * np.log(freq / 50) - 1j * alpha / 2) ** 2

            v_min, v_max = np.min(v), np.max(v)
            Gmin = v_min / (h * freq)
            Gmax = v_max / (h * freq)
            b_c, c_c, d_c, e_c = optimal_Parameters(Gmin, Gmax)
            b_c, c_c, d_c, e_c = float(b_c), float(c_c), float(d_c), float(e_c)

            print(f'  [{model_idx+1}/{n_models}] freq={freq}Hz, '
                  f'v=[{v_min:.0f},{v_max:.0f}], Gmin={Gmin:.2f}')

            # 均匀背景 (1500 m/s)
            v0 = np.ones((n[1], n[0])) * 1500
            mv0 = rhot / (np.reshape(v0, (n[0] * n[1], 1), order='F') / 1000) ** 2
            mv0 = np.reshape(FPML * mv0, (nz, nx))

            vel_out[sample_idx] = np.reshape(
                FPML * np.reshape(v, (n[0] * n[1], 1), order='F'), (nz, nx)
            ).T

            mv = rhot / (np.reshape(v, (n[0] * n[1], 1), order='F') / 1000) ** 2
            mv = np.reshape(FPML * mv, (nz, nx))

            # 构建系数矩阵并求解
            A, _, _, _ = getA9_PML(n_pml, nz, nx, freq, F0, h, mv, b_c, c_c, d_c, e_c)
            A0, _, _, _ = getA9_PML(n_pml, nz, nx, freq, F0, h, mv0, b_c, c_c, d_c, e_c)

            d0 = spsolve(csc_matrix(A), bb[0]).toarray()
            d00 = spsolve(csc_matrix(A0), bb[0]).toarray()

            for s in range(N_SRC):
                ds = np.reshape(d0[:, s], (nx, nz))
                u_all[s].append([np.real(ds), np.imag(ds)])
                ds0 = np.reshape(d00[:, s], (nx, nz))
                u0_all[s].append([np.real(ds0), np.imag(ds0)])

            freq_used.append(freq)
            sample_idx += 1

    # 合并: source-major → [src0_*_all_samples, src1_*_all_samples, ...]
    UU = np.concatenate([np.array(u) for u in u_all], axis=0)
    UU0 = np.concatenate([np.array(u) for u in u0_all], axis=0)

    # 降采样
    if downsample_times is not None and times > downsample_times:
        stride = times // downsample_times
        UU = UU[:, :, ::stride, ::stride]
        UU0 = UU0[:, :, ::stride, ::stride]
        vel_out = vel_out[:, ::stride, ::stride]
        print(f'  降采样 stride={stride}: UU -> {UU.shape}, vel -> {vel_out.shape}')

    return UU, UU0, vel_out, np.array(freq_used)


def save_dataset(prefix, UU, UU0, vel_out, freq_used):
    os.makedirs(SAVE_ROOT, exist_ok=True)
    np.save(os.path.join(SAVE_ROOT, f'{prefix}_wavefield.npy'), UU)
    np.save(os.path.join(SAVE_ROOT, f'{prefix}_background.npy'), UU0)
    np.save(os.path.join(SAVE_ROOT, f'{prefix}_velocity.npy'), vel_out)
    np.save(os.path.join(SAVE_ROOT, f'{prefix}_freq_used.npy'), freq_used)
    print(f'  已保存到 {SAVE_ROOT}/{prefix}_*.npy')
    print(f'    velocity:   {vel_out.shape}')
    print(f'    wavefield:  {UU.shape}')
    print(f'    background: {UU0.shape}')
    print(f'    freq_used:  {freq_used.shape}  {freq_used.tolist()}')


# =====================================================================
# 主流程
# =====================================================================
def run_marmousi(FPML, bb, n, n_pml, nz, nx, h, prefix='marmousi'):
    print(f'\n{"="*60}')
    print(f'Marmousi (1 model × {len(TEST_FREQS)} freqs) [{prefix}]')
    print(f'{"="*60}')

    vel = np.load(os.path.join(DATA_ROOT, 'Marmousi_vel_70_70.npy'))
    print(f'  原始: {vel.shape}')
    vel_interp = [interp_velocity(vel, TIMES)]
    print(f'  插值后: {vel_interp[0].shape}')

    t0 = _time.time()
    UU, UU0, vel_out, freq_used = solve_all_freqs(
        vel_interp, TEST_FREQS, n, n_pml, nz, nx,
        h, FPML, bb, DOWNSAMPLE_TIMES, TIMES
    )
    save_dataset(prefix, UU, UU0, vel_out, freq_used)
    print(f'  耗时: {_time.time()-t0:.1f}s')


def run_overthrust(FPML, bb, n, n_pml, nz, nx, h):
    print(f'\n{"="*60}')
    print(f'Overthrust (1 model × {len(TEST_FREQS)} freqs)')
    print(f'{"="*60}')

    vel = np.load(os.path.join(DATA_ROOT, 'Overthrust_vel_70_70.npy'))
    print(f'  原始: {vel.shape}')
    vel_interp = [interp_velocity(vel, TIMES)]
    print(f'  插值后: {vel_interp[0].shape}')

    t0 = _time.time()
    UU, UU0, vel_out, freq_used = solve_all_freqs(
        vel_interp, TEST_FREQS, n, n_pml, nz, nx,
        h, FPML, bb, DOWNSAMPLE_TIMES, TIMES
    )
    save_dataset('overthrust', UU, UU0, vel_out, freq_used)
    print(f'  耗时: {_time.time()-t0:.1f}s')


def run_random(FPML, bb, n, n_pml, nz, nx, h):
    print(f'\n{"="*60}')
    print(f'随机模型 ({N_RANDOM_MODELS} models × {len(TEST_FREQS)} freqs)')
    print(f'{"="*60}')

    vel_curve = sio.loadmat(os.path.join(DATA_ROOT, 'feature5_curve.mat'))['feature']
    vel_merged = np.load(os.path.join(DATA_ROOT, 'merged_velocity_models.npy'))
    vel_all = np.concatenate([vel_curve, vel_merged], axis=0)
    print(f'  模型池: {vel_all.shape[0]} 个')

    np.random.seed(RANDOM_SEED)
    indices = np.sort(np.random.choice(vel_all.shape[0], N_RANDOM_MODELS, replace=False))
    print(f'  抽取索引: {indices.tolist()}')

    vel_interp = []
    for idx in indices:
        vel_interp.append(interp_velocity(vel_all[idx], TIMES))
    print(f'  插值完成: {len(vel_interp)} 个, 形状 {vel_interp[0].shape}')

    t0 = _time.time()
    UU, UU0, vel_out, freq_used = solve_all_freqs(
        vel_interp, TEST_FREQS, n, n_pml, nz, nx,
        h, FPML, bb, DOWNSAMPLE_TIMES, TIMES
    )
    save_dataset('random', UU, UU0, vel_out, freq_used)
    np.save(os.path.join(SAVE_ROOT, 'random_model_indices.npy'), indices)
    print(f'  耗时: {_time.time()-t0:.1f}s')


def main():
    import sys
    valid_types = ('marmousi', 'overthrust', 'random', 'marmousi_alt')
    if len(sys.argv) < 2 or sys.argv[1] not in valid_types:
        print(f'用法: python gen_external_test.py <{"|".join(valid_types)}>')
        sys.exit(1)

    model_type = sys.argv[1]
    t_total = _time.time()

    # 网格参数 (与训练 modeling.py 完全一致)
    nz_base = nx_base = 70 * TIMES          # 280
    h = 40 / TIMES                           # 10
    Lpml = 10 * TIMES                        # 40

    n = np.array([nx_base, nz_base])         # [280, 280]
    n_pml = np.array([[Lpml, 0], [Lpml, Lpml]])
    ne = n + np.sum(n_pml, axis=0)           # [360, 320]
    nz, nx = ne[0], ne[1]

    stride = TIMES // DOWNSAMPLE_TIMES if DOWNSAMPLE_TIMES else 1
    print(f'正演网格: {nz}×{nx}, h={h}m, PML={Lpml}, stencil={STENCIL_TYPE}')
    print(f'降采样后: {nz // stride}×{nx // stride} (stride={stride})')
    print(f'测试频率: {TEST_FREQS} Hz')
    print(f'生成目标: {model_type}')

    FPML = getFPML(n_pml, n)
    src_positions = SRC_X_ORIG_ALT if model_type == 'marmousi_alt' else SRC_X_ORIG
    bb = build_source_terms(nz, nx, n_pml, TIMES, src_x_orig=src_positions)

    if model_type in ('marmousi', 'marmousi_alt'):
        run_marmousi(FPML, bb, n, n_pml, nz, nx, h, prefix=model_type)
    elif model_type == 'overthrust':
        run_overthrust(FPML, bb, n, n_pml, nz, nx, h)
    elif model_type == 'random':
        run_random(FPML, bb, n, n_pml, nz, nx, h)

    print(f'\n完成, 总耗时: {_time.time()-t_total:.1f}s')
    print(f'数据保存位置: {SAVE_ROOT}')


if __name__ == '__main__':
    main()
