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
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

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
TIMES = 2                # 细化倍数: 新网格数 = 140 * TIMES, h = 20 / TIMES
DOWNSAMPLE_TIMES = 2     # 1/None = 不降采样; >1 = 输出时按该 stride 降采样

N_MODELS_PER_CATEGORY = 500  # 每类速度模型抽取数量, 总计 = 该值 × 类别数
INTERP_SIGMA = 3.0           # 插值后高斯平滑核宽度 (像素)
F0 = 10                  # 参考频率 (Hz)
Q = 75                   # 品质因子

# 三组频率配置: (频率池, 存储目录名, 文件名中频率标签)
# 频率维度顺序固定为: 低频 -> 中频 -> 高频
FREQ_CONFIGS = [
    ([11,  9,  7,  5,  3], 'freq_3to11',  'freq3to11'),
    ([12, 13, 15, 17, 18], 'freq_12to18', 'freq12to18'),
    ([25, 23, 22, 21, 19], 'freq_18to25', 'freq18to25'),
]

# 震源坐标 (基于 140 点基础网格, 后续乘 TIMES)
SRC_X_ORIG = [30, 60, 90, 120, 150]
N_SRC = len(SRC_X_ORIG)
FIELD_DTYPE = np.float32
VELOCITY_DTYPE = np.float32
FREQ_DTYPE = np.float32
FREQ_RANDOM_SEED = 42
N_WORKERS = int(os.environ.get('MODELING_N_WORKERS', '1'))

# 数据根路径
DATA_ROOT = '/home/sharedata/zdg'
SAVE_ROOT = os.path.join(DATA_ROOT, 'multifreq')
MERGED_SAVE_ROOT = os.path.join(DATA_ROOT, 'multifreq_merged1')
DS2_SAVE_ROOT = os.path.join(DATA_ROOT, 'multifreq_merged1_ds2')
GENERATE_DS2 = True
DS2_STRIDE = 2
DS2_CHUNK_SIZE = 25
VELOCITY_BANK_ROOT = os.path.join(DATA_ROOT, 'velocity_banks')
VELOCITY_BANKS = [
    ('flat_layers', 'flat_layers.npy'),
    ('flat_layers_2', 'flat_layers_2.npy'),
    ('fold', 'fold.npy'),
    ('fault', 'fault.npy'),
    ('salt', 'salt.npy'),
]
MERGE_STAGE_CONFIGS = [
    ('freq_3to11', 'freq3to11'),
    ('freq_12to18', 'freq12to18'),
    ('freq_18to25', 'freq18to25'),
]

_WORKER_STATE = {}

# =====================================================================
# 辅助函数
# =====================================================================
def get_velocity_bank_paths():
    return [
        (category, os.path.join(VELOCITY_BANK_ROOT, filename))
        for category, filename in VELOCITY_BANKS
    ]


def load_velocity_banks():
    """
    加载多类别速度模型库，并返回:
      velocity_models: [N_total, Z, X]
      model_category: [N_total] int category id
      model_source_index: [N_total] local index in source npy
      category_names: list[str]

    若五类 velocity bank 文件不存在，则回退到旧数据源。
    """
    bank_paths = get_velocity_bank_paths()
    if not all(os.path.exists(path) for _, path in bank_paths):
        missing = [path for _, path in bank_paths if not os.path.exists(path)]
        print('未找到完整 velocity_banks，回退到旧速度模型源:')
        for path in missing:
            print(f'  missing: {path}')
        vel = load_velocity_models()
        model_category = np.zeros(vel.shape[0], dtype=np.int64)
        model_source_index = np.arange(vel.shape[0], dtype=np.int64)
        return vel, model_category, model_source_index, ['legacy']

    velocities = []
    categories = []
    source_indices = []
    category_names = []
    for category_id, (category_name, path) in enumerate(bank_paths):
        arr = np.load(path, mmap_mode='r')
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim != 3:
            raise ValueError(f'velocity bank {path} shape {arr.shape} 不符合 [N, Z, X]')
        arr_np = np.asarray(arr, dtype=VELOCITY_DTYPE)
        velocities.append(arr_np)
        categories.append(np.full(arr_np.shape[0], category_id, dtype=np.int64))
        source_indices.append(np.arange(arr_np.shape[0], dtype=np.int64))
        category_names.append(category_name)
        print(f'加载 velocity bank: {category_name}, shape={arr_np.shape}, path={path}')

    return (
        np.concatenate(velocities, axis=0),
        np.concatenate(categories, axis=0),
        np.concatenate(source_indices, axis=0),
        category_names,
    )


def save_category_metadata(dst_root, selected_category, selected_source_index, category_names):
    os.makedirs(dst_root, exist_ok=True)
    selected_category = np.asarray(selected_category, dtype=np.int64)
    selected_source_index = np.asarray(selected_source_index, dtype=np.int64)

    np.save(os.path.join(dst_root, 'model_category.npy'), selected_category)
    np.save(os.path.join(dst_root, 'model_source_index.npy'), selected_source_index)

    category_map = {str(i): name for i, name in enumerate(category_names)}
    with open(os.path.join(dst_root, 'category_names.json'), 'w', encoding='utf-8') as f:
        json.dump(category_map, f, ensure_ascii=False, indent=2)

    by_category_dir = os.path.join(dst_root, 'by_category')
    os.makedirs(by_category_dir, exist_ok=True)
    for category_id, category_name in enumerate(category_names):
        indices = np.where(selected_category == category_id)[0].astype(np.int64)
        np.save(os.path.join(by_category_dir, f'{category_name}_indices.npy'), indices)
        print(f'  category {category_name}: {len(indices)} samples')


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


def interp_velocity_models(vel_original, model_indices, times, sigma=INTERP_SIGMA):
    vel_interp = []
    for i in model_indices:
        v = zoom(vel_original[i], times, order=3)
        v = gaussian_filter(v, sigma * times)
        vel_interp.append(v)
    return vel_interp


def solve_to_dense(matrix, rhs):
    solved = spsolve(csc_matrix(matrix), rhs)
    return solved.toarray() if hasattr(solved, 'toarray') else np.asarray(solved)


def get_output_shape(nz, nx, downsample_times, times):
    stride = int(downsample_times) if downsample_times is not None and downsample_times > 1 else 1
    out_z = len(range(0, nx, stride))
    out_x = len(range(0, nz, stride))
    return stride, out_z, out_x


def build_freq_matrix(n_models, freq_configs, seed=FREQ_RANDOM_SEED):
    rng = np.random.default_rng(seed)
    freq_used = np.empty((n_models, len(freq_configs)), dtype=FREQ_DTYPE)
    for freq_idx, (freq_list, _, _) in enumerate(freq_configs):
        freq_used[:, freq_idx] = rng.choice(freq_list, size=n_models)
    return freq_used


def get_full_dataset_paths(dst_root=MERGED_SAVE_ROOT):
    os.makedirs(dst_root, exist_ok=True)
    return {
        'wavefield': os.path.join(dst_root, 'freesurface_full_5sources_wavefield.npy'),
        'background': os.path.join(dst_root, 'freesurface_full_5sources_background.npy'),
        'velocity': os.path.join(dst_root, 'freesurface_full_5sources_velocity.npy'),
        'freq_used': os.path.join(dst_root, 'freesurface_full_5sources_freq_used.npy'),
    }


def downsample_multifreq_dataset(src_root=MERGED_SAVE_ROOT, dst_root=DS2_SAVE_ROOT,
                                 stride=DS2_STRIDE, chunk_size=DS2_CHUNK_SIZE):
    """
    将完整多频数据集的空间维度再次下采样，并复制类别与索引元数据。

    velocity/background/wavefield 的最后两个维度按 [..., ::stride, ::stride]
    写入新 mmap 文件；freq_used 和类别元数据保持不变。
    """
    if stride <= 0:
        raise ValueError(f'stride 必须为正整数, 当前: {stride}')
    if chunk_size <= 0:
        raise ValueError(f'chunk_size 必须为正整数, 当前: {chunk_size}')
    if os.path.abspath(src_root) == os.path.abspath(dst_root):
        raise ValueError('下采样源目录与目标目录不能相同')

    print('\n' + '=' * 60)
    print(f'生成空间下采样数据集: stride={stride}')
    print('=' * 60)
    print(f'源目录: {src_root}')
    print(f'目标目录: {dst_root}')
    os.makedirs(dst_root, exist_ok=True)

    paths = get_full_dataset_paths(src_root)
    array_names = ('velocity', 'background', 'wavefield')
    for array_name in array_names:
        src_path = paths[array_name]
        if not os.path.exists(src_path):
            raise FileNotFoundError(f'缺少下采样源文件: {src_path}')

        src = np.load(src_path, mmap_mode='r')
        if src.ndim < 2:
            raise ValueError(f'{src_path} ndim={src.ndim}, 无法对空间维度下采样')

        out_shape = src.shape[:-2] + (
            len(range(0, src.shape[-2], stride)),
            len(range(0, src.shape[-1], stride)),
        )
        dst_path = os.path.join(dst_root, os.path.basename(src_path))
        dst = np.lib.format.open_memmap(
            dst_path, mode='w+', dtype=src.dtype, shape=out_shape
        )
        print(f'{array_name}: {src.shape} -> {out_shape}')

        for start in range(0, src.shape[0], chunk_size):
            end = min(start + chunk_size, src.shape[0])
            dst[start:end] = src[start:end, ..., ::stride, ::stride]
            dst.flush()
            print(f'  {array_name}: {end}/{src.shape[0]}')
        del dst, src

    metadata_files = (
        'freesurface_full_5sources_freq_used.npy',
        'model_category.npy',
        'model_source_index.npy',
        'category_names.json',
    )
    for filename in metadata_files:
        src_path = os.path.join(src_root, filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(dst_root, filename))

    src_by_category = os.path.join(src_root, 'by_category')
    dst_by_category = os.path.join(dst_root, 'by_category')
    if os.path.isdir(src_by_category):
        os.makedirs(dst_by_category, exist_ok=True)
        for filename in os.listdir(src_by_category):
            if filename.endswith('.npy'):
                shutil.copy2(
                    os.path.join(src_by_category, filename),
                    os.path.join(dst_by_category, filename),
                )

    src_freq = np.load(paths['freq_used'], mmap_mode='r')
    dst_freq = np.load(
        os.path.join(dst_root, os.path.basename(paths['freq_used'])),
        mmap_mode='r',
    )
    if not np.array_equal(src_freq, dst_freq):
        raise ValueError('下采样数据集的 freq_used 与源数据不一致')

    for array_name in array_names:
        src = np.load(paths[array_name], mmap_mode='r')
        dst = np.load(
            os.path.join(dst_root, os.path.basename(paths[array_name])),
            mmap_mode='r',
        )
        expected_shape = src.shape[:-2] + (
            len(range(0, src.shape[-2], stride)),
            len(range(0, src.shape[-1], stride)),
        )
        if dst.shape != expected_shape or dst.dtype != src.dtype:
            raise ValueError(
                f'{array_name} 下采样结果不匹配: shape={dst.shape}, dtype={dst.dtype}, '
                f'expected_shape={expected_shape}, expected_dtype={src.dtype}'
            )
        print(f'验证通过: {array_name}, shape={dst.shape}, dtype={dst.dtype}')

    print('空间下采样数据集生成完成:', dst_root)
    return get_full_dataset_paths(dst_root)


def compute_velocity_output(v, FPML, n, nz, nx, stride):
    vel_model = np.reshape(
        FPML * np.reshape(v, (n[0] * n[1], 1), order='F'), (nz, nx)
    ).T
    return vel_model[::stride, ::stride].astype(VELOCITY_DTYPE, copy=False)


def solve_background_for_frequency(freq, n, n_pml, nz, nx, h, FPML, bb, stride):
    alpha = 0.0
    rhot = (1 - alpha / np.pi * np.log(freq / 50) - 1j * alpha / 2) ** 2
    v0_flat = np.ones((n[1], n[0]), dtype=FIELD_DTYPE).reshape(n[0] * n[1], 1, order='F')
    mv0 = rhot / (v0_flat * 1.5) ** 2
    mv0 = np.reshape(FPML * mv0, (nz, nx))

    # 9pt 的背景矩阵也需要 b/c/d/e；用均匀背景对应的 Gmin 估算即可。
    Gmin = 1500.0 / (h * freq)
    b, c, d, e = optimal_Parameters(Gmin, Gmin)
    b, c, d, e = float(b), float(c), float(d), float(e)

    if STENCIL_TYPE == '25pt':
        A0, _, _, _ = getA25_PML(n_pml, nz, nx, freq, F0, h, mv0)
    else:
        A0, _, _, _ = getA9_PML(n_pml, nz, nx, freq, F0, h, mv0, b, c, d, e)

    d00 = solve_to_dense(A0, bb[0])
    out = np.empty((N_SRC, 2, len(range(0, nx, stride)), len(range(0, nz, stride))), dtype=FIELD_DTYPE)
    for src_idx in range(N_SRC):
        ds0 = np.reshape(d00[:, src_idx], (nx, nz))
        out[src_idx, 0] = np.real(ds0[::stride, ::stride]).astype(FIELD_DTYPE, copy=False)
        out[src_idx, 1] = np.imag(ds0[::stride, ::stride]).astype(FIELD_DTYPE, copy=False)
    return out


def solve_wavefield_for_velocity_frequency(v, freq, n, n_pml, nz, nx, h, FPML, bb, stride):
    alpha = 0.0
    rhot = (1 - alpha / np.pi * np.log(freq / 50) - 1j * alpha / 2) ** 2

    v_min, v_max = np.min(v), np.max(v)
    Gmin = v_min / (h * freq)
    b, c, d, e = optimal_Parameters(Gmin, v_max / (h * freq))
    b, c, d, e = float(b), float(c), float(d), float(e)

    mv = rhot / (np.reshape(v, (n[0] * n[1], 1), order='F') / 1000) ** 2
    mv = np.reshape(FPML * mv, (nz, nx))

    if STENCIL_TYPE == '25pt':
        A, _, _, _ = getA25_PML(n_pml, nz, nx, freq, F0, h, mv)
    else:
        A, _, _, _ = getA9_PML(n_pml, nz, nx, freq, F0, h, mv, b, c, d, e)

    d0 = solve_to_dense(A, bb[0])
    out = np.empty((N_SRC, 2, len(range(0, nx, stride)), len(range(0, nz, stride))), dtype=FIELD_DTYPE)
    for src_idx in range(N_SRC):
        ds = np.reshape(d0[:, src_idx], (nx, nz))
        out[src_idx, 0] = np.real(ds[::stride, ::stride]).astype(FIELD_DTYPE, copy=False)
        out[src_idx, 1] = np.imag(ds[::stride, ::stride]).astype(FIELD_DTYPE, copy=False)
    return out, v_min, v_max, Gmin


def init_worker(vel_interp, n, n_pml, nz, nx, h, FPML, bb, stride):
    _WORKER_STATE.clear()
    _WORKER_STATE.update({
        'vel_interp': vel_interp,
        'n': n,
        'n_pml': n_pml,
        'nz': nz,
        'nx': nx,
        'h': h,
        'FPML': FPML,
        'bb': bb,
        'stride': stride,
    })


def solve_task_worker(task):
    vel_idx, freq_idx, freq = task
    state = _WORKER_STATE
    wavefield, v_min, v_max, Gmin = solve_wavefield_for_velocity_frequency(
        state['vel_interp'][vel_idx], freq, state['n'], state['n_pml'],
        state['nz'], state['nx'], state['h'], state['FPML'], state['bb'], state['stride']
    )
    velocity = compute_velocity_output(
        state['vel_interp'][vel_idx], state['FPML'], state['n'],
        state['nz'], state['nx'], state['stride']
    )
    return vel_idx, freq_idx, wavefield, velocity, v_min, v_max, Gmin


def generate_full_multifreq_dataset(vel_interp, model_indices, n, n_pml, nz, nx,
                                    h, FPML, bb, downsample_times, times,
                                    dst_root=MERGED_SAVE_ROOT, n_workers=N_WORKERS):
    """
    直接生成完整多频数据集。

    每个速度模型对应 len(FREQ_CONFIGS) 个频率；第 freq_idx 个频率从对应频段池中抽取。
    输出:
      velocity: [nvel, n_freq, Z, X]
      freq: [nvel, n_freq]
      background/wavefield: [nvel, n_freq, n_source, 2, Z, X]
    """
    n_models = len(model_indices)
    n_freq = len(FREQ_CONFIGS)
    stride, out_z, out_x = get_output_shape(nz, nx, downsample_times, times)
    paths = get_full_dataset_paths(dst_root)

    print('\n' + '=' * 60)
    print('直接生成完整多频数据集')
    print('=' * 60)
    print(f'layout: [nvel={n_models}, n_freq={n_freq}, n_source={N_SRC}, 2, Z={out_z}, X={out_x}]')
    print(f'并行 worker 数: {n_workers}')

    freq_used = build_freq_matrix(n_models, FREQ_CONFIGS)
    np.save(paths['freq_used'], freq_used)

    velocity_out = np.lib.format.open_memmap(
        paths['velocity'], mode='w+', dtype=VELOCITY_DTYPE,
        shape=(n_models, n_freq, out_z, out_x)
    )
    wavefield_out = np.lib.format.open_memmap(
        paths['wavefield'], mode='w+', dtype=FIELD_DTYPE,
        shape=(n_models, n_freq, N_SRC, 2, out_z, out_x)
    )
    background_out = np.lib.format.open_memmap(
        paths['background'], mode='w+', dtype=FIELD_DTYPE,
        shape=(n_models, n_freq, N_SRC, 2, out_z, out_x)
    )

    unique_freqs = sorted({float(f) for f in freq_used.reshape(-1)})
    print(f'预计算背景场缓存: {unique_freqs}')
    background_cache = {}
    for freq in unique_freqs:
        background_cache[freq] = solve_background_for_frequency(
            freq, n, n_pml, nz, nx, h, FPML, bb, stride
        )

    tasks = [
        (vel_idx, freq_idx, float(freq_used[vel_idx, freq_idx]))
        for vel_idx in range(n_models)
        for freq_idx in range(n_freq)
    ]

    def write_result(vel_idx, freq_idx, wavefield, velocity, v_min, v_max, Gmin):
        freq = float(freq_used[vel_idx, freq_idx])
        wavefield_out[vel_idx, freq_idx] = wavefield
        background_out[vel_idx, freq_idx] = background_cache[freq]
        velocity_out[vel_idx, freq_idx] = velocity
        print(
            f'  [{vel_idx + 1}/{n_models}, freq_idx={freq_idx}] '
            f'freq={freq:g}Hz, v=[{v_min:.0f},{v_max:.0f}]m/s, Gmin={Gmin:.2f}'
        )

    if n_workers and n_workers > 1:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_worker,
            initargs=(vel_interp, n, n_pml, nz, nx, h, FPML, bb, stride),
        ) as executor:
            futures = [executor.submit(solve_task_worker, task) for task in tasks]
            for future in as_completed(futures):
                write_result(*future.result())
    else:
        for vel_idx, freq_idx, freq in tasks:
            wavefield, v_min, v_max, Gmin = solve_wavefield_for_velocity_frequency(
                vel_interp[vel_idx], freq, n, n_pml, nz, nx, h, FPML, bb, stride
            )
            velocity = compute_velocity_output(vel_interp[vel_idx], FPML, n, nz, nx, stride)
            write_result(vel_idx, freq_idx, wavefield, velocity, v_min, v_max, Gmin)

    for arr in (velocity_out, wavefield_out, background_out):
        arr.flush()

    print('\n=== 完整数据集结果验证 ===')
    vel_v = np.load(paths['velocity'], mmap_mode='r')
    wf_v = np.load(paths['wavefield'], mmap_mode='r')
    bg_v = np.load(paths['background'], mmap_mode='r')
    fq_v = np.load(paths['freq_used'], mmap_mode='r')
    print(f'velocity:   {vel_v.shape}')
    print(f'wavefield:  {wf_v.shape}')
    print(f'background: {bg_v.shape}')
    print(f'freq_used:  {fq_v.shape}')

    expected_field_shape = (n_models, n_freq, N_SRC, 2, out_z, out_x)
    if vel_v.shape != (n_models, n_freq, out_z, out_x):
        raise ValueError(f'velocity shape {vel_v.shape} 不符合预期')
    if wf_v.shape != expected_field_shape:
        raise ValueError(f'wavefield shape {wf_v.shape} != {expected_field_shape}')
    if bg_v.shape != expected_field_shape:
        raise ValueError(f'background shape {bg_v.shape} != {expected_field_shape}')
    if fq_v.shape != (n_models, n_freq):
        raise ValueError(f'freq_used shape {fq_v.shape} 不符合预期')

    print('完整多频数据集生成完成:', dst_root)
    return paths


def solve_stage(freq_list, vel_interp, model_indices, n, n_pml, nz, nx,
                h, FPML, bb, downsample_times, times, output_paths=None):
    """
    对一组频率进行正演，返回 (UU, UU0, vel_out, freq_used)。

    UU/UU0 layout: [nvel, n_source, 2, Z, X]
    vel_out layout: [nvel, Z, X]
    """
    n_models = len(model_indices)
    stride, out_z, out_x = get_output_shape(nz, nx, downsample_times, times)

    if output_paths:
        vel_out = np.lib.format.open_memmap(
            output_paths['velocity'], mode='w+', dtype=VELOCITY_DTYPE,
            shape=(n_models, out_z, out_x)
        )
        UU = np.lib.format.open_memmap(
            output_paths['wavefield'], mode='w+', dtype=FIELD_DTYPE,
            shape=(n_models, N_SRC, 2, out_z, out_x)
        )
        UU0 = np.lib.format.open_memmap(
            output_paths['background'], mode='w+', dtype=FIELD_DTYPE,
            shape=(n_models, N_SRC, 2, out_z, out_x)
        )
        freq_used = np.lib.format.open_memmap(
            output_paths['freq_used'], mode='w+', dtype=FREQ_DTYPE,
            shape=(n_models,)
        )
    else:
        vel_out = np.empty((n_models, out_z, out_x), dtype=VELOCITY_DTYPE)
        UU = np.empty((n_models, N_SRC, 2, out_z, out_x), dtype=FIELD_DTYPE)
        UU0 = np.empty_like(UU)
        freq_used = np.empty(n_models, dtype=FREQ_DTYPE)

    v0_flat = np.ones((n[1], n[0]), dtype=FIELD_DTYPE).reshape(n[0] * n[1], 1, order='F')

    def solve_to_dense(matrix, rhs):
        solved = spsolve(csc_matrix(matrix), rhs)
        return solved.toarray() if hasattr(solved, 'toarray') else np.asarray(solved)

    for idx in range(n_models):
        freq = np.random.choice(freq_list)
        freq_used[idx] = freq

        # 衰减系数
        alpha = 1 / Q
        alpha = 0.0
        rhot = (1 - alpha / np.pi * np.log(freq / 50) - 1j * alpha / 2) ** 2

        v = vel_interp[idx]
        v_min, v_max = np.min(v), np.max(v)
        Gmin = v_min / (h * freq)

        b, c, d, e = optimal_Parameters(Gmin, v_max / (h * freq))
        b, c, d, e = float(b), float(c), float(d), float(e)
        print(f'  [{idx+1}/{n_models}] freq={freq}Hz, v=[{v_min:.0f},{v_max:.0f}]m/s, Gmin={Gmin:.2f}')

        # 均匀背景
        mv0 = rhot / (v0_flat * 1.5) ** 2
        mv0 = np.reshape(FPML * mv0, (nz, nx))

        vel_model = np.reshape(
            FPML * np.reshape(v, (n[0] * n[1], 1), order='F'), (nz, nx)
        ).T
        vel_out[idx] = vel_model[::stride, ::stride].astype(VELOCITY_DTYPE, copy=False)

        mv = rhot / (np.reshape(v, (n[0] * n[1], 1), order='F') / 1000) ** 2
        mv = np.reshape(FPML * mv, (nz, nx))

        # 构建系数矩阵并求解
        if STENCIL_TYPE == '25pt':
            A, _, _, _ = getA25_PML(n_pml, nz, nx, freq, F0, h, mv)
            A0, _, _, _ = getA25_PML(n_pml, nz, nx, freq, F0, h, mv0)
        else:
            A, _, _, _ = getA9_PML(n_pml, nz, nx, freq, F0, h, mv, b, c, d, e)
            A0, _, _, _ = getA9_PML(n_pml, nz, nx, freq, F0, h, mv0, b, c, d, e)

        d0 = solve_to_dense(A, bb[0])
        d00 = solve_to_dense(A0, bb[0])

        for s in range(N_SRC):
            ds = np.reshape(d0[:, s], (nx, nz))
            ds0 = np.reshape(d00[:, s], (nx, nz))
            UU[idx, s, 0] = np.real(ds[::stride, ::stride]).astype(FIELD_DTYPE, copy=False)
            UU[idx, s, 1] = np.imag(ds[::stride, ::stride]).astype(FIELD_DTYPE, copy=False)
            UU0[idx, s, 0] = np.real(ds0[::stride, ::stride]).astype(FIELD_DTYPE, copy=False)
            UU0[idx, s, 1] = np.imag(ds0[::stride, ::stride]).astype(FIELD_DTYPE, copy=False)

    if stride > 1:
        print(f'  降采样 stride={stride}: field -> {UU.shape}, velocity -> {vel_out.shape}')

    for arr in (UU, UU0, vel_out, freq_used):
        if hasattr(arr, 'flush'):
            arr.flush()

    return UU, UU0, vel_out, freq_used


def get_stage_paths(dir_name, freq_tag):
    save_dir = os.path.join(SAVE_ROOT, dir_name)
    os.makedirs(save_dir, exist_ok=True)
    prefix = f'freesurface_{freq_tag}_5sources'
    return {
        'wavefield': os.path.join(save_dir, f'{prefix}_wavefield.npy'),
        'background': os.path.join(save_dir, f'{prefix}_background.npy'),
        'velocity': os.path.join(save_dir, f'{prefix}_velocity.npy'),
        'freq_used': os.path.join(save_dir, f'{prefix}_freq_used.npy'),
    }


def save_stage(dir_name, freq_tag, UU, UU0, vel_out, freq_used):
    paths = get_stage_paths(dir_name, freq_tag)

    np.save(paths['wavefield'], UU)
    np.save(paths['background'], UU0)
    np.save(paths['velocity'], vel_out)
    np.save(paths['freq_used'], freq_used)
    print(f'  已保存到 {os.path.dirname(paths["wavefield"])}/')


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
    model_idx = 0
    src_idx = 0

    def adaptive_lim(data, pct=99):
        mx = np.percentile(np.abs(data), pct)
        return -mx, mx

    fields = [
        (axes[0, 0], UU[model_idx, src_idx, 0], 'seismic', 'Wavefield Real'),
        (axes[0, 1], UU0[model_idx, src_idx, 0], 'seismic', 'Background Real'),
        (axes[0, 2], UU[model_idx, src_idx, 0] - UU0[model_idx, src_idx, 0], 'seismic', 'Scattered Field'),
        (axes[1, 0], UU[model_idx, src_idx, 1], 'seismic', 'Wavefield Imag'),
        (axes[1, 1], vel[model_idx], 'seismic', 'Velocity Model'),
        (axes[1, 2], np.abs(UU[model_idx, src_idx, 0]) / (np.abs(UU0[model_idx, src_idx, 0]) + 1e-10), 'hot', 'Scattered/BG Ratio'),
    ]
    for ax, data, cmap, title in fields:
        if cmap == 'seismic' and title == 'Velocity Model':
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


def visualize_full_dataset(paths):
    """可视化完整新 layout 数据集中的第一个速度模型、低频段、0号震源。"""
    UU = np.load(paths['wavefield'], mmap_mode='r')
    UU0 = np.load(paths['background'], mmap_mode='r')
    vel = np.load(paths['velocity'], mmap_mode='r')
    freq_used = np.load(paths['freq_used'], mmap_mode='r')

    model_idx = 0
    freq_idx = 0
    src_idx = 0

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    def adaptive_lim(data, pct=99):
        mx = np.percentile(np.abs(data), pct)
        return -mx, mx

    fields = [
        (axes[0, 0], UU[model_idx, freq_idx, src_idx, 0], 'seismic', 'Wavefield Real'),
        (axes[0, 1], UU0[model_idx, freq_idx, src_idx, 0], 'seismic', 'Background Real'),
        (
            axes[0, 2],
            UU[model_idx, freq_idx, src_idx, 0] - UU0[model_idx, freq_idx, src_idx, 0],
            'seismic',
            'Scattered Field',
        ),
        (axes[1, 0], UU[model_idx, freq_idx, src_idx, 1], 'seismic', 'Wavefield Imag'),
        (axes[1, 1], vel[model_idx, freq_idx], 'seismic', 'Velocity Model'),
        (
            axes[1, 2],
            np.abs(UU[model_idx, freq_idx, src_idx, 0]) /
            (np.abs(UU0[model_idx, freq_idx, src_idx, 0]) + 1e-10),
            'hot',
            'Scattered/BG Ratio',
        ),
    ]
    for ax, data, cmap, title in fields:
        if cmap == 'seismic' and title == 'Velocity Model':
            vmin, vmax = data.min(), data.max()
        elif cmap == 'hot':
            vmin, vmax = 0, np.percentile(data, 99)
        else:
            vmin, vmax = adaptive_lim(data)
        im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'{title}\n[{vmin:.1f}, {vmax:.1f}]')
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    fig.suptitle(
        f'Stencil: {STENCIL_TYPE}, freq={freq_used[model_idx, freq_idx]:g}Hz, Grid: {40/TIMES}m',
        fontsize=12,
    )
    plt.subplots_adjust(top=0.88)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'可视化已保存到 {out_path}')


def merge_multifreq_stages(src_root=SAVE_ROOT, dst_root=MERGED_SAVE_ROOT):
    """
    合并三个频段阶段数据，生成训练脚本直接使用的 multifreq_merged1 数据。

    输出 layout:
      velocity: [nvel, n_freq, Z, X]
      freq: [nvel, n_freq]
      background/wavefield: [nvel, n_freq, n_source, 2, Z, X]
    """
    print('\n' + '=' * 60)
    print('合并三阶段多频数据')
    print('=' * 60)

    os.makedirs(dst_root, exist_ok=True)

    def normalize_stage_field(field, n_vel, field_name, stage_name):
        """兼容旧 flatten layout，并统一为 [nvel, n_source, 2, Z, X]。"""
        if field.ndim == 5:
            expected = (n_vel, N_SRC)
            if field.shape[:2] != expected:
                raise ValueError(
                    f'{stage_name} {field_name} shape {field.shape} 不符合 [nvel, n_source, 2, Z, X]'
                )
            return field
        if field.ndim == 4:
            if field.shape[0] != n_vel * N_SRC:
                raise ValueError(
                    f'{stage_name} {field_name} 第一维 {field.shape[0]} != nvel {n_vel} * N_SRC {N_SRC}'
                )
            print(f'  {stage_name} {field_name}: 检测到旧 layout，转换为 [nvel, n_source, 2, Z, X]')
            return field.reshape(N_SRC, n_vel, *field.shape[1:]).transpose(1, 0, 2, 3, 4)
        raise ValueError(f'{stage_name} {field_name} 不支持的 shape: {field.shape}')

    def validate_freq(freq, n_vel, stage_name):
        if freq.ndim != 1 or freq.shape[0] != n_vel:
            raise ValueError(f'{stage_name} freq shape {freq.shape} 不符合 [nvel={n_vel}]')

    def arrays_equal_chunked(a, b, chunk_size=64):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        for start in range(0, a.shape[0], chunk_size):
            end = min(start + chunk_size, a.shape[0])
            if not np.array_equal(a[start:end], b[start:end]):
                return False
        return True

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
    first_vel = np.load(os.path.join(src_root, first_dir, f'{first_prefix}_velocity.npy'), mmap_mode='r')
    n_vel = first_vel.shape[0]

    for dir_name, freq_tag in MERGE_STAGE_CONFIGS[1:]:
        prefix = f'freesurface_{freq_tag}_5sources'
        stage_vel = np.load(os.path.join(src_root, dir_name, f'{prefix}_velocity.npy'), mmap_mode='r')
        if not arrays_equal_chunked(first_vel, stage_vel):
            raise ValueError(f'{dir_name} velocity 与 {first_dir} 不一致，不能直接合并')

    n_stages = len(MERGE_STAGE_CONFIGS)
    out_z, out_x = first_vel.shape[-2], first_vel.shape[-1]

    vel_path = os.path.join(dst_root, 'freesurface_full_5sources_velocity.npy')
    freq_path = os.path.join(dst_root, 'freesurface_full_5sources_freq_used.npy')
    bg_path = os.path.join(dst_root, 'freesurface_full_5sources_background.npy')
    wf_path = os.path.join(dst_root, 'freesurface_full_5sources_wavefield.npy')

    vel_merged = np.lib.format.open_memmap(
        vel_path, mode='w+', dtype=VELOCITY_DTYPE,
        shape=(n_vel, n_stages, out_z, out_x)
    )
    freq_merged = np.lib.format.open_memmap(
        freq_path, mode='w+', dtype=FREQ_DTYPE,
        shape=(n_vel, n_stages)
    )
    background_merged = np.lib.format.open_memmap(
        bg_path, mode='w+', dtype=FIELD_DTYPE,
        shape=(n_vel, n_stages, N_SRC, 2, out_z, out_x)
    )
    wavefield_merged = np.lib.format.open_memmap(
        wf_path, mode='w+', dtype=FIELD_DTYPE,
        shape=(n_vel, n_stages, N_SRC, 2, out_z, out_x)
    )

    for stage_idx, (dir_name, freq_tag) in enumerate(MERGE_STAGE_CONFIGS):
        stage_dir = os.path.join(src_root, dir_name)
        prefix = f'freesurface_{freq_tag}_5sources'

        stage_vel = np.load(os.path.join(stage_dir, f'{prefix}_velocity.npy'), mmap_mode='r')
        vel_merged[:, stage_idx] = stage_vel

        freq = np.load(os.path.join(stage_dir, f'{prefix}_freq_used.npy'), mmap_mode='r')
        validate_freq(freq, n_vel, dir_name)
        freq_merged[:, stage_idx] = freq

        bg = np.load(os.path.join(stage_dir, f'{prefix}_background.npy'), mmap_mode='r')
        wf = np.load(os.path.join(stage_dir, f'{prefix}_wavefield.npy'), mmap_mode='r')
        background_merged[:, stage_idx] = normalize_stage_field(bg, n_vel, 'background', dir_name)
        wavefield_merged[:, stage_idx] = normalize_stage_field(wf, n_vel, 'wavefield', dir_name)

        print(f'  写入 stage {stage_idx}: {dir_name}')

    for arr in (vel_merged, freq_merged, background_merged, wavefield_merged):
        arr.flush()

    print(f'background: saved -> {bg_path}, shape={background_merged.shape}')
    print(f'wavefield: saved -> {wf_path}, shape={wavefield_merged.shape}')
    print(f'velocity: saved -> {vel_path}, shape={vel_merged.shape}')
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

    expected_field_shape = (n_vel, len(MERGE_STAGE_CONFIGS), N_SRC, 2, vel_v.shape[-2], vel_v.shape[-1])
    if vel_v.shape[:2] != fq_v.shape:
        raise ValueError(f'vel 前两维 {vel_v.shape[:2]} 与 freq shape {fq_v.shape} 不匹配')
    if bg_v.shape != expected_field_shape:
        raise ValueError(f'background shape {bg_v.shape} != {expected_field_shape}')
    if wf_v.shape != bg_v.shape:
        raise ValueError(f'wavefield shape {wf_v.shape} 与 background shape {bg_v.shape} 不匹配')

    print('合并完成:', dst_root)


# =====================================================================
# 主流程
# =====================================================================
def main():
    t_total = _time.time()

    # 网格参数
    nz_base = nx_base = 140 * TIMES
    h = 20 / TIMES
    Lpml = 20 * TIMES

    n = np.array([nx_base, nz_base])
    n_pml = np.array([[Lpml, 0], [Lpml, Lpml]])
    ne = n + np.sum(n_pml, axis=0)
    nz, nx = ne[0], ne[1]

    print(f'网格: {nz}x{nx}, h={h}m, PML={Lpml}, stencil={STENCIL_TYPE}')

    # PML 吸收系数 (所有阶段共用)
    FPML = getFPML(n_pml, n)
    bb = build_source_terms(nz, nx, n_pml, TIMES)

    # 加载 & 插值速度模型
    vel_original, model_category, model_source_index, category_names = load_velocity_banks()
    print(f'速度模型池: {vel_original.shape}, 范围: [{vel_original.min():.2f}, {vel_original.max():.2f}] km/s')

    # bank 存储单位 km/s → 内部计算和输出统一用 m/s
    vel_original = vel_original * 1000.0
    print(f'单位转换: km/s → m/s, 范围: [{vel_original.min():.0f}, {vel_original.max():.0f}] m/s')

    # 按类别均等抽取
    np.random.seed(42)
    model_indices = []
    n_categories = len(category_names)
    for cat_id in range(n_categories):
        cat_mask = model_category == cat_id
        cat_indices = np.where(cat_mask)[0]
        if len(cat_indices) < N_MODELS_PER_CATEGORY:
            raise ValueError(
                f'类别 {category_names[cat_id]} 只有 {len(cat_indices)} 个模型, '
                f'不足 N_MODELS_PER_CATEGORY={N_MODELS_PER_CATEGORY}'
            )
        selected = np.random.choice(cat_indices, size=N_MODELS_PER_CATEGORY, replace=False)
        model_indices.append(selected)
    model_indices = np.concatenate(model_indices)
    np.random.shuffle(model_indices)
    n_models_total = len(model_indices)
    print(f'每类抽取 {N_MODELS_PER_CATEGORY} 个, {n_categories} 类, 共 {n_models_total} 个模型')
    save_category_metadata(
        MERGED_SAVE_ROOT,
        model_category[model_indices],
        model_source_index[model_indices],
        category_names,
    )

    # 根据 bank 实际尺寸计算 zoom 倍数，自动适配不同分辨率输入
    bank_nz, bank_nx = vel_original.shape[-2], vel_original.shape[-1]
    if bank_nz != bank_nx:
        raise ValueError(f'速度模型必须为方形, 当前: {bank_nz}×{bank_nx}')
    zoom_factor = nz_base / bank_nx
    print(f'插值速度模型... zoom={zoom_factor:.2f} ({bank_nz}×{bank_nx} → {nz_base}×{nz_base})')

    vel_interp = interp_velocity_models(vel_original, model_indices, zoom_factor)
    del vel_original
    gc.collect()

    # 验证插值后 shape 与网格匹配
    actual_shape = vel_interp[0].shape
    expected_shape = (nz_base, nz_base)
    if actual_shape != expected_shape:
        raise ValueError(
            f'插值结果 shape {actual_shape} != 网格 {expected_shape}, '
            f'zoom_factor={zoom_factor:.4f}'
        )
    print(f'插值完成: {len(vel_interp)} 个模型, shape={actual_shape}')

    dataset_paths = generate_full_multifreq_dataset(
        vel_interp, model_indices, n, n_pml, nz, nx,
        h, FPML, bb, DOWNSAMPLE_TIMES, TIMES,
        dst_root=MERGED_SAVE_ROOT, n_workers=N_WORKERS,
    )

    if GENERATE_DS2:
        downsample_multifreq_dataset(
            src_root=MERGED_SAVE_ROOT,
            dst_root=DS2_SAVE_ROOT,
            stride=DS2_STRIDE,
            chunk_size=DS2_CHUNK_SIZE,
        )

    # 释放插值模型 & PML
    del vel_interp, FPML, bb
    gc.collect()

    visualize_full_dataset(dataset_paths)

    print(f'\n全部完成, 总耗时: {_time.time()-t_total:.1f}s')


def run_synthetic_test():
    """
    用两个解析速度模型做独立正演测试，不存储 .npy，只输出图像。

    模型1: 全局均匀 v = 1500 m/s
    模型2: 上半 v = 1500, 下半 v = 3000 m/s

    震源: 中心震源 (index=2, 即 SRC_X_ORIG[2]=90)
    频率: 三组各取一个代表频率 (5, 15, 25 Hz)
    输出: 每个模型×每个频率 → 全波场 + 散射波场 + 速度模型 图像
    """
    t0 = _time.time()

    # 网格参数 (与 main 一致)
    nz_base = nx_base = 140 * TIMES
    h = 20 / TIMES
    Lpml = 20 * TIMES
    n = np.array([nx_base, nz_base])
    n_pml = np.array([[Lpml, 0], [Lpml, Lpml]])
    ne = n + np.sum(n_pml, axis=0)
    nz, nx = ne[0], ne[1]

    FPML = getFPML(n_pml, n)
    bb = build_source_terms(nz, nx, n_pml, TIMES)
    stride, _, _ = get_output_shape(nz, nx, DOWNSAMPLE_TIMES, TIMES)
    center_src = 2  # SRC_X_ORIG[2] = 90 → 中心震源
    test_freqs = [5, 15, 25]  # 低/中/高

    # 构造两个速度模型 (m/s)
    vel_uniform = np.full((nz_base, nz_base), 1500.0, dtype=VELOCITY_DTYPE)
    vel_two_layer = np.zeros((nz_base, nz_base), dtype=VELOCITY_DTYPE)
    vel_two_layer[: nz_base // 2, :] = 1500.0
    vel_two_layer[nz_base // 2 :, :] = 3000.0
    test_models = [
        ('uniform_1500', vel_uniform),
        ('two_layer_1500_3000', vel_two_layer),
    ]

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_output')
    os.makedirs(out_dir, exist_ok=True)

    for model_name, v in test_models:
        for freq in test_freqs:
            print(f'--- {model_name}, freq={freq}Hz ---')

            # 求解全波场
            wf_all, v_min, v_max, Gmin = solve_wavefield_for_velocity_frequency(
                v, freq, n, n_pml, nz, nx, h, FPML, bb, stride
            )
            # 求解背景场
            bg_all = solve_background_for_frequency(
                freq, n, n_pml, nz, nx, h, FPML, bb, stride
            )
            # 散射场 = 全波场 - 背景场
            scat_all = wf_all - bg_all

            # 只取中心震源
            wf = wf_all[center_src]   # (2, out_z, out_x)
            bg = bg_all[center_src]
            scat = scat_all[center_src]

            # 速度模型 (PML 扩展域)
            vel_ext = compute_velocity_output(v, FPML, n, nz, nx, stride)

            # 绘图: 2行3列
            #   row0: 全波场实部, 背景场实部, 散射场实部
            #   row1: 全波场虚部, 速度模型, 散射场振幅
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            def sym_lim(data, pct=99):
                mx = np.percentile(np.abs(data), pct)
                return -mx, mx

            plots = [
                (axes[0, 0], wf[0], 'seismic', 'Wavefield Real'),
                (axes[0, 1], bg[0], 'seismic', 'Background Real'),
                (axes[0, 2], scat[0], 'seismic', 'Scattered Real'),
                (axes[1, 0], wf[1], 'seismic', 'Wavefield Imag'),
                (axes[1, 1], vel_ext, 'seismic', 'Velocity (m/s)'),
                (axes[1, 2], np.sqrt(scat[0]**2 + scat[1]**2), 'hot', '|Scattered|'),
            ]
            for ax, data, cmap, title in plots:
                if cmap == 'viridis':
                    vmin, vmax = data.min(), data.max()
                elif cmap == 'hot':
                    vmin, vmax = 0, np.percentile(data, 99)
                else:
                    vmin, vmax = sym_lim(data)
                im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f'{title}\n[{vmin:.2e}, {vmax:.2e}]')
                fig.colorbar(im, ax=ax, shrink=0.8)

            fig.suptitle(
                f'{model_name} | freq={freq}Hz | v=[{v_min:.0f},{v_max:.0f}]m/s | '
                f'Gmin={Gmin:.1f} | h={h}m | stencil={STENCIL_TYPE}',
                fontsize=11,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.94])

            fname = f'{model_name}_f{freq}Hz.png'
            plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  saved: {fname}')

    print(f'\n测试完成, 总耗时: {_time.time()-t0:.1f}s')
    print(f'图像保存在: {out_dir}')


if __name__ == '__main__':
    main()
