"""
诊断 output1 绘图问题：source_list=[0] 但画出了 source 2 的图像。

模拟 dataloader 完整数据流，检查经 multi-freq 重排后，
UU_loc[0] 中的数据实际对应哪个震源。
"""

import os
import numpy as np
import torch

MULTIFREQ_ROOT = '/home/sharedata/zdg/multifreq_selected'

STAGES = [
    {'dir': 'freq_3to11', 'tag': 'freq3to11'},
    {'dir': 'freq_12to18', 'tag': 'freq12to18'},
    {'dir': 'freq_18to25', 'tag': 'freq18to25'},
]

def load_mmap(path):
    return np.load(path, mmap_mode='r')

def main():
    # ---- 1. 加载各阶段 ground truth（震源排列已知） ----
    print('[1] 加载各阶段 ground truth ...')
    stage_bg = {}
    for stage in STAGES:
        tag = stage['tag']
        path = os.path.join(MULTIFREQ_ROOT, stage['dir'],
                            f'freesurface_backgroundfield_{tag}_5sources_160_180_pml20_n1.npy')
        stage_bg[stage['dir']] = load_mmap(path)
        print(f'  {stage["dir"]}: shape={stage_bg[stage["dir"]].shape}')

    # 各阶段数据布局: [src0×2000, src1×2000, src2×2000, src3×2000, src4×2000]
    N_VEL = 2000

    # ---- 2. 加载 combined freq3to25 数据 ----
    print('\n[2] 加载 combined freq3to25 数据 ...')
    bg_combined = torch.tensor(np.load(
        os.path.join(MULTIFREQ_ROOT, 'freesurface_backgroundfield_freq3to25_5sources_160_180_pml20_n1.npy')
    ), dtype=torch.float32)
    vel_combined = torch.tensor(np.load(
        os.path.join(MULTIFREQ_ROOT, 'freesurface_velocity_freq3to25_5sources_160_180_pml20_n1.npy')
    ), dtype=torch.float32)
    print(f'  bg_combined: {bg_combined.shape}')
    print(f'  vel_combined: {vel_combined.shape}')

    n_vel_total = vel_combined.shape[0]  # 6000
    n_freq = 3
    n_src = bg_combined.shape[0] // n_vel_total  # 5
    n_vel_per_freq = n_vel_total // n_freq       # 2000

    # ---- 3. 模拟 dataloader 的 multi-freq 重排 ----
    print(f'\n[3] 模拟 dataloader 重排: n_freq={n_freq}, n_src={n_src}, n_vel_per_freq={n_vel_per_freq}')
    UU0 = bg_combined.reshape(n_freq, n_src, n_vel_per_freq, *bg_combined.shape[1:])
    UU0 = UU0.permute(1, 0, 2, *range(3, UU0.dim())).contiguous().reshape(n_src * n_vel_total, *bg_combined.shape[1:])

    # 震源拆分
    UU0_loc = [UU0[loc * n_vel_total : (loc + 1) * n_vel_total] for loc in range(5)]

    print(f'  UU0_loc[0] shape: {UU0_loc[0].shape}')
    print(f'  UU0_loc[0] 布局: [freq0_0..freq0_N, freq1_0..freq1_N, freq2_0..freq2_N]')

    # ---- 4. 核心验证：UU_loc[0] 中的数据实际对应哪个震源？ ----
    print(f'\n{"="*70}')
    print('核心验证: UU0_loc[0] 中的数据实际对应哪个震源？')
    print(f'{"="*70}')

    # 对每个 (freq_stage, vel_idx)，检查 UU0_loc[0] 的数据与哪个 source 的 ground truth 匹配
    for f_idx in range(n_freq):
        # UU0_loc[0] 中第 f_idx 段: indices [f_idx*2000 : (f_idx+1)*2000]
        dl_data = UU0_loc[0][f_idx * N_VEL : (f_idx + 1) * N_VEL]  # [2000, 2, NZ, NX]

        # Ground truth: 各震源在各阶段的数据
        stage_key = STAGES[f_idx]['dir']
        gt = stage_bg[stage_key]  # [10000, 2, NZ, NX]

        # 抽样比较: 取 vel_idx=100
        v_idx = 100
        dl_sample = dl_data[v_idx].numpy()  # [2, NZ, NX]

        for src in range(5):
            gt_sample = np.array(gt[src * N_VEL + v_idx])
            match = np.allclose(dl_sample, gt_sample, atol=0)
            if match:
                print(f'  freq_stage={f_idx} ({STAGES[f_idx]["dir"]}), vel_idx={v_idx}: '
                      f'UU0_loc[0] 匹配 source {src} 的 ground truth')
                break
        else:
            # 没有任何 source 匹配 — 检查偏差
            print(f'  freq_stage={f_idx}, vel_idx={v_idx}: 未匹配任何 source!')
            for src in range(5):
                gt_sample = np.array(gt[src * N_VEL + v_idx])
                diff = np.abs(dl_sample - gt_sample).max()
                print(f'    vs source {src}: max_diff = {diff:.4e}')

    # ---- 5. 检查 combined 文件本身是否由 per-stage 简单拼接 ----
    print(f'\n{"="*70}')
    print('检查 combined 文件是否由 per-stage 文件简单拼接')
    print(f'{"="*70}')

    for f_idx in range(n_freq):
        start = f_idx * 10000  # 每个 stage 的 bg 有 10000 行
        end = start + 10000
        combined_slice = bg_combined[start:end]

        stage_key = STAGES[f_idx]['dir']
        gt = torch.tensor(np.array(stage_bg[stage_key]), dtype=torch.float32)

        match = torch.allclose(combined_slice, gt, atol=0)
        print(f'  combined[{start}:{end}] vs {stage_key} ground truth: '
              f'{"✓ MATCH" if match else "✗ MISMATCH"}')

        if not match:
            diff = (combined_slice - gt).abs().max().item()
            print(f'    max_diff = {diff:.4e}')

    # ---- 6. 模拟完整绘图数据选取 ----
    print(f'\n{"="*70}')
    print('模拟完整绘图数据选取 (source_list=[0])')
    print(f'{"="*70}')

    # PML crop (free_surface)
    pml_crop = 15
    z_slice = slice(0, -pml_crop)
    x_slice = slice(pml_crop, -pml_crop)
    UU0_cropped = UU0[:, :, z_slice, x_slice]

    UU0_loc_crop = [UU0_cropped[loc * n_vel_total : (loc + 1) * n_vel_total] for loc in range(5)]

    # freq
    freq_combined = np.load(
        os.path.join(MULTIFREQ_ROOT, 'freesurface_freq_used_5sources_160_180_pml20_n1.npy'))
    freq = torch.tensor(freq_combined, dtype=torch.float32)

    # Training_data: source_list=[0], nvel_train=4500
    np.random.seed(1)
    idx = np.random.choice(n_vel_total, 4500, replace=False)

    # source 0 的数据
    u0_train_s0 = UU0_loc_crop[0][idx[:4500]]
    freq_train = freq[idx[:4500]]
    hf_test_idx = freq_train.argmax().item()
    print(f'  hf_test_idx = {hf_test_idx}, freq = {freq_train[hf_test_idx].item():.1f} Hz')
    print(f'  原始 vel index = {idx[hf_test_idx]}')

    # 对应的 UU0 在重排后的位置
    vel_idx = idx[hf_test_idx]
    u0_plot = UU0_loc[0][vel_idx].numpy()  # UU0_loc[0] 中第 vel_idx 个

    # 判断这个 u0_plot 属于哪个 source: 在 ground truth 中查找
    if vel_idx < N_VEL:
        f_idx = 0
    elif vel_idx < 2 * N_VEL:
        f_idx = 1
    else:
        f_idx = 2
    local_v = vel_idx - f_idx * N_VEL

    stage_key = STAGES[f_idx]['dir']
    gt = stage_bg[stage_key]

    for src in range(5):
        gt_sample = np.array(gt[src * N_VEL + local_v])
        # 需要 PML crop
        gt_sample_cropped = gt_sample[:, :145-pml_crop, pml_crop:180-pml_crop] if gt_sample.ndim == 3 else gt_sample
        # 直接比较未 crop 的
        if np.allclose(u0_plot, gt_sample, atol=0):
            print(f'  → 绘图用的 UU0 实际属于 source {src} (stage={STAGES[f_idx]["dir"]}, vel_local={local_v})')
            break
    else:
        print(f'  → 未匹配任何 source! 检查各 source 的差异:')
        for src in range(5):
            gt_sample = np.array(gt[src * N_VEL + local_v])
            diff = np.abs(u0_plot - gt_sample).max()
            print(f'      vs source {src}: max_diff = {diff:.4e}')


if __name__ == '__main__':
    main()
