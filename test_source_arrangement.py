"""
验证 multifreq_selected 中拼接后的多频训练数据震源排列与 dataloader 读取是否一致。

核心逻辑：
  1. 从各阶段子目录读取"每阶段真实数据"（ground truth）
  2. 从 multifreq_selected/ 根目录读取"合并后的 freq3to25 数据"
  3. 模拟 dataloader.py 中 prepare_training_dataloaders 的 multi-freq 重排逻辑
  4. 逐个 (freq_stage, source, vel_idx) 比较两者是否一致
"""

import os
import numpy as np
import torch

# ===================== 路径配置 =====================
MULTIFREQ_ROOT = '/home/sharedata/zdg/multifreq_selected'

STAGES = [
    {'dir': 'freq_3to11', 'tag': 'freq3to11'},
    {'dir': 'freq_12to18', 'tag': 'freq12to18'},
    {'dir': 'freq_18to25', 'tag': 'freq18to25'},
]
N_FREQ = len(STAGES)
N_SRC = 5

# 合并数据文件名
COMBINED_FILES = {
    'velocity':        'freesurface_velocity_freq3to25_5sources_160_180_pml20_n1.npy',
    'backgroundfield': 'freesurface_backgroundfield_freq3to25_5sources_160_180_pml20_n1.npy',
    'wavefield':       'freesurface_wavefield_freq3to25_5sources_160_180_pml20_n1.npy',
    'freq_used':       'freesurface_freq_used_5sources_160_180_pml20_n1.npy',
}

# ===================== 工具函数 =====================

def load_mmap(path):
    return np.load(path, mmap_mode='r')


def load_stage_data(stage_info):
    """加载单个阶段的 ground truth 数据（mmap 模式）"""
    d = stage_info['dir']
    tag = stage_info['tag']
    base = os.path.join(MULTIFREQ_ROOT, d)
    return {
        'velocity':        load_mmap(os.path.join(base, f'freesurface_velocity_{tag}_5sources_160_180_pml20_n1.npy')),
        'backgroundfield': load_mmap(os.path.join(base, f'freesurface_backgroundfield_{tag}_5sources_160_180_pml20_n1.npy')),
        'wavefield':       load_mmap(os.path.join(base, f'freesurface_wavefield_{tag}_5sources_160_180_pml20_n1.npy')),
        'freq_used':       load_mmap(os.path.join(base, f'freesurface_freq_used_5sources_160_180_pml20_n1.npy')),
    }


def simulate_dataloader_reshuffle(field_tensor, vel_shape_0):
    """
    模拟 dataloader.py 中 prepare_training_dataloaders 的 multi-freq 重排逻辑。

    输入: field_tensor [n_freq * n_src * n_vel_per_freq, C, H, W]  (频率优先排列)
    输出: 重排后的 tensor [n_src * vel_shape_0, C, H, W]  (震源优先排列)
    """
    n_freq = N_FREQ
    n_src = field_tensor.shape[0] // vel_shape_0
    n_vel_per_freq = vel_shape_0 // n_freq

    t = torch.tensor(np.array(field_tensor), dtype=torch.float32)

    # reshape: (n_freq, n_src, n_vel_per_freq, C, H, W)
    t = t.reshape(n_freq, n_src, n_vel_per_freq, *t.shape[1:])
    # permute: (n_src, n_freq, n_vel_per_freq, C, H, W)
    t = t.permute(1, 0, 2, *range(3, t.dim())).contiguous()
    # reshape: (n_src * vel_shape_0, C, H, W)
    t = t.reshape(n_src * vel_shape_0, *t.shape[3:])
    return t


def simulate_source_split(field_tensor, n_vel):
    """模拟 dataloader.py 中按震源拆分的逻辑"""
    return [field_tensor[loc * n_vel : (loc + 1) * n_vel] for loc in range(N_SRC)]


# ===================== 主验证逻辑 =====================

def main():
    print('=' * 70)
    print('验证 multifreq_selected 拼接数据震源排列一致性')
    print('=' * 70)

    # ---- 1. 加载各阶段 ground truth ----
    print('\n[1] 加载各阶段 ground truth 数据 ...')
    stage_data = [load_stage_data(s) for s in STAGES]
    for i, (s, d) in enumerate(zip(STAGES, stage_data)):
        vel = d['velocity']
        bg = d['backgroundfield']
        wf = d['wavefield']
        fq = d['freq_used']
        print(f'  Stage {i} ({s["dir"]}): vel={vel.shape}, bg={bg.shape}, '
              f'wf={wf.shape}, freq={fq.shape}')

    n_vel_per_stage = stage_data[0]['velocity'].shape[0]
    n_total_vel = n_vel_per_stage * N_FREQ

    # ---- 2. 加载合并数据 ----
    print('\n[2] 加载合并后的 freq3to25 数据 ...')
    combined = {}
    for key, fname in COMBINED_FILES.items():
        path = os.path.join(MULTIFREQ_ROOT, fname)
        combined[key] = load_mmap(path)
        print(f'  {key}: {combined[key].shape}')

    vel_combined = combined['velocity']
    bg_combined = combined['backgroundfield']
    wf_combined = combined['wavefield']
    freq_combined = combined['freq_used']

    # ---- 3. 验证 velocity 排列 ----
    print('\n[3] 验证 velocity 排列: 合并数据 = [stage0_vel, stage1_vel, stage2_vel]')
    vel_ok = True
    for i in range(N_FREQ):
        start = i * n_vel_per_stage
        end = start + n_vel_per_stage
        match = np.array_equal(np.array(vel_combined[start:end]),
                               np.array(stage_data[i]['velocity']))
        status = '✓ PASS' if match else '✗ FAIL'
        print(f'  Stage {i} vel [{start}:{end}] vs ground truth: {status}')
        vel_ok = vel_ok and match

    # ---- 4. 验证 freq 排列 ----
    print('\n[4] 验证 freq_used 排列: 合并数据 = [stage0_freq, stage1_freq, stage2_freq]')
    freq_ok = True
    for i in range(N_FREQ):
        start = i * n_vel_per_stage
        end = start + n_vel_per_stage
        match = np.array_equal(np.array(freq_combined[start:end]),
                               np.array(stage_data[i]['freq_used']))
        status = '✓ PASS' if match else '✗ FAIL'
        print(f'  Stage {i} freq [{start}:{end}] vs ground truth: {status}')
        freq_ok = freq_ok and match

    # ---- 5. 模拟 dataloader 重排并验证 backgroundfield / wavefield ----
    print('\n[5] 模拟 dataloader multi-freq 重排 + 震源拆分，逐条验证 ...')

    all_ok = True
    for field_name in ['backgroundfield', 'wavefield']:
        print(f'\n  --- {field_name} ---')

        # 模拟重排
        reshuffled = simulate_dataloader_reshuffle(combined[field_name], n_total_vel)
        # 模拟震源拆分
        locs = simulate_source_split(reshuffled, n_total_vel)
        # locs[s] 形状: [n_total_vel, 2, H, W]，其中 n_total_vel = n_freq * n_vel_per_stage
        # 排列: [freq0_0..freq0_N, freq1_0..freq1_N, freq2_0..freq2_N]

        for src in range(N_SRC):
            for f_idx in range(N_FREQ):
                freq_start = f_idx * n_vel_per_stage
                freq_end = freq_start + n_vel_per_stage

                # 从重排后的数据中取出: source=src, freq_stage=f_idx 对应的数据
                dataloader_slice = locs[src][freq_start:freq_end].numpy()

                # 从 ground truth 中取出: stage=f_idx, source=src 对应的数据
                gt_start = src * n_vel_per_stage
                gt_end = gt_start + n_vel_per_stage
                gt_slice = np.array(stage_data[f_idx][field_name][gt_start:gt_end])

                match = np.allclose(dataloader_slice, gt_slice, atol=0)

                status = '✓ PASS' if match else '✗ FAIL'
                print(f'    src={src}, freq_stage={f_idx} ({STAGES[f_idx]["dir"]}): {status}')
                all_ok = all_ok and match

                if not match:
                    diff = np.abs(dataloader_slice - gt_slice)
                    print(f'      max_diff={diff.max():.6e}, '
                          f'mean_diff={diff.mean():.6e}, '
                          f'mismatch_ratio={np.mean(diff > 0):.4f}')

    # ---- 6. 随机抽样深度验证（避免只检查边界） ----
    print('\n[6] 随机抽样深度验证（抽取 20 个随机 (src, stage, vel_idx) 组合）...')
    np.random.seed(42)
    bg_reshuffled = simulate_dataloader_reshuffle(bg_combined, n_total_vel)
    bg_locs = simulate_source_split(bg_reshuffled, n_total_vel)
    wf_reshuffled = simulate_dataloader_reshuffle(wf_combined, n_total_vel)
    wf_locs = simulate_source_split(wf_reshuffled, n_total_vel)

    sample_ok = True
    for _ in range(20):
        src = np.random.randint(N_SRC)
        f_idx = np.random.randint(N_FREQ)
        v_idx = np.random.randint(n_vel_per_stage)

        # dataloader 路径
        dl_idx = f_idx * n_vel_per_stage + v_idx
        dl_bg = bg_locs[src][dl_idx].numpy()
        dl_wf = wf_locs[src][dl_idx].numpy()

        # ground truth 路径
        gt_idx = src * n_vel_per_stage + v_idx
        gt_bg = np.array(stage_data[f_idx]['backgroundfield'][gt_idx])
        gt_wf = np.array(stage_data[f_idx]['wavefield'][gt_idx])

        bg_match = np.allclose(dl_bg, gt_bg, atol=0)
        wf_match = np.allclose(dl_wf, gt_wf, atol=0)

        if not (bg_match and wf_match):
            print(f'  ✗ src={src}, stage={f_idx}, vel_idx={v_idx}: '
                  f'bg_match={bg_match}, wf_match={wf_match}')
            sample_ok = False

    if sample_ok:
        print('  ✓ 全部 20 个随机样本验证通过')

    # ---- 7. 总结 ----
    print('\n' + '=' * 70)
    print('验证结果总结:')
    print(f'  velocity 排列:     {"✓ PASS" if vel_ok else "✗ FAIL"}')
    print(f'  freq_used 排列:    {"✓ PASS" if freq_ok else "✗ FAIL"}')
    print(f'  backgroundfield:   {"✓ PASS" if all_ok else "✗ FAIL"}')
    print(f'  wavefield:         {"✓ PASS" if all_ok else "✗ FAIL"}')
    print(f'  随机抽样验证:      {"✓ PASS" if sample_ok else "✗ FAIL"}')

    overall = vel_ok and freq_ok and all_ok and sample_ok
    print(f'\n  总体结论: {"✓ 全部一致，震源排列正确" if overall else "✗ 存在不一致！"}')
    print('=' * 70)


if __name__ == '__main__':
    main()
