"""
验证 prepare_multifreq_selected.py 生成的各阶段数据的震源排列是否正确。

验证思路：
  1. 复现 prepare_multifreq_selected.py 的选取逻辑，得到 selected_idx
  2. 从原始数据 /home/sharedata/zdg/multifreq/ 按索引抽取作为 ground truth
  3. 从选取后数据 /home/sharedata/zdg/multifreq_selected/ 按震源拆分读取
  4. 对比两者在若干采样点 (stage, source, vel_idx) 上是否一致
"""

import os
import numpy as np

SRC_DATA = '/home/sharedata/zdg/multifreq'
DST_DATA = '/home/sharedata/zdg/multifreq_selected'
SEED = 1
N_SELECT = 2000
N_SRC = 5

STAGES = [
    {'dir': 'freq_3to11', 'tag': 'freq3to11'},
    {'dir': 'freq_12to18', 'tag': 'freq12to18'},
    {'dir': 'freq_18to25', 'tag': 'freq18to25'},
]


def load_mmap(path):
    return np.load(path, mmap_mode='r')


def get_field_paths(stage):
    """返回某阶段的 4 个文件路径 (原始, 选取后)"""
    tag = stage['tag']
    names = {
        'velocity':        f'freesurface_velocity_{tag}_5sources_160_180_pml20_n1.npy',
        'backgroundfield': f'freesurface_backgroundfield_{tag}_5sources_160_180_pml20_n1.npy',
        'wavefield':       f'freesurface_wavefield_{tag}_5sources_160_180_pml20_n1.npy',
        'freq_used':       f'freesurface_freq_used_5sources_160_180_pml20_n1.npy',
    }
    src_paths = {k: os.path.join(SRC_DATA, stage['dir'], v) for k, v in names.items()}
    dst_paths = {k: os.path.join(DST_DATA, stage['dir'], v) for k, v in names.items()}
    return src_paths, dst_paths


def main():
    # ---- 1. 复现选取索引 ----
    vel0 = load_mmap(os.path.join(SRC_DATA, STAGES[0]['dir'],
                                   f'freesurface_velocity_{STAGES[0]["tag"]}_5sources_160_180_pml20_n1.npy'))
    n_total = vel0.shape[0]
    np.random.seed(SEED)
    selected_idx = np.sort(np.random.choice(n_total, N_SELECT, replace=False))
    print(f'原始数据总量: {n_total}, 选取: {N_SELECT}')
    print(f'selected_idx 范围: [{selected_idx[0]}, {selected_idx[-1]}]')

    # ---- 2. 抽样验证 ----
    np.random.seed(99)
    n_samples = 20
    test_cases = [(np.random.randint(N_SRC),
                   np.random.randint(N_SELECT)) for _ in range(n_samples)]

    print(f'\n{"="*70}')
    print(f'逐阶段验证: 从原始数据按 selected_idx 抽取 vs 选取后文件按震源拆分')
    print(f'{"="*70}')

    overall_ok = True

    for stage in STAGES:
        src_paths, dst_paths = get_field_paths(stage)
        print(f'\n--- Stage: {stage["dir"]} ---')

        # 加载原始 & 选取后的数据 (mmap)
        src_vel = load_mmap(src_paths['velocity'])
        src_bg  = load_mmap(src_paths['backgroundfield'])
        src_wf  = load_mmap(src_paths['wavefield'])
        src_fq  = load_mmap(src_paths['freq_used'])

        dst_vel = load_mmap(dst_paths['velocity'])
        dst_bg  = load_mmap(dst_paths['backgroundfield'])
        dst_wf  = load_mmap(dst_paths['wavefield'])
        dst_fq  = load_mmap(dst_paths['freq_used'])

        print(f'  原始: vel={src_vel.shape}, bg={src_bg.shape}, wf={src_wf.shape}')
        print(f'  选取: vel={dst_vel.shape}, bg={dst_bg.shape}, wf={dst_wf.shape}')

        # (a) velocity: dst_vel[v] == src_vel[selected_idx[v]]
        vel_ok = np.array_equal(np.array(dst_vel), np.array(src_vel[selected_idx]))
        print(f'  velocity 一致: {"✓" if vel_ok else "✗ FAIL"}')
        overall_ok = overall_ok and vel_ok

        # (b) freq_used: dst_fq[v] == src_fq[selected_idx[v]]
        fq_ok = np.array_equal(np.array(dst_fq), np.array(src_fq[selected_idx]))
        print(f'  freq_used 一致: {"✓" if fq_ok else "✗ FAIL"}')
        overall_ok = overall_ok and fq_ok

        # (c) backgroundfield / wavefield: 逐 (source, vel_idx) 抽样
        for field_name in ['backgroundfield', 'wavefield']:
            src_field = load_mmap(src_paths[field_name])
            dst_field = load_mmap(dst_paths[field_name])

            field_ok = True
            for src_i, v_i in test_cases:
                # 选取后数据按震源排列: [src0×N, src1×N, ..., src4×N]
                # dst 中 source=src_i, vel_idx=v_i 的位置: src_i * N_SELECT + v_i
                dst_val = dst_field[src_i * N_SELECT + v_i]

                # 原始数据按震源排列: [src0×n_total, src1×n_total, ..., src4×n_total]
                # src 中 source=src_i, 选取第 v_i 个速度模型的位置:
                #   src_i * n_total + selected_idx[v_i]
                src_val = src_field[src_i * n_total + selected_idx[v_i]]

                if not np.array_equal(dst_val, src_val):
                    field_ok = False
                    print(f'    ✗ {field_name} MISMATCH: src={src_i}, v_idx={v_i}, '
                          f'orig_offset={src_i * n_total + selected_idx[v_i]}, '
                          f'dst_offset={src_i * N_SELECT + v_i}')

            status = '✓' if field_ok else '✗ FAIL'
            print(f'  {field_name} 抽样验证 ({n_samples} 组): {status}')
            overall_ok = overall_ok and field_ok

    # ---- 3. 验证 dataloader 式震源拆分 ----
    print(f'\n{"="*70}')
    print('验证 dataloader 式拆分: UU_loc[s] = field[s * N_vel : (s+1) * N_vel]')
    print(f'{"="*70}')

    for stage in STAGES:
        _, dst_paths = get_field_paths(stage)
        dst_bg = load_mmap(dst_paths['backgroundfield'])
        dst_wf = load_mmap(dst_paths['wavefield'])

        n_vel = dst_vel.shape[0]  # 2000
        print(f'\n  {stage["dir"]}: N_vel={n_vel}, field shape={dst_bg.shape}')

        split_ok = True
        for src_i in range(N_SRC):
            bg_loc = dst_bg[src_i * n_vel : (src_i + 1) * n_vel]
            wf_loc = dst_wf[src_i * n_vel : (src_i + 1) * n_vel]
            if bg_loc.shape[0] != n_vel or wf_loc.shape[0] != n_vel:
                split_ok = False
                print(f'    ✗ src={src_i}: shape 不正确 bg={bg_loc.shape}, wf={wf_loc.shape}')

        status = '✓' if split_ok else '✗ FAIL'
        print(f'  5 震源拆分形状正确: {status}')
        overall_ok = overall_ok and split_ok

    # ---- 总结 ----
    print(f'\n{"="*70}')
    print(f'总体结论: {"✓ 全部一致，震源拆分正确" if overall_ok else "✗ 存在不一致！"}')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
