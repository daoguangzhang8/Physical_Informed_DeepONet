"""
测试脚本：通过 prepare_training_dataloaders 生成与训练时完全一致的数据集，
验证散射波场的震源位置与 config.source_list 配置一致。
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Args
from model.dataloader import prepare_training_dataloaders
import torch


def main():
    args = Args()
    device = torch.device('cpu')

    # 直接调用 dataloader 的完整流水线，生成与训练时完全一致的数据
    train_loaders, plot_data = prepare_training_dataloaders(args, device)

    # 从 DataLoader 中取出全部训练数据
    ds = train_loaders['train'].dataset
    vel_all = ds.tensors[0]       # [N, 1, nz, nx]
    UU0_all = ds.tensors[1]       # [N, 2, nz, nx]
    labels_all = ds.tensors[2]    # [N, 2, nz, nx]
    freq_all = ds.tensors[3]      # [N]

    print(f'\n=== 训练集数据 ===')
    print(f'vel: {vel_all.shape}, UU0: {UU0_all.shape}, labels: {labels_all.shape}, freq: {freq_all.shape}')

    # PML active 裁切（与 plotting.py 一致）
    L = args.pml_active
    if args.boundary_type == 'free_surface':
        z_sl, x_sl = slice(0, -L), slice(L, -L)
    else:
        z_sl, x_sl = slice(L, -L), slice(L, -L)

    out_dir = 'output_test_dataloader'
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. 震源位置检测：对每个样本的背景场顶部求能量峰值行
    # ------------------------------------------------------------------
    print(f'\n=== 震源位置一致性检测 ===')
    print(f'config.source_list = {args.source_list}')

    # 用背景场检测震源 x 坐标（取每个样本顶部 3 行的绝对值之和的峰值）
    src_x_list = []
    for i in range(len(UU0_all)):
        bg = UU0_all[i, 0].numpy()
        top_energy = np.sum(np.abs(bg[:3, :]), axis=0)
        src_x_list.append(np.argmax(top_energy))
    src_x_arr = np.array(src_x_list)

    # 期望的震源位置（从 src0 的背景场确定）
    ref_bg = UU0_all[0, 0].numpy()
    ref_top = np.sum(np.abs(ref_bg[:3, :]), axis=0)
    expected_x = np.argmax(ref_top)
    n_mismatch = np.sum(src_x_arr != expected_x)
    print(f'参考震源 x 位置 (src{args.source_list[0]}): {expected_x}')
    print(f'所有 {len(src_x_arr)} 个样本中震源 x 不匹配数: {n_mismatch}')

    if n_mismatch > 0:
        mismatch_idx = np.where(src_x_arr != expected_x)[0]
        print(f'不匹配样本索引 (前20): {mismatch_idx[:20]}')
        for idx in mismatch_idx[:5]:
            print(f'  sample {idx}: x={src_x_arr[idx]}, freq={freq_all[idx].item():.1f}Hz')
    else:
        print('✓ 全部样本震源位置一致')

    # ------------------------------------------------------------------
    # 2. 可视化：按频率分组输出散射波场
    # ------------------------------------------------------------------
    unique_freqs = freq_all.unique().sort()[0].tolist()
    print(f'\n频率值 ({len(unique_freqs)}): {unique_freqs}')
    print(f'开始输出散射波场图片到 {out_dir}/')

    for fv in unique_freqs:
        mask = (freq_all == fv).nonzero(as_tuple=True)[0]
        n_show = min(6, len(mask))
        pick = mask[torch.randperm(len(mask))[:n_show]]

        fig, axes = plt.subplots(2, n_show, figsize=(3.5 * n_show, 7))
        fig.suptitle(f'Scattered Field | freq={fv:.1f}Hz | source_list={args.source_list}',
                     fontsize=12, fontweight='bold')

        for col, idx in enumerate(pick):
            lab = labels_all[idx].numpy()
            real_c = lab[0, z_sl, x_sl]
            imag_c = lab[1, z_sl, x_sl]
            rmax = max(np.abs(real_c).max(), 1e-8)
            imax = max(np.abs(imag_c).max(), 1e-8)

            axes[0, col].imshow(real_c, aspect='auto', cmap='seismic', vmin=-rmax, vmax=rmax)
            axes[0, col].set_title(f'#{idx.item()} real', fontsize=8)
            axes[1, col].imshow(imag_c, aspect='auto', cmap='seismic', vmin=-imax, vmax=imax)
            axes[1, col].set_title(f'#{idx.item()} imag', fontsize=8)

        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(f'{out_dir}/scatter_freq{fv:.1f}Hz.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------------------------------
    # 3. 总览：速度 + 背景场 + 散射场
    # ------------------------------------------------------------------
    n_panels = 20
    step = max(1, len(labels_all) // n_panels)
    indices = list(range(0, len(labels_all), step))[:n_panels]

    fig, axes = plt.subplots(3, n_panels, figsize=(3 * n_panels, 9))
    fig.suptitle(f'Velocity / Background / Scatter | source_list={args.source_list}',
                 fontsize=12, fontweight='bold')
    for col, idx in enumerate(indices):
        v = vel_all[idx, 0].numpy()
        axes[0, col].imshow(v, aspect='auto', cmap='viridis')
        axes[0, col].set_title(f'#{idx}', fontsize=7)

        bg = UU0_all[idx, 0].numpy()[z_sl, x_sl]
        bm = max(np.abs(bg).max(), 1e-8)
        axes[1, col].imshow(bg, aspect='auto', cmap='seismic', vmin=-bm, vmax=bm)

        lab = labels_all[idx].numpy()[0, z_sl, x_sl]
        rm = max(np.abs(lab).max(), 1e-8)
        fstr = f'{freq_all[idx].item():.1f}Hz'
        axes[2, col].imshow(lab, aspect='auto', cmap='seismic', vmin=-rm, vmax=rm)
        axes[2, col].set_title(fstr, fontsize=7)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'{out_dir}/overview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. 连续样本检查
    # ------------------------------------------------------------------
    n_check = 30
    fig, axes = plt.subplots(2, n_check, figsize=(2 * n_check, 5))
    fig.suptitle(f'First {n_check} Consecutive Samples', fontsize=11, fontweight='bold')
    for i in range(n_check):
        lab = labels_all[4000 + i].numpy()[0, z_sl, x_sl]
        rm = max(np.abs(lab).max(), 1e-8)
        axes[0, i].imshow(lab, aspect='auto', cmap='seismic', vmin=-rm, vmax=rm)
        axes[0, i].set_title(f'{freq_all[4000 + i].item():.0f}Hz', fontsize=6)
        axes[1, i].imshow(vel_all[4000 + i, 0].numpy(), aspect='auto', cmap='viridis')
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel('Scatter Real', fontsize=8)
    axes[1, 0].set_ylabel('Velocity', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/consecutive.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'\n全部图片已保存到 {out_dir}/')
    print(f'  scatter_freq*.png  — 每个频率 6 个样本')
    print(f'  overview.png       — 速度/背景场/散射场总览')
    print(f'  consecutive.png    — 前 30 个连续样本')


if __name__ == '__main__':
    main()
