import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = "/home/sharedata/zdg/multifreq_merged1"
OUT_ROOT = "/home/zhangdaoguang/Code/modeling_scatter_qc_seismic"


def load_category_names(root):
    path = os.path.join(root, "category_names.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [raw[str(i)] for i in range(len(raw))]


def robust_limits(values, low=1.0, high=99.0):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [low, high])
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def symmetric_limits(values, percentile=99.0):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    vmax = np.percentile(np.abs(finite), percentile)
    if vmax == 0:
        vmax = 1.0
    return float(-vmax), float(vmax)


def plot_group(velocity, wavefield, background, freq_used, indices, freq_idx, freq,
               source_idx, category_name, out_path, n_samples, rng):
    if len(indices) > n_samples:
        chosen = np.sort(rng.choice(indices, size=n_samples, replace=False))
    else:
        chosen = np.asarray(indices)

    n = len(chosen)
    pairs_per_row = 5
    n_rows = int(np.ceil(n / pairs_per_row))
    fig, axes = plt.subplots(
        n_rows,
        pairs_per_row * 2,
        figsize=(pairs_per_row * 3.6, n_rows * 1.9),
        squeeze=False,
    )

    vel_slices = velocity[chosen, freq_idx]
    scatter_slices = []
    for sample_idx in chosen:
        sc = wavefield[sample_idx, freq_idx, source_idx] - background[sample_idx, freq_idx, source_idx]
        scatter_slices.append(sc[0])
    scatter_slices = np.asarray(scatter_slices)

    vel_vmin, vel_vmax = robust_limits(vel_slices)
    sc_vmin, sc_vmax = symmetric_limits(scatter_slices, 99.0)

    for ax in axes.ravel():
        ax.axis("off")

    for local_i, sample_idx in enumerate(chosen):
        row = local_i // pairs_per_row
        pair = local_i % pairs_per_row
        ax_v = axes[row, pair * 2]
        ax_s = axes[row, pair * 2 + 1]

        vel_img = vel_slices[local_i] / 1000.0
        sc_img = scatter_slices[local_i]

        ax_v.imshow(vel_img, cmap="turbo", aspect="auto", vmin=vel_vmin / 1000.0, vmax=vel_vmax / 1000.0)
        ax_s.imshow(sc_img, cmap="seismic", aspect="auto", vmin=sc_vmin, vmax=sc_vmax)

        ax_v.set_title(f"#{sample_idx} vel", fontsize=7)
        actual_freq = freq_used[sample_idx, freq_idx]
        ax_s.set_title(f"{actual_freq:g}Hz src{source_idx + 1} real", fontsize=7)

    fig.suptitle(
        f"{category_name} | scattered real wavefield | freq={freq:g} Hz | source={source_idx + 1} | samples={n}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return n


def main():
    parser = argparse.ArgumentParser(description="Plot velocity and scattered wavefield QC montages.")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--out-root", default=OUT_ROOT)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260529)
    args = parser.parse_args()

    velocity = np.load(os.path.join(args.data_root, "freesurface_full_5sources_velocity.npy"), mmap_mode="r")
    wavefield = np.load(os.path.join(args.data_root, "freesurface_full_5sources_wavefield.npy"), mmap_mode="r")
    background = np.load(os.path.join(args.data_root, "freesurface_full_5sources_background.npy"), mmap_mode="r")
    freq_used = np.load(os.path.join(args.data_root, "freesurface_full_5sources_freq_used.npy"), mmap_mode="r")
    model_category = np.load(os.path.join(args.data_root, "model_category.npy"), mmap_mode="r")
    category_names = load_category_names(args.data_root)

    if wavefield.shape != background.shape:
        raise ValueError(f"wavefield shape {wavefield.shape} != background shape {background.shape}")
    if velocity.shape[:2] != wavefield.shape[:2]:
        raise ValueError(f"velocity shape {velocity.shape} incompatible with wavefield {wavefield.shape}")

    rng = np.random.default_rng(args.seed)
    summary = []

    for category_id, category_name in enumerate(category_names):
        category_indices = np.where(model_category == category_id)[0]
        for freq_idx in range(freq_used.shape[1]):
            freqs = np.unique(freq_used[category_indices, freq_idx])
            for freq in freqs:
                group_indices = category_indices[freq_used[category_indices, freq_idx] == freq]
                if len(group_indices) < args.samples:
                    print(
                        f"warning: {category_name} freq={freq:g}Hz has only "
                        f"{len(group_indices)} samples; plotting all of them"
                    )
                for source_idx in range(wavefield.shape[2]):
                    out_path = os.path.join(
                        args.out_root,
                        category_name,
                        f"freq_{freq:g}Hz",
                        f"src_{source_idx + 1:02d}.png",
                    )
                    plotted = plot_group(
                        velocity, wavefield, background, freq_used, group_indices,
                        freq_idx, float(freq), source_idx, category_name, out_path,
                        args.samples, rng,
                    )
                    summary.append((category_name, float(freq), source_idx + 1, plotted, out_path))
                    print(f"saved {out_path} ({plotted} samples)")

    summary_path = os.path.join(args.out_root, "summary.csv")
    os.makedirs(args.out_root, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("category,freq_hz,source,n_samples,path\n")
        for row in summary:
            f.write(f"{row[0]},{row[1]:g},{row[2]},{row[3]},{row[4]}\n")
    print(f"summary saved to {summary_path}")


if __name__ == "__main__":
    main()
