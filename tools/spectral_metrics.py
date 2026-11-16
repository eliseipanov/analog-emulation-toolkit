#!/usr/bin/env python3
"""Standalone dev script: compare high-frequency noise statistics between
real photos and AI-generated images in the frequency domain.

Usage:
    python tools/spectral_metrics.py --real debug/REAL --gen debug/GEN --out debug/out
"""

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import windows


def discover_images(directory):
    """Return sorted list of image file paths in *directory*."""
    exts = (".jpg", ".jpeg", ".png")
    files = []
    if not os.path.isdir(directory):
        return files
    for name in os.listdir(directory):
        lower = name.lower()
        if lower.endswith(exts):
            files.append(os.path.join(directory, name))
    files.sort(key=lambda p: os.path.basename(p))
    return files


def load_image(path):
    """Load image, convert to grayscale, optionally downscale to max side 1024,
    return float32 array in [0, 1]. Never upscales."""
    img = Image.open(path).convert("L")
    w, h = img.size
    max_side = max(w, h)
    if max_side > 1024:
        scale = 1024.0 / max_side
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.BOX)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def radial_power_spectrum(arr):
    """Compute radially averaged power spectrum.

    Returns freqs (bin centers over [0, 0.5]) and power (average power per bin).
    """
    h, w = arr.shape
    # Subtract mean
    arr = arr - arr.mean()
    # 2D Hann window (outer product of two 1D Hann windows)
    hann_y = windows.hann(h, sym=False)
    hann_x = windows.hann(w, sym=False)
    window = np.outer(hann_y, hann_x).astype(np.float32)
    arr = arr * window
    # 2D FFT
    F = np.fft.fftshift(np.fft.fft2(arr))
    power = np.abs(F) ** 2
    # Radial distance from center, normalized so Nyquist = 0.5
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt(((yy - cy) / h) ** 2 + ((xx - cx) / w) ** 2)
    # 64 radial bins over [0, 0.5]
    n_bins = 64
    bin_edges = np.linspace(0.0, 0.5, n_bins + 1)
    r_flat = r.ravel()
    p_flat = power.ravel()
    bin_idx = np.minimum(np.floor(r_flat / 0.5 * n_bins).astype(int), n_bins - 1)
    bin_sum = np.bincount(bin_idx, weights=p_flat, minlength=n_bins)
    bin_count = np.bincount(bin_idx, minlength=n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_power = np.where(bin_count > 0, bin_sum / np.maximum(bin_count, 1), 0.0)
    freqs = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return freqs, avg_power


def spectral_distance(freqs_real, power_real, freqs_gen, power_gen):
    """Compute spectral distance metrics restricted to band [0.1, 0.5]."""
    band_mask = (freqs_real >= 0.1) & (freqs_real <= 0.5)
    fr = freqs_real[band_mask]
    pr = power_real[band_mask]
    pg = power_gen[band_mask]

    log_pr = np.log10(pr + 1e-12)
    log_pg = np.log10(pg + 1e-12)

    diff = log_pg - log_pr
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mean_diff_db = float(np.mean(diff) * 10.0)

    # Linear fit slope of log10(power) vs log10(freq)
    # Guard against zero/negative freqs
    mask_r = fr > 0
    mask_g = fr > 0
    if np.count_nonzero(mask_r) >= 2:
        slope_real = float(np.polyfit(np.log10(fr[mask_r]), log_pr[mask_r], 1)[0])
    else:
        slope_real = 0.0
    if np.count_nonzero(mask_g) >= 2:
        slope_gen = float(np.polyfit(np.log10(fr[mask_g]), log_pg[mask_g], 1)[0])
    else:
        slope_gen = 0.0
    slope_delta = slope_gen - slope_real

    return rmse, mean_diff_db, slope_real, slope_gen, slope_delta


def save_plot(freqs_real, power_real, freqs_gen, power_gen, rmse, mean_diff_db,
              slope_real, slope_gen, slope_delta, out_path):
    """Save log-log spectrum comparison plot."""
    plt.figure(figsize=(6, 4))
    mask_r = freqs_real > 0
    mask_g = freqs_gen > 0
    plt.loglog(freqs_real[mask_r], power_real[mask_r] + 1e-12, label="real", color="C0")
    plt.loglog(freqs_gen[mask_g], power_gen[mask_g] + 1e-12, label="gen", color="C1")
    plt.axvspan(0.1, 0.5, color="gray", alpha=0.2, label="0.1–0.5 band")
    plt.xlabel("Normalized frequency (Nyquist = 0.5)")
    plt.ylabel("Power")
    plt.title(f"RMSE={rmse:.4f}  mean_diff_db={mean_diff_db:.2f}  "
              f"slope_real={slope_real:.3f}  slope_gen={slope_gen:.3f}  delta={slope_delta:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare spectral noise stats between real and generated images."
    )
    parser.add_argument("--real", required=True, help="Directory with real photos")
    parser.add_argument("--gen", required=True, help="Directory with generated images")
    parser.add_argument("--out", required=True, help="Output directory for plots and CSV")
    args = parser.parse_args()

    real_files = discover_images(args.real)
    gen_files = discover_images(args.gen)

    if not real_files and not gen_files:
        print("No image files found in either directory.")
        return

    n_pairs = min(len(real_files), len(gen_files))
    if n_pairs == 0:
        print("No matching image pairs to process.")
        return

    os.makedirs(args.out, exist_ok=True)

    rows = []
    for i in range(n_pairs):
        real_path = real_files[i]
        gen_path = gen_files[i]
        real_name = os.path.basename(real_path)
        gen_name = os.path.basename(gen_path)

        real_arr = load_image(real_path)
        gen_arr = load_image(gen_path)

        freqs_real, power_real = radial_power_spectrum(real_arr)
        freqs_gen, power_gen = radial_power_spectrum(gen_arr)

        rmse, mean_diff_db, slope_real, slope_gen, slope_delta = spectral_distance(
            freqs_real, power_real, freqs_gen, power_gen
        )

        plot_name = f"spectrum_compare_{i:02d}.png"
        plot_path = os.path.join(args.out, plot_name)
        save_plot(freqs_real, power_real, freqs_gen, power_gen,
                  rmse, mean_diff_db, slope_real, slope_gen, slope_delta, plot_path)

        rows.append({
            "real_file": real_name,
            "gen_file": gen_name,
            "rmse": rmse,
            "mean_diff_db": mean_diff_db,
            "slope_real": slope_real,
            "slope_gen": slope_gen,
            "slope_delta": slope_delta,
        })

        print(f"[{i+1}/{n_pairs}] {real_name} vs {gen_name}: "
              f"rmse={rmse:.4f} mean_diff_db={mean_diff_db:.2f} "
              f"slope_delta={slope_delta:.3f}")

    csv_path = os.path.join(args.out, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "real_file", "gen_file", "rmse", "mean_diff_db",
            "slope_real", "slope_gen", "slope_delta",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPlots saved to: {args.out}")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
