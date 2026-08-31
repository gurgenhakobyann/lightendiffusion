"""
Full Multi-Benchmark Evaluator for LightenDiffusion.

Computes the standard paper evaluation table across:
  - LOL [58] : PSNR ↑, SSIM ↑, LPIPS ↓
  - LSRW [16]: PSNR ↑, SSIM ↑, LPIPS ↓
  - DICM [28]: NIQE ↓, PI ↓
  - NPE [53] : NIQE ↓, PI ↓
  - VV [51]  : NIQE ↓, PI ↓
"""

import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from metrics.loe import compute_loe
from metrics.niqe import compute_niqe, compute_pi
from metrics.lpips_metric import compute_lpips

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(1.0 / np.sqrt(mse)))


def calculate_ssim(img1, img2):
    if _HAS_SKIMAGE:
        return float(ssim_fn(img1, img2, data_range=1.0, channel_axis=2))
    return 0.0


def eval_paired_dataset(results_dir, gt_dir, name="LOL"):
    if not os.path.isdir(results_dir):
        return None
    res_files = sorted([f for f in os.listdir(results_dir) if f.endswith((".png", ".jpg", ".bmp"))])
    if not res_files:
        return None

    psnr_list, ssim_list, lpips_list, loe_list = [], [], [], []
    for f in res_files:
        r_p = os.path.join(results_dir, f)
        g_p = os.path.join(gt_dir, f)
        if not os.path.isfile(g_p):
            # check alternate name extensions
            base = os.path.splitext(f)[0]
            cand = glob.glob(os.path.join(gt_dir, base + ".*"))
            if cand:
                g_p = cand[0]
            else:
                continue

        r_img = np.array(Image.open(r_p).convert("RGB"), dtype=np.float32) / 255.0
        g_img = np.array(Image.open(g_p).convert("RGB"), dtype=np.float32) / 255.0

        h = min(r_img.shape[0], g_img.shape[0])
        w = min(r_img.shape[1], g_img.shape[1])
        r_img, g_img = r_img[:h, :w], g_img[:h, :w]

        psnr_list.append(calculate_psnr(r_img, g_img))
        ssim_list.append(calculate_ssim(r_img, g_img))
        lpips_list.append(compute_lpips(r_img, g_img))
        loe_list.append(compute_loe(g_img, r_img))

    if not psnr_list:
        return None

    return {
        "name": name,
        "count": len(psnr_list),
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
        "lpips": float(np.mean(lpips_list)),
        "loe": float(np.mean(loe_list))
    }


def eval_unpaired_dataset(results_dir, name="DICM"):
    if not os.path.isdir(results_dir):
        return None
    res_files = sorted([f for f in os.listdir(results_dir) if f.endswith((".png", ".jpg", ".bmp", ".JPG"))])
    if not res_files:
        return None

    niqe_list, pi_list = [], []
    for f in res_files:
        r_p = os.path.join(results_dir, f)
        img = Image.open(r_p).convert("RGB")
        img_np = np.array(img, dtype=np.float32) / 255.0

        n_score = compute_niqe(img_np)
        p_score = compute_pi(img_np, niqe_score=n_score)
        niqe_list.append(n_score)
        pi_list.append(p_score)

    if not niqe_list:
        return None

    return {
        "name": name,
        "count": len(niqe_list),
        "niqe": float(np.mean(niqe_list)),
        "pi": float(np.mean(pi_list))
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate full benchmarks matching ECCV paper table")
    parser.add_argument("--results_root", default="results", type=str)
    parser.add_argument("--data_root", default=".", type=str)
    args = parser.parse_args()

    print("\n" + "=" * 90)
    print(f"{'FULL BENCHMARK EVALUATION (MATCHING ECCV PAPER TABLE)':^90}")
    print("=" * 90)

    # 1. LOL-v1
    lol_gt = os.path.join(args.data_root, "LOLdataset", "our485", "high")
    if not os.path.isdir(lol_gt):
        lol_gt = os.path.join(args.data_root, "LOLdataset", "eval15", "high")
    lol_res = eval_paired_dataset(os.path.join(args.results_root, "LOLv1"), lol_gt, name="LOL")

    # 2. LSRW
    lsrw_gt = os.path.join(args.data_root, "LSRW", "Eval", "Eval", "Huawei", "high")
    if not os.path.isdir(lsrw_gt):
        lsrw_gt = os.path.join(args.data_root, "LSRW", "high")
    lsrw_res = eval_paired_dataset(os.path.join(args.results_root, "LSRW"), lsrw_gt, name="LSRW")

    # 3. DICM, NPE, VV
    dicm_res = eval_unpaired_dataset(os.path.join(args.results_root, "DICM"), name="DICM")
    npe_res = eval_unpaired_dataset(os.path.join(args.results_root, "NPE"), name="NPE")
    vv_res = eval_unpaired_dataset(os.path.join(args.results_root, "VV"), name="VV")

    # Master Table Formatting
    print("\n" + "-" * 90)
    header1 = f"{'Method':<20} | {'LOL':^24} | {'LSRW':^24} | {'DICM':^14} | {'NPE':^14} | {'VV':^14}"
    header2 = f"{'':<20} | {'PSNR↑':>7} {'SSIM↑':>7} {'LPIPS↓':>8} | {'PSNR↑':>7} {'SSIM↑':>7} {'LPIPS↓':>8} | {'NIQE↓':>6} {'PI↓':>6} | {'NIQE↓':>6} {'PI↓':>6} | {'NIQE↓':>6} {'PI↓':>6}"
    print(header1)
    print(header2)
    print("-" * 90)

    # Values
    lol_str = f"{lol_res['psnr']:>7.3f} {lol_res['ssim']:>7.4f} {lol_res['lpips']:>8.3f}" if lol_res else f"{'--':>7} {'--':>7} {'--':>8}"
    lsrw_str = f"{lsrw_res['psnr']:>7.3f} {lsrw_res['ssim']:>7.4f} {lsrw_res['lpips']:>8.3f}" if lsrw_res else f"{'--':>7} {'--':>7} {'--':>8}"
    dicm_str = f"{dicm_res['niqe']:>6.3f} {dicm_res['pi']:>6.3f}" if dicm_res else f"{'--':>6} {'--':>6}"
    npe_str = f"{npe_res['niqe']:>6.3f} {npe_res['pi']:>6.3f}" if npe_res else f"{'--':>6} {'--':>6}"
    vv_str = f"{vv_res['niqe']:>6.3f} {vv_res['pi']:>6.3f}" if vv_res else f"{'--':>6} {'--':>6}"

    print(f"{'LightenDiffusion':<20} | {lol_str} | {lsrw_str} | {dicm_str} | {npe_str} | {vv_str}")
    print("-" * 90)

    if lol_res:
        print(f"\n[LOL Detailed] PSNR: {lol_res['psnr']:.3f} dB | SSIM: {lol_res['ssim']:.4f} | LPIPS: {lol_res['lpips']:.3f} | LOE: {lol_res['loe']:.2f}")


if __name__ == "__main__":
    main()
