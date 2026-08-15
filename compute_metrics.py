import os
import sys
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from metrics.loe import compute_loe

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


def calculate_psnr(img1, img2):
    # img1 and img2 in float [0, 1]
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(1.0 / np.sqrt(mse)))


def calculate_ssim(img1, img2):
    if _HAS_SKIMAGE:
        # channel_axis=2 for multichannel RGB
        return float(ssim_fn(img1, img2, data_range=1.0, channel_axis=2))
    else:
        # Fallback simple SSIM approximation
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1_sq = ((img1 - mu1) ** 2).mean()
        sigma2_sq = ((img2 - mu2) ** 2).mean()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(ssim)


def evaluate_results(results_dir="results/LOLv1", gt_dir="LOLdataset/our485/high"):
    if not os.path.isdir(results_dir):
        print(f"Error: Results directory {results_dir} not found. Run evaluate.py first.")
        return

    result_files = sorted([f for f in os.listdir(results_dir) if f.endswith((".png", ".jpg"))])
    if not result_files:
        print(f"No image files found in {results_dir}")
        return

    psnr_list, ssim_list, loe_list = [], [], []

    print(f"\nEvaluating {len(result_files)} restored images in {results_dir} against GT in {gt_dir}...\n")
    print(f"{'Image':<12} {'PSNR (dB)':<12} {'SSIM':<12} {'LOE':<12}")
    print("-" * 50)

    for fname in result_files:
        res_path = os.path.join(results_dir, fname)
        gt_path = os.path.join(gt_dir, fname)

        if not os.path.isfile(gt_path):
            # check alternate eval15/high
            gt_path = os.path.join("LOLdataset/eval15/high", fname)
            if not os.path.isfile(gt_path):
                continue

        res_img = np.array(Image.open(res_path).convert("RGB"), dtype=np.float32) / 255.0
        gt_img = np.array(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0

        # Match dimensions if needed
        if res_img.shape != gt_img.shape:
            h = min(res_img.shape[0], gt_img.shape[0])
            w = min(res_img.shape[1], gt_img.shape[1])
            res_img = res_img[:h, :w]
            gt_img = gt_img[:h, :w]

        cur_psnr = calculate_psnr(res_img, gt_img)
        cur_ssim = calculate_ssim(res_img, gt_img)
        cur_loe = compute_loe(gt_img, res_img)

        psnr_list.append(cur_psnr)
        ssim_list.append(cur_ssim)
        loe_list.append(cur_loe)

        print(f"{fname:<12} {cur_psnr:<12.3f} {cur_ssim:<12.4f} {cur_loe:<12.2f}")

    print("=" * 50)
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_loe = np.mean(loe_list)
    print(f"{'AVERAGE':<12} {avg_psnr:<12.3f} {avg_ssim:<12.4f} {avg_loe:<12.2f}")
    print("=" * 50)


if __name__ == "__main__":
    evaluate_results()
