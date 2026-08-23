import os
import glob
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from metrics.loe import compute_loe


def main():
    restored_dir = os.path.join("results", "LOLv1")
    if not os.path.isdir(restored_dir):
        # Fallback check
        restored_dir = "results"

    gt_dir = os.path.join("LOLdataset", "our485", "high")
    if not os.path.isdir(gt_dir):
        gt_dir = os.path.join("LOLdataset", "eval15", "high")

    if not os.path.isdir(gt_dir):
        print(f"Error: GT directory not found at {gt_dir}")
        return

    restored_files = sorted(glob.glob(os.path.join(restored_dir, "*.png")) + glob.glob(os.path.join(restored_dir, "*.jpg")))
    if not restored_files:
        print(f"Error: No restored images found in {restored_dir}")
        return

    print(f"\nEvaluating {len(restored_files)} restored images in {restored_dir} against GT in {gt_dir}...\n")
    print(f"{'Image':<12} {'PSNR (dB)':<12} {'SSIM':<12} {'LOE':<12}")
    print("-" * 50)

    psnr_list, ssim_list, loe_list = [], [], []

    for r_path in restored_files:
        fname = os.path.basename(r_path)
        gt_path = os.path.join(gt_dir, fname)

        if not os.path.isfile(gt_path):
            continue

        r_img = np.array(Image.open(r_path).convert("RGB"))
        gt_img = np.array(Image.open(gt_path).convert("RGB"))

        # Resize if dimensions differ slightly due to padding
        if r_img.shape != gt_img.shape:
            h, w = min(r_img.shape[0], gt_img.shape[0]), min(r_img.shape[1], gt_img.shape[1])
            r_img = r_img[:h, :w]
            gt_img = gt_img[:h, :w]

        cur_psnr = psnr(gt_img, r_img, data_range=255)
        cur_ssim = ssim(gt_img, r_img, channel_axis=2, data_range=255)
        cur_loe = compute_loe(r_img, gt_img)

        psnr_list.append(cur_psnr)
        ssim_list.append(cur_ssim)
        loe_list.append(cur_loe)

        print(f"{fname:<12} {cur_psnr:<12.3f} {cur_ssim:<12.4f} {cur_loe:<12.2f}")

    if psnr_list:
        print("=" * 50)
        print(f"{'AVERAGE':<12} {np.mean(psnr_list):<12.3f} {np.mean(ssim_list):<12.4f} {np.mean(loe_list):<12.2f}")
        print("=" * 50)


if __name__ == "__main__":
    main()
