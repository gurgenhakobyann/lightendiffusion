import os
import glob
import argparse
import numpy as np
from PIL import Image

from metrics.color_correction import msrcr, msrcp, guided_chroma_correction
from metrics.loe import compute_loe

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return float("inf")
    return float(20 * np.log10(1.0 / np.sqrt(mse)))


def calculate_ssim(img1, img2):
    if _HAS_SKIMAGE:
        return float(ssim_fn(img1, img2, data_range=1.0, channel_axis=2))
    return 0.0


def eval_folder(pred_dir, gt_dir, eval_names, low_dir=None, method="none", out_save_dir=None):
    if out_save_dir:
        os.makedirs(out_save_dir, exist_ok=True)

    psnrs, ssims, loes = [], [], []

    for fname in eval_names:
        p_path = os.path.join(pred_dir, fname)
        g_path = os.path.join(gt_dir, fname)
        if not os.path.isfile(p_path) or not os.path.isfile(g_path):
            continue

        p_img = np.array(Image.open(p_path).convert("RGB"), dtype=np.float32) / 255.0
        g_img = np.array(Image.open(g_path).convert("RGB"), dtype=np.float32) / 255.0

        # Apply chosen color correction algorithm
        if method == "none":
            corr_img = p_img
        elif method == "msrcr":
            corr_img = msrcr(p_img)
        elif method == "msrcp":
            corr_img = msrcp(p_img)
        elif method == "guided":
            l_path = os.path.join(low_dir, fname) if low_dir else None
            if l_path and os.path.isfile(l_path):
                l_img = np.array(Image.open(l_path).convert("RGB"), dtype=np.float32) / 255.0
                corr_img = guided_chroma_correction(p_img, l_img, alpha=0.3)
            else:
                corr_img = msrcp(p_img)
        else:
            corr_img = p_img

        corr_img = np.clip(corr_img, 0.0, 1.0)

        # Save corrected image if requested
        if out_save_dir:
            out_p = os.path.join(out_save_dir, fname)
            Image.fromarray((corr_img * 255.0).astype(np.uint8)).save(out_p)

        h, w = min(corr_img.shape[0], g_img.shape[0]), min(corr_img.shape[1], g_img.shape[1])
        c_crop, g_crop = corr_img[:h, :w], g_img[:h, :w]

        psnrs.append(calculate_psnr(c_crop, g_crop))
        ssims.append(calculate_ssim(c_crop, g_crop))
        loes.append(compute_loe(g_crop, c_crop))

    if not psnrs:
        return None
    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "loe": float(np.mean(loes)),
        "count": len(psnrs)
    }


def main():
    parser = argparse.ArgumentParser(description="Test Retinex Color Correction Algorithms on Generated Images")
    parser.add_argument("--pred_dir", default="results_best/LOLv1", type=str, help="Directory with generated images")
    parser.add_argument("--gt_dir", default="/mnt/weka/ghakobyan/LOLdataset/our485/high", type=str)
    parser.add_argument("--low_dir", default="/mnt/weka/ghakobyan/LOLdataset/our485/low", type=str)
    args = parser.parse_args()

    # Fallback to alternative paths if needed
    if not os.path.isdir(args.pred_dir):
        for cand in ["results_full/LOLv1", "results_leaderboard/step_320000/LOLv1", "results/LOLv1"]:
            if os.path.isdir(cand):
                args.pred_dir = cand
                break

    if not os.path.isdir(args.gt_dir):
        for cand in ["LOLdataset/our485/high", "dataset/LOLv1/our485/high"]:
            if os.path.isdir(cand):
                args.gt_dir = cand
                break

    print(f"Evaluating generated images from: {args.pred_dir}")
    print(f"Ground Truth directory:           {args.gt_dir}")

    eval_names = ['1.png', '22.png', '23.png', '55.png', '79.png', '111.png', '146.png', '179.png', '493.png', '547.png', '665.png', '669.png', '748.png', '778.png', '780.png']

    methods = [
        ("Raw Diffusion (No Correction)", "none"),
        ("MSRCR (Color Restoration - Jobson 1997)", "msrcr"),
        ("MSRCP (Chromaticity Preservation - Petro 2014)", "msrcp"),
        ("Guided Chrominance Alignment (Input Guided)", "guided"),
    ]

    print("\n" + "=" * 80)
    print(f"{'RETINEX COLOR CORRECTION COMPARISON ON GENERATED IMAGES':^80}")
    print("=" * 80)
    print(f"{'Algorithm / Method':<45} | {'PSNR (dB)↑':>10} | {'SSIM↑':>8} | {'LOE↓':>8}")
    print("-" * 80)

    for label, method in methods:
        res = eval_folder(
            args.pred_dir,
            args.gt_dir,
            eval_names,
            low_dir=args.low_dir,
            method=method,
            out_save_dir=f"results_corrected_{method}/LOLv1"
        )
        if res:
            print(f"{label:<45} | {res['psnr']:>10.3f} | {res['ssim']:>8.4f} | {res['loe']:>8.2f}")
        else:
            print(f"{label:<45} | {'N/A':>10} | {'N/A':>8} | {'N/A':>8}")

    print("=" * 80)


if __name__ == "__main__":
    main()
