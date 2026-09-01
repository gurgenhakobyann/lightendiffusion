import os
import glob
import re
import yaml
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration
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
    if mse == 0: return float("inf")
    return float(20 * np.log10(1.0 / np.sqrt(mse)))


def calculate_ssim(img1, img2):
    if _HAS_SKIMAGE:
        return float(ssim_fn(img1, img2, data_range=1.0, channel_axis=2))
    return 0.0


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def extract_step(filename):
    match = re.search(r'step_(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="unsupervised.yml", type=str)
    parser.add_argument("--ckpt_dir", default="ckpt/stage2", type=str)
    parser.add_argument("--mode", default="evaluation", type=str)
    parser.add_argument("--image_folder", default="results_leaderboard", type=str)
    parser.add_argument("--gamma", default=False, action="store_true")
    args = parser.parse_args()

    with open(f"configs/{args.config}", "r") as f:
        config_dict = yaml.safe_load(f)
    config = dict2namespace(config_dict)
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find all checkpoint files
    pattern = os.path.join(args.ckpt_dir, "model_*.pth.tar")
    ckpt_files = sorted(glob.glob(pattern), key=lambda x: extract_step(x))

    if not ckpt_files:
        print(f"No checkpoint files found matching {pattern}")
        return

    print(f"Found {len(ckpt_files)} checkpoints to evaluate:")
    for c in ckpt_files:
        print(f"  - {c}")

    config.data.val_dataset = "LOLv1"
    DATASET = datasets.__dict__[config.data.type](config)
    _, val_loader = DATASET.get_loaders()

    lol_gt = "LOLdataset/our485/high"
    if not os.path.isdir(lol_gt):
        lol_gt = "/mnt/weka/ghakobyan/LOLdataset/our485/high"

    results = []

    for ckpt_path in ckpt_files:
        step_num = extract_step(ckpt_path)
        step_label = f"Step {step_num}" if step_num >= 0 else os.path.basename(ckpt_path)
        print(f"\nEvaluating {ckpt_path} ({step_label})...")

        args.resume = ckpt_path
        diffusion = DenoisingDiffusion(args, config)
        restorer = DiffusiveRestoration(diffusion, args, config)

        save_dir = os.path.join(args.image_folder, f"step_{step_num}" if step_num >= 0 else "latest", "LOLv1")
        os.makedirs(save_dir, exist_ok=True)
        restorer.restore(val_loader)

        # Compute metrics
        res_files = sorted([f for f in os.listdir(save_dir) if f.lower().endswith(('.png', '.jpg'))])
        psnrs, ssims, loes = [], [], []
        for f in res_files:
            r_p = os.path.join(save_dir, f)
            g_p = os.path.join(lol_gt, f)
            if not os.path.isfile(g_p): continue
            r_img = np.array(Image.open(r_p).convert("RGB"), dtype=np.float32) / 255.0
            g_img = np.array(Image.open(g_p).convert("RGB"), dtype=np.float32) / 255.0
            h, w = min(r_img.shape[0], g_img.shape[0]), min(r_img.shape[1], g_img.shape[1])
            r_img, g_img = r_img[:h, :w], g_img[:h, :w]
            psnrs.append(calculate_psnr(r_img, g_img))
            ssims.append(calculate_ssim(r_img, g_img))
            loes.append(compute_loe(g_img, r_img))

        if psnrs:
            mean_p = float(np.mean(psnrs))
            mean_s = float(np.mean(ssims))
            mean_l = float(np.mean(loes))
            results.append({
                "step": step_num,
                "label": step_label,
                "file": os.path.basename(ckpt_path),
                "psnr": mean_p,
                "ssim": mean_s,
                "loe": mean_l
            })
            print(f"--> {step_label}: PSNR = {mean_p:.3f} dB | SSIM = {mean_s:.4f} | LOE = {mean_l:.2f}")

    # Sort results by SSIM descending
    results.sort(key=lambda x: x["ssim"], reverse=True)

    print("\n" + "=" * 70)
    print(f"{'RANKING OF ALL CHECKPOINTS (BEST TO WORST BY SSIM)':^70}")
    print("=" * 70)
    print(f"{'Step':<12} {'PSNR (dB)':<14} {'SSIM':<14} {'LOE':<14} {'Checkpoint File'}")
    print("-" * 70)
    for r in results:
        step_str = str(r["step"]) if r["step"] >= 0 else "latest"
        print(f"{step_str:<12} {r['psnr']:<14.3f} {r['ssim']:<14.4f} {r['loe']:<14.2f} {r['file']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
