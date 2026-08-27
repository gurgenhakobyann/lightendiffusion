import os
import glob
import re
import yaml
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration
from metrics.loe import compute_loe

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


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="unsupervised.yml", type=str)
    parser.add_argument("--ckpt_dir", default="/mnt/weka/ghakobyan/ckpt_main/stage2", type=str)
    parser.add_argument("--gt_dir", default="LOLdataset/our485/high", type=str)
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    # Find all checkpoint files
    ckpt_files = sorted(glob.glob(os.path.join(args.ckpt_dir, "model_step_*.pth.tar")),
                       key=lambda x: int(re.search(r"model_step_(\d+)", x).group(1)) if re.search(r"model_step_(\d+)", x) else 0)

    if not ckpt_files:
        # Fallback to model_latest
        ckpt_files = glob.glob(os.path.join(args.ckpt_dir, "model_latest.pth.tar"))

    print(f"Found {len(ckpt_files)} checkpoints to evaluate in {args.ckpt_dir}")

    DATASET = datasets.__dict__[config.data.type](config)
    _, val_loader = DATASET.get_loaders()

    summary_results = []

    for ckpt_path in ckpt_files:
        step_match = re.search(r"model_step_(\d+)", ckpt_path)
        step_num = step_match.group(1) if step_match else "latest"

        print(f"\n---> Evaluating Checkpoint Step {step_num}: {os.path.basename(ckpt_path)}")

        eval_args = argparse.Namespace(
            config=args.config,
            mode="evaluation",
            resume=ckpt_path,
            image_folder=f"results/step_{step_num}"
        )

        diffusion = DenoisingDiffusion(eval_args, config)
        model = DiffusiveRestoration(diffusion, eval_args, config)

        # Restore images
        os.makedirs(f"results/step_{step_num}/LOLv1", exist_ok=True)
        model.restore(val_loader)

        # Compute metrics
        res_dir = f"results/step_{step_num}/LOLv1"
        res_files = sorted([f for f in os.listdir(res_dir) if f.endswith((".png", ".jpg"))])

        psnr_list, ssim_list, loe_list = [], [], []
        for fname in res_files:
            r_path = os.path.join(res_dir, fname)
            g_path = os.path.join(args.gt_dir, fname)
            if not os.path.isfile(g_path):
                g_path = os.path.join("LOLdataset/eval15/high", fname)
                if not os.path.isfile(g_path):
                    continue

            r_img = np.array(Image.open(r_path).convert("RGB"), dtype=np.float32) / 255.0
            g_img = np.array(Image.open(g_path).convert("RGB"), dtype=np.float32) / 255.0

            h = min(r_img.shape[0], g_img.shape[0])
            w = min(r_img.shape[1], g_img.shape[1])
            r_img, g_img = r_img[:h, :w], g_img[:h, :w]

            psnr_list.append(calculate_psnr(r_img, g_img))
            ssim_list.append(calculate_ssim(r_img, g_img))
            loe_list.append(compute_loe(g_img, r_img))

        avg_p = float(np.mean(psnr_list))
        avg_s = float(np.mean(ssim_list))
        avg_l = float(np.mean(loe_list))

        summary_results.append({
            "step": step_num,
            "ckpt": os.path.basename(ckpt_path),
            "psnr": avg_p,
            "ssim": avg_s,
            "loe": avg_l
        })

        print(f"Step {step_num} -> PSNR: {avg_p:.3f} dB | SSIM: {avg_s:.4f} | LOE: {avg_l:.2f}")

    print("\n" + "=" * 65)
    print(f"{'RANKING OF ALL CHECKPOINTS (BEST TO WORST BY SSIM)':^65}")
    print("=" * 65)
    print(f"{'Step':<12} {'PSNR (dB)':<14} {'SSIM':<14} {'LOE':<14} {'Checkpoint File'}")
    print("-" * 65)

    summary_results.sort(key=lambda x: x["ssim"], reverse=True)
    for r in summary_results:
        print(f"{r['step']:<12} {r['psnr']:<14.3f} {r['ssim']:<14.4f} {r['loe']:<14.2f} {r['ckpt']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
