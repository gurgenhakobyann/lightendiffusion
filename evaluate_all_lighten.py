import os
import sys
import yaml
import argparse
import numpy as np
from PIL import Image
import torch

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


def eval_paired(res_dir, gt_dir):
    if not os.path.isdir(res_dir) or not os.path.isdir(gt_dir): return None
    res_files = sorted([f for f in os.listdir(res_dir) if f.lower().endswith(('.png', '.jpg'))])
    psnrs, ssims, lpips_l, loes = [], [], [], []
    for f in res_files:
        r_p = os.path.join(res_dir, f)
        g_p = os.path.join(gt_dir, f)
        if not os.path.isfile(g_p): continue
        r_img = np.array(Image.open(r_p).convert("RGB"), dtype=np.float32) / 255.0
        g_img = np.array(Image.open(g_p).convert("RGB"), dtype=np.float32) / 255.0
        h, w = min(r_img.shape[0], g_img.shape[0]), min(r_img.shape[1], g_img.shape[1])
        r_img, g_img = r_img[:h, :w], g_img[:h, :w]
        psnrs.append(calculate_psnr(r_img, g_img))
        ssims.append(calculate_ssim(r_img, g_img))
        lpips_l.append(compute_lpips(r_img, g_img))
        loes.append(compute_loe(g_img, r_img))
    if not psnrs: return None
    return {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)), "lpips": float(np.mean(lpips_l)), "loe": float(np.mean(loes))}


def eval_unpaired(res_dir):
    if not os.path.isdir(res_dir): return None
    res_files = sorted([f for f in os.listdir(res_dir) if f.lower().endswith(('.png', '.jpg', '.JPG'))])
    niqes, pis = [], []
    for f in res_files:
        r_p = os.path.join(res_dir, f)
        r_img = np.array(Image.open(r_p).convert("RGB"), dtype=np.float32) / 255.0
        n = compute_niqe(r_img)
        p = compute_pi(r_img, niqe_val=n)
        niqes.append(n)
        pis.append(p)
    if not niqes: return None
    return {"niqe": float(np.mean(niqes)), "pi": float(np.mean(pis))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="unsupervised.yml", type=str)
    parser.add_argument("--resume", default="ckpt/stage2/model_latest.pth.tar", type=str)
    parser.add_argument("--mode", default="evaluation", type=str)
    parser.add_argument("--image_folder", default="results_full", type=str)
    parser.add_argument("--gamma", default=False, action="store_true")
    args = parser.parse_args()

    with open(f"configs/{args.config}", "r") as f:
        config_dict = yaml.safe_load(f)
    config = dict2namespace(config_dict)
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint {args.resume}...")
    diffusion = DenoisingDiffusion(args, config)
    restorer = DiffusiveRestoration(diffusion, args, config)

    # 1. Run inference on all 5 datasets
    benchmarks = ["LOLv1", "LSRW", "DICM", "NPE", "VV"]
    for bench in benchmarks:
        print(f"\n---> Generating restorations for {bench}...")
        config.data.val_dataset = bench
        DATASET = datasets.__dict__[config.data.type](config)
        _, val_loader = DATASET.get_loaders()
        if val_loader:
            os.makedirs(os.path.join(args.image_folder, bench), exist_ok=True)
            restorer.restore(val_loader)

    # 2. Compute Metrics across all 5
    lol_gt = "LOLdataset/our485/high"
    if not os.path.isdir(lol_gt): lol_gt = "/mnt/weka/ghakobyan/LOLdataset/our485/high"
    lol_res = eval_paired(os.path.join(args.image_folder, "LOLv1"), lol_gt)

    lsrw_gt = "/mnt/weka/ghakobyan/LSRW/Eval/Eval/Huawei/high"
    lsrw_res = eval_paired(os.path.join(args.image_folder, "LSRW"), lsrw_gt)

    dicm_res = eval_unpaired(os.path.join(args.image_folder, "DICM"))
    npe_res = eval_unpaired(os.path.join(args.image_folder, "NPE"))
    vv_res = eval_unpaired(os.path.join(args.image_folder, "VV"))

    print("\n" + "=" * 90)
    print(f"{'FULL BENCHMARK TABLE FOR LightenDiffusion (400k+ STEPS)':^90}")
    print("=" * 90)
    header1 = f"{'Method':<20} | {'LOL':^24} | {'LSRW':^24} | {'DICM':^14} | {'NPE':^14} | {'VV':^14}"
    header2 = f"{'':<20} | {'PSNR↑':>7} {'SSIM↑':>7} {'LPIPS↓':>8} | {'PSNR↑':>7} {'SSIM↑':>7} {'LPIPS↓':>8} | {'NIQE↓':>6} {'PI↓':>6} | {'NIQE↓':>6} {'PI↓':>6} | {'NIQE↓':>6} {'PI↓':>6}"
    print(header1)
    print(header2)
    print("-" * 90)

    lol_s = f"{lol_res['psnr']:>7.3f} {lol_res['ssim']:>7.4f} {lol_res['lpips']:>8.3f}" if lol_res else f"{'--':>7} {'--':>7} {'--':>8}"
    lsrw_s = f"{lsrw_res['psnr']:>7.3f} {lsrw_res['ssim']:>7.4f} {lsrw_res['lpips']:>8.3f}" if lsrw_res else f"{'--':>7} {'--':>7} {'--':>8}"
    dicm_s = f"{dicm_res['niqe']:>6.3f} {dicm_res['pi']:>6.3f}" if dicm_res else f"{'--':>6} {'--':>6}"
    npe_s = f"{npe_res['niqe']:>6.3f} {npe_res['pi']:>6.3f}" if npe_res else f"{'--':>6} {'--':>6}"
    vv_s = f"{vv_res['niqe']:>6.3f} {vv_res['pi']:>6.3f}" if vv_res else f"{'--':>6} {'--':>6}"

    print(f"{'LightenDiffusion':<20} | {lol_s} | {lsrw_s} | {dicm_s} | {npe_s} | {vv_s}")
    print("=" * 90)
    if lol_res: print(f"[LOL Details] PSNR: {lol_res['psnr']:.3f} dB | SSIM: {lol_res['ssim']:.4f} | LOE: {lol_res['loe']:.2f}")


if __name__ == "__main__":
    main()
