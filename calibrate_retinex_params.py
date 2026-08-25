"""
Calibrate (tau, lambda, vartheta) for the generalized Retinex composition
f(R, L) = tau * R^lambda * L^vartheta by minimizing Lightness Order Error (LOE)
on the LOL training set, following the calibration protocol in Trongtirakul,
Agaian & Wu (IEEE Access, 2023), Eq. 14-15 / Fig. 3.

Uses a FROZEN, already-trained stage-1 checkpoint (CTDN decomposition).
No diffusion-model training happens here — this only searches for the best
fixed exponents to later plug into stage-2 training.

Usage:
    python calibrate_retinex_params.py --config configs/unsupervised.yml \
        --checkpoint stage1_weight.pth.tar \
        --data_root LOLdataset \
        --n_calib_images 100 \
        --target scc
"""

import argparse
import itertools
import json
import os
import sys
import numpy as np
from PIL import Image
import torch
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable


sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from metrics.loe import compute_loe
from models.decom import CTDN
from models.generalized_retinex import generalized_retinex_compose


def load_calibration_pairs(data_root, n_images=100):
    """
    Load up to n_images (I_low, I_ref) pairs from LOL training split (our485).
    Returns list of tuples: (I_low_np, I_ref_np) with float32 in [0, 1].
    """
    low_dir = os.path.join(data_root, "our485", "low")
    high_dir = os.path.join(data_root, "our485", "high")

    if not os.path.isdir(low_dir) or not os.path.isdir(high_dir):
        # Check if data_root itself has low / high
        if os.path.isdir(os.path.join(data_root, "low")):
            low_dir = os.path.join(data_root, "low")
            high_dir = os.path.join(data_root, "high")
        else:
            raise FileNotFoundError(f"Cannot find low/high directories in {data_root}")

    low_filenames = sorted([f for f in os.listdir(low_dir) if f.endswith((".png", ".jpg", ".bmp"))])
    
    pairs = []
    for fname in low_filenames:
        high_path = os.path.join(high_dir, fname)
        low_path = os.path.join(low_dir, fname)
        if os.path.isfile(high_path):
            low_img = np.array(Image.open(low_path).convert("RGB"), dtype=np.float32) / 255.0
            high_img = np.array(Image.open(high_path).convert("RGB"), dtype=np.float32) / 255.0
            pairs.append((low_img, high_img))
        if len(pairs) >= n_images:
            break

    print(f"Loaded {len(pairs)} calibration image pairs from {low_dir}")
    return pairs


def precompute_decompositions(ctdn, calib_pairs, target, device):
    """
    Precompute R and L tensors for all calibration images to accelerate grid search.
    """
    cached_data = []
    ctdn.eval()
    with torch.no_grad():
        for low_np, ref_np in calib_pairs:
            # (H, W, 3) -> (1, 3, H, W)
            low_t = torch.from_numpy(low_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            ref_t = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

            # Pad to multiple of 64 if necessary
            h, w = low_t.shape[2], low_t.shape[3]
            img_h_64 = int(64 * np.ceil(h / 64.0))
            img_w_64 = int(64 * np.ceil(w / 64.0))
            pad_h = img_h_64 - h
            pad_w = img_w_64 - w

            if pad_h > 0 or pad_w > 0:
                low_t_padded = torch.nn.functional.pad(low_t, (0, pad_w, 0, pad_h), mode="reflect")
                ref_t_padded = torch.nn.functional.pad(ref_t, (0, pad_w, 0, pad_h), mode="reflect")
            else:
                low_t_padded = low_t
                ref_t_padded = ref_t

            # Forward CTDN on 6-channel concatenation [low, ref]
            input_6ch = torch.cat([low_t_padded, ref_t_padded], dim=1)
            output = ctdn(input_6ch, pred_fea=None)

            if target == "scc":
                # SCC pseudo-label uses R_low and L_low
                R = output["low_R"]
                L = output["low_L"]
            elif target == "x0":
                # x0 target uses R_low and L_high
                R = output["low_R"]
                L = output["high_L"]
            else:
                raise ValueError(f"Unknown target: {target}")

            cached_data.append({
                "low_np": low_np,
                "ref_np": ref_np,
                "low_t_padded": low_t_padded,
                "R": R,
                "L": L,
                "orig_h": h,
                "orig_w": w
            })

    return cached_data


def decode_and_evaluate_candidate(ctdn, item, tau, lam, vartheta, eps=1e-4):
    """
    Recompose with candidate parameters, decode back to RGB image, and calculate LOE.
    """
    with torch.no_grad():
        composed = generalized_retinex_compose(item["R"], item["L"], tau=tau, lam=lam, vartheta=vartheta, eps=eps)
        pred_img_padded = ctdn(item["low_t_padded"][:, :3, ...], pred_fea=composed)["pred_img"]
        
        # Crop back to original dimensions
        h, w = item["orig_h"], item["orig_w"]
        pred_img = pred_img_padded[:, :, :h, :w]
        cand_np = pred_img.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()

    loe_input = compute_loe(item["low_np"], cand_np)
    loe_ref = compute_loe(item["ref_np"], cand_np)
    return loe_input, loe_ref


def run_grid_search(ctdn, cached_data, tau_grid, lambda_grid, vartheta_grid, stage1_weight=0.5, eps=1e-4):
    combos = list(itertools.product(tau_grid, lambda_grid, vartheta_grid))
    results = []

    print(f"Starting grid search over {len(combos)} parameter combinations on {len(cached_data)} images...")
    for tau, lam, vartheta in tqdm(combos, desc="LOE Grid Search"):
        loe_in_list = []
        loe_ref_list = []
        for item in cached_data:
            loe_in, loe_ref = decode_and_evaluate_candidate(ctdn, item, tau, lam, vartheta, eps=eps)
            loe_in_list.append(loe_in)
            loe_ref_list.append(loe_ref)

        avg_in = float(np.mean(loe_in_list))
        avg_ref = float(np.mean(loe_ref_list))
        combined = stage1_weight * avg_in + (1.0 - stage1_weight) * avg_ref

        results.append({
            "tau": float(tau),
            "lambda": float(lam),
            "vartheta": float(vartheta),
            "loe_input": avg_in,
            "loe_ref": avg_ref,
            "combined": combined,
        })

    results.sort(key=lambda r: r["combined"])
    return results


def run_comparison(ctdn, cached_data, best_result, eps=1e-4):
    """
    Evaluate a fixed set of baseline + optimal configs and print a comparison table.
    Baselines:
      - Classical Retinex  : tau=1, lambda=1, vartheta=1  (original LightenDiffusion)
      - Gamma mild         : tau=1, lambda=1, vartheta=0.7 (common photo app gamma)
      - Gamma strong       : tau=1, lambda=1, vartheta=0.5 (strong gamma correction)
      - Random A           : tau=0.75, lambda=0.6, vartheta=0.3
      - Random B           : tau=1.5,  lambda=1.4, vartheta=0.75
      - Grid-Search Optimal: best values from grid search
    """
    configs = [
        {"name": "Classical Retinex (I=R·L)",   "tau": 1.00, "lam": 1.00, "vartheta": 1.00},
        {"name": "Gamma mild (ϑ=0.7)",           "tau": 1.00, "lam": 1.00, "vartheta": 0.70},
        {"name": "Gamma strong (ϑ=0.5)",         "tau": 1.00, "lam": 1.00, "vartheta": 0.50},
        {"name": "Random A (τ=0.75,λ=0.6,ϑ=0.3)","tau": 0.75, "lam": 0.60, "vartheta": 0.30},
        {"name": "Random B (τ=1.5,λ=1.4,ϑ=0.75)","tau": 1.50, "lam": 1.40, "vartheta": 0.75},
        {"name": f"Grid-Search Optimal (best)",
         "tau": best_result["tau"], "lam": best_result["lambda"], "vartheta": best_result["vartheta"]},
    ]

    print("\n" + "=" * 90)
    print("COMPARISON TABLE: Baseline vs Grid-Search Optimal")
    print(f"{'Config':<38} {'tau':<6} {'lambda':<8} {'vartheta':<10} {'LOE(in)':<10} {'LOE(ref)':<10} {'Combined':<10}")
    print("-" * 90)

    for cfg in configs:
        loe_in_list, loe_ref_list = [], []
        for item in cached_data:
            loe_in, loe_ref = decode_and_evaluate_candidate(
                ctdn, item, cfg["tau"], cfg["lam"], cfg["vartheta"], eps=eps
            )
            loe_in_list.append(loe_in)
            loe_ref_list.append(loe_ref)
        avg_in  = float(np.mean(loe_in_list))
        avg_ref = float(np.mean(loe_ref_list))
        combined = 0.5 * avg_in + 0.5 * avg_ref
        print(f"{cfg['name']:<38} {cfg['tau']:<6.2f} {cfg['lam']:<8.2f} {cfg['vartheta']:<10.2f} "
              f"{avg_in:<10.2f} {avg_ref:<10.2f} {combined:<10.2f}")

    print("=" * 90)
    print("Lower LOE = better lightness order preservation.")


def main():
    parser = argparse.ArgumentParser(description="Calibrate Generalized Retinex parameters using LOE search.")
    parser.add_argument("--config", default="configs/unsupervised.yml", type=str, help="Path to config file")
    parser.add_argument("--checkpoint", default="stage1_weight.pth.tar", type=str, help="Path to stage-1 checkpoint")
    parser.add_argument("--data_root", default="LOLdataset", type=str, help="Path to LOL dataset root")
    parser.add_argument("--n_calib_images", type=int, default=100, help="Number of calibration images to use")
    parser.add_argument("--target", choices=["scc", "x0"], default="scc", help="Target composition to calibrate")
    parser.add_argument("--stage1_weight", type=float, default=0.5, help="Weight for input LOE vs reference LOE")
    parser.add_argument("--save_json", default="", type=str, help="Path to save ranked results as JSON")
    parser.add_argument("--compare", action="store_true",
                        help="After grid search, also print a comparison table with classical/random/optimal configs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load frozen CTDN from Stage-1 Checkpoint
    print(f"Loading Stage-1 checkpoint from {args.checkpoint}...")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    ctdn = CTDN(channels=64).to(device)
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    ctdn.load_state_dict(state_dict, strict=True)
    ctdn.eval()
    for p in ctdn.parameters():
        p.requires_grad = False

    # Load calibration pairs
    calib_pairs = load_calibration_pairs(args.data_root, args.n_calib_images)
    if len(calib_pairs) == 0:
        raise RuntimeError("No calibration pairs found!")

    # Precompute latent decompositions
    print(f"Precomputing CTDN latent decompositions for target='{args.target}'...")
    cached_data = precompute_decompositions(ctdn, calib_pairs, args.target, device)

    # Search grids
    # Grid centered around baseline values
    if args.target == "scc":
        tau_grid = [0.5, 0.75, 1.0, 1.25, 1.5]
        lambda_grid = [0.6, 0.8, 1.0, 1.2, 1.4]
        vartheta_grid = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    else:  # x0
        tau_grid = [0.5, 0.75, 1.0, 1.25, 1.5]
        lambda_grid = [0.6, 0.8, 1.0, 1.2, 1.4]
        vartheta_grid = [0.6, 0.8, 1.0, 1.2, 1.4]

    results = run_grid_search(ctdn, cached_data, tau_grid, lambda_grid, vartheta_grid, stage1_weight=args.stage1_weight)

    print("\n" + "=" * 80)
    print(f"Top 10 Candidates for target='{args.target}':")
    print(f"{'Rank':<6} {'tau':<8} {'lambda':<8} {'vartheta':<10} {'LOE(input)':<14} {'LOE(ref)':<14} {'Combined':<12}")
    print("-" * 80)
    for i, r in enumerate(results[:10]):
        print(f"{i+1:<6} {r['tau']:<8.3f} {r['lambda']:<8.3f} {r['vartheta']:<10.3f} {r['loe_input']:<14.2f} {r['loe_ref']:<14.2f} {r['combined']:<12.2f}")
    print("=" * 80)

    best = results[0]
    print(f"\nBest Optimum: tau={best['tau']}, lambda={best['lambda']}, vartheta={best['vartheta']}")
    print(f"LOE(input)={best['loe_input']:.2f}, LOE(ref)={best['loe_ref']:.2f}, Combined={best['combined']:.2f}")

    if args.compare:
        run_comparison(ctdn, cached_data, best)

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved complete results to {args.save_json}")


if __name__ == "__main__":
    main()
