# Task: Add Agaian's Generalized Retinex Composition to LightenDiffusion, Calibrated via LOE Search

## Repository
`https://github.com/JianghaiSCU/LightenDiffusion` (ECCV 2024, "LightenDiffusion: Unsupervised
Low-Light Image Enhancement with Latent-Retinex Diffusion Models", Jiang et al.)

Assume this repo is already cloned and its dependencies installed. Do not assume exact file
names below are correct — locate the real files by searching for the identifiers given in each
step, then make the edits there.

---

## 1. Background and Motivation

### 1.1 What LightenDiffusion currently does

LightenDiffusion assumes the **classical multiplicative Retinex model**:

```
I = R ⊙ L        (Hadamard/elementwise product)
```

This assumption is used in (at least) four places in the codebase:

1. **CTDN initial estimate** — the illumination/reflectance initialization from the encoded
   feature `F`, following `L̃ = max_c F^c`, `R̃ = F / (L̃ + τ)`.
2. **Decomposition reconstruction loss** `L_rec` — forces `F_low^j ≈ R_low^i ⊙ L_low^j` for the
   two paired low-light crops used in stage-1 training (paper Eq. 8).
3. **Diffusion forward-process target** — `x0 = R_low ⊙ L_high`, i.e. the low-light content
   combined with a normal-light illumination map (paper Sec 3.3, "Forward Diffusion").
4. **Self-constrained consistency pseudo-label** — `F̃_low = R_low ⊙ L_low^γ`, where `γ` (default
   0.2) is a fixed gamma-correction exponent applied *only* to `L`, not to `R` (paper Eq. 6 area).
   This is already a partial deviation from pure `R⊙L` — note `R` still has an implicit exponent
   of 1.

### 1.2 What we're substituting in

From Trongtirakul, Agaian & Wu, *"Adaptive Single Low-Light Image Enhancement by Fractional
Stretching in Logarithmic Domain,"* IEEE Access, 2023 — the generalized Retinex composition:

```
f(R, L) = τ · R^λ · L^ϑ
```

which reduces to the classical model when `τ = λ = ϑ = 1`. In the log domain (their Eq. 9–13):

```
log(f + ε) = log(τ + ε) + λ·log(R + ε) + ϑ·log(L + ε)
```

**Goal of this task:** replace the four hard-coded `R ⊙ L` recompositions above with this
generalized form, where `(τ, λ, ϑ)` are **fixed constants found by minimizing Lightness Order
Error (LOE)** on the LOL training set — mirroring exactly how the Agaian paper calibrates its
own `α, β` constants (their Eq. 14–15, Fig. 3), rather than making them learnable at this stage.
This is the cheapest, most controlled variant to implement and evaluate first; treat it as
Ablation Row in the style of LightenDiffusion's own Table 2.

---

## 2. Step 0 — Locate the relevant code

Before changing anything, search the repo for these identifiers and note the file + line
numbers. Do not proceed until you've found all of them (or determined a given one doesn't exist
under that name and found its equivalent):

```bash
grep -rn "class .*CTDN\|ContentTransfer\|content.transfer" --include="*.py" .
grep -rn "Rlow\|R_low\|reflect" --include="*.py" .
grep -rn "Lhigh\|L_high\|illum" --include="*.py" .
grep -rn "x0\s*=" --include="*.py" .
grep -rn "gamma\|\*\* *0.2\|pow(.*0.2" --include="*.py" .
grep -rn "def train\b" --include="*.py" .
grep -rn "L_rec\|rec_loss\|reconstruction" --include="*.py" .
grep -rn "scc\|consistency" --include="*.py" .
grep -rn "unsupervised.yml\|configs/" --include="*.py" .
```

Report back (in comments or a short summary) which file contains: (a) the CTDN
forward pass, (b) the diffusion training loop building `x0`, (c) the `L_scc` pseudo-label
construction, (d) the config loading logic. All subsequent steps reference these by role, not
by guessed filename.

---

## 3. Step 1 — Implement the generalized composition primitive

Add a new small module, e.g. `models/generalized_retinex.py`, with:

```python
import math
import torch

def generalized_retinex_compose(R: torch.Tensor, L: torch.Tensor,
                                 tau: float = 1.0, lam: float = 1.0, vartheta: float = 1.0,
                                 eps: float = 1e-4) -> torch.Tensor:
    """
    Generalized Retinex composition f(R, L) = tau * R^lambda * L^vartheta,
    from Trongtirakul, Agaian & Wu (IEEE Access, 2023), "Adaptive Single
    Low-Light Image Enhancement by Fractional Stretching in Logarithmic Domain."

    Reduces exactly to the classical Retinex model I = R ⊙ L used in
    LightenDiffusion (Jiang et al., ECCV 2024) when tau = lambda = vartheta = 1.

    Computed in log domain for numerical stability:
        log(f + eps) = log(tau + eps) + lambda * log(R + eps) + vartheta * log(L + eps)

    IMPORTANT: R and L must be non-negative going in (see Step 2 note on CTDN
    output range). This function clamps defensively but the network should
    already guarantee non-negativity.
    """
    R_ = torch.clamp(R, min=0.0)
    L_ = torch.clamp(L, min=0.0)
    log_f = (math.log(tau + eps)
              + lam * torch.log(R_ + eps)
              + vartheta * torch.log(L_ + eps))
    return torch.exp(log_f)


def classical_retinex_compose(R: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Baseline I = R ⊙ L, kept for the tau=lambda=vartheta=1 ablation row."""
    return R * L
```

Add a unit test (e.g. `tests/test_generalized_retinex.py`) asserting that
`generalized_retinex_compose(R, L, 1.0, 1.0, 1.0)` matches `classical_retinex_compose(R, L)`
within floating-point tolerance (`eps` will introduce a tiny, bounded discrepancy — assert
`atol=1e-3` or derive the exact expected offset from `eps`).

### Note on non-negativity of R, L in latent space

Unlike the Agaian paper (which operates on bounded `L*` pixel values in Lab space), CTDN's `R`
and `L` are outputs of conv blocks operating on **encoder latent features**, which are not
guaranteed non-negative or bounded. Before wiring in `generalized_retinex_compose`, check the
final layers of CTDN (the `Convs(R'' + L'')` and `Convs(L' - L'')` outputs from paper Fig. 4). If
they don't already end in a `Softplus`/`ReLU`/`Sigmoid`, flag this to the user rather than
silently clamping — clamping at 0 will zero out negative activations and could measurably change
behavior versus the paper's trained checkpoint. Prefer testing with `torch.clamp(..., min=0)` +
a log warning first; only add an activation change if numerically necessary, and treat that as a
separate, clearly-labeled change.

---

## 4. Step 2 — Wire a config flag and constants

In the YAML config loaded by `train.py` (e.g. `configs/unsupervised.yml`), add:

```yaml
retinex:
  mode: "generalized"     # "classical" or "generalized" — toggles which compose fn is used
  # Composition used for the diffusion forward-process target x0 = compose(R_low, L_high)
  tau_x0: 1.0
  lambda_x0: 1.0
  vartheta_x0: 1.0
  # Composition used for the self-constrained consistency pseudo-label
  # F_tilde_low = compose(R_low, L_low); replaces the old fixed gamma=0.2 on L only
  tau_scc: 1.0
  lambda_scc: 1.0
  vartheta_scc: 0.2        # matches existing default so behavior is unchanged until recalibrated
  eps: 1e-4
```

In the code, load these into the training loop / model class (wherever `x0 = R_low * L_high` and
`F_tilde_low = R_low * L_low ** gamma` currently live), and replace with calls to
`generalized_retinex_compose(...)` or `classical_retinex_compose(...)` depending on
`config.retinex.mode`. Keep `classical` as a selectable mode so you can always reproduce the
paper's original numbers as a control.

Do **not** touch `L_rec` (Eq. 8) or the CTDN-internal estimate yet — get `x0` and `L_scc` working
and evaluated first, since those are the two places generalization is most likely to matter
(they directly shape what the diffusion model learns to generate). Note the other two locations
in a TODO/comment for a possible follow-up round.

---

## 5. Step 3 — Implement LOE (Lightness Order Error)

Add `metrics/loe.py` (or extend the existing metrics module — search for existing
`PSNR`/`SSIM`/`NIQE` implementations first and match their style/interface):

```python
import numpy as np
import cv2

def compute_loe(I_ref: np.ndarray, I_enh: np.ndarray, downsample_size: int = 50) -> float:
    """
    Lightness Order Error (Wang et al., "Naturalness Preserved Enhancement
    Algorithm for Non-Uniform Illumination Images," IEEE TIP 2013), used as
    the calibration metric in the Agaian et al. paper (Eq. 14-15, Fig. 3).

    Lower is better. Both images must be same size, uint8 or float [0,1] RGB.
    Downsamples to `downsample_size` x `downsample_size` for tractable O(N^2)
    pairwise comparison, matching common practice in the LLIE literature.
    """
    def luminance(img):
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        L = np.max(img, axis=2)  # max-channel proxy for lightness, matches CTDN's own L estimate
        L_small = cv2.resize(L, (downsample_size, downsample_size), interpolation=cv2.INTER_AREA)
        return L_small.flatten()

    Lr = luminance(I_ref)
    Le = luminance(I_enh)
    n = len(Lr)

    # Pairwise relative-order comparison, vectorized
    Ur = (Lr[:, None] >= Lr[None, :]).astype(np.uint8)
    Ue = (Le[:, None] >= Le[None, :]).astype(np.uint8)
    RD = np.sum(np.abs(Ur - Ue), axis=1)
    loe = np.mean(RD) / n
    return float(loe)
```

Add a unit test: `compute_loe(img, img)` should be ~0 for an image compared with itself.

---

## 6. Step 4 — Calibration script (the actual "LOE search")

Create `calibrate_retinex_params.py` at the repo root. This mirrors Agaian et al. Eq. 14–15
exactly: first fit against the *input* image to preserve lightness order, then fit against the
*reference* image to match the target brightness order — but applied here to LightenDiffusion's
learned `R_low`, `L_low` rather than to a hand-crafted MSR pyramid.

```python
"""
Calibrate (tau, lambda, vartheta) for the generalized Retinex composition
f(R, L) = tau * R^lambda * L^vartheta by minimizing Lightness Order Error (LOE)
on the LOL training set, following the calibration protocol in Trongtirakul,
Agaian & Wu (IEEE Access, 2023), Eq. 14-15 / Fig. 3.

Uses a FROZEN, already-trained stage-1 checkpoint (encoder + CTDN + decoder).
No diffusion-model training happens here — this only searches for the best
fixed exponents to later plug into stage-2 training.

Usage:
    python calibrate_retinex_params.py --config configs/unsupervised.yml \
        --checkpoint <path to stage-1 checkpoint> \
        --data_root <path to LOL training pairs> \
        --n_calib_images 100 \
        --target scc   # or "x0" — which composition to calibrate
"""
import argparse
import itertools
import numpy as np
import torch
from tqdm import tqdm

# TODO(agent): import the real Encoder, CTDN, Decoder classes from wherever
# Step 0 located them, plus the real dataset loader for LOL pairs, plus
# compute_loe and generalized_retinex_compose from the modules above.
# from models import Encoder, CTDN, Decoder
# from metrics.loe import compute_loe
# from models.generalized_retinex import generalized_retinex_compose


def load_calibration_pairs(data_root, n_images):
    """Return a list of (I_low, I_ref) numpy RGB pairs from the LOL training split."""
    raise NotImplementedError("Wire to the existing LOL dataset loader — see datasets/dataset.py")


def decode_composition(encoder, ctdn, decoder, I_low_tensor, tau, lam, vartheta, device):
    """
    Encode I_low -> F_low -> (R_low, L_low) via frozen CTDN, recompose with the
    candidate (tau, lambda, vartheta), decode back to an image, return as numpy RGB.
    """
    with torch.no_grad():
        F_low = encoder(I_low_tensor.to(device))
        R_low, L_low = ctdn(F_low)   # confirm actual CTDN return signature during Step 0
        composed = generalized_retinex_compose(R_low, L_low, tau, lam, vartheta)
        I_cand = decoder(composed)
    img = I_cand.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return img


def grid_search(encoder, ctdn, decoder, calib_pairs, device,
                 tau_grid, lambda_grid, vartheta_grid, stage1_weight=0.5):
    """
    Two-term objective mirroring Agaian Eq. 14-15:
      stage A: LOE(I_low, I_cand)   -- preserve the low-light image's own lightness order
      stage B: LOE(I_ref,  I_cand)  -- match the reference (ground-truth normal-light) order
    Combined here as a weighted sum so a single grid search suffices; report both
    components separately so you can inspect the tradeoff (see printed table).
    """
    results = []
    combos = list(itertools.product(tau_grid, lambda_grid, vartheta_grid))
    for tau, lam, vartheta in tqdm(combos, desc="grid search"):
        loe_input_list, loe_ref_list = [], []
        for I_low_np, I_ref_np in calib_pairs:
            I_low_t = torch.from_numpy(I_low_np).permute(2, 0, 1).unsqueeze(0).float()
            I_cand = decode_composition(encoder, ctdn, decoder, I_low_t, tau, lam, vartheta, device)
            loe_input_list.append(compute_loe(I_low_np, I_cand))
            loe_ref_list.append(compute_loe(I_ref_np, I_cand))
        loe_input = float(np.mean(loe_input_list))
        loe_ref = float(np.mean(loe_ref_list))
        combined = stage1_weight * loe_input + (1 - stage1_weight) * loe_ref
        results.append({"tau": tau, "lambda": lam, "vartheta": vartheta,
                         "loe_input": loe_input, "loe_ref": loe_ref, "combined": combined})
    results.sort(key=lambda r: r["combined"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--n_calib_images", type=int, default=100)
    parser.add_argument("--target", choices=["x0", "scc"], required=True)
    parser.add_argument("--stage1_weight", type=float, default=0.5)
    args = parser.parse_args()

    # TODO(agent): load config, build + load frozen encoder/ctdn/decoder from checkpoint,
    # set eval() and requires_grad_(False) on all three.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calib_pairs = load_calibration_pairs(args.data_root, args.n_calib_images)

    # Coarse log-spaced grid first (matches Agaian's own logarithmic-scale search,
    # their Sec III.A, "computation is conducted within a logarithmic scale").
    tau_grid = [0.5, 0.75, 1.0, 1.25, 1.5]
    lambda_grid = [0.6, 0.8, 1.0, 1.2, 1.4]
    vartheta_grid = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

    results = grid_search(encoder, ctdn, decoder, calib_pairs, device,
                           tau_grid, lambda_grid, vartheta_grid, args.stage1_weight)

    print(f"\nTop 10 candidates for target='{args.target}':")
    for r in results[:10]:
        print(f"  tau={r['tau']:.3f}  lambda={r['lambda']:.3f}  vartheta={r['vartheta']:.3f}  "
              f"LOE(input)={r['loe_input']:.2f}  LOE(ref)={r['loe_ref']:.2f}  "
              f"combined={r['combined']:.2f}")

    best = results[0]
    print(f"\nBest: tau={best['tau']}, lambda={best['lambda']}, vartheta={best['vartheta']}")
    print("Manually inspect the top few candidates' decoded images before committing —"
          " pick the one that looks best AND has low LOE, exactly as Agaian et al. cross-check"
          " Fig. 3's numeric optimum against Fig. 9's visual comparison.")

    # Optionally: run a finer grid centered on `best` (half-step refinement) before finalizing.


if __name__ == "__main__":
    main()
```

**Agent instructions for this step:**
- Fill in the `TODO(agent)` blocks using the real classes/loaders found in Step 0.
- Confirm CTDN's actual forward signature (does it return `(R, L)` for one image, or does it need
  both low/normal-light images at once per the paper's two-branch design? Check Fig. 2/Fig. 4 of
  the paper against the real `forward()` — the paper decomposes `F_low` and `F_high` together but
  for calibration we only need the low-light branch, i.e. `R_low, L_low`).
- Run this once for `--target scc` (calibrating the `L_scc` pseudo-label composition, which only
  needs `R_low, L_low` — no paired normal-light image required beyond the LOL ground truth used
  for the reference-LOE term) before attempting `--target x0` (which needs `L_high` from an
  actual normal-light image, requires slightly more loader wiring).
- Use a calibration subset (100–200 image pairs) from the LOL **training** split only — never the
  test split, to avoid contaminating the final evaluation numbers.

---

## 7. Step 5 — Retrain stage 2 with calibrated constants, run the ablation

1. Take the best `(tau, lambda, vartheta)` from calibration, write them into
   `configs/unsupervised.yml` under `retinex.tau_scc/lambda_scc/vartheta_scc` (and `_x0` if that
   grid was also run).
2. Set `retinex.mode: "generalized"`.
3. Re-run stage-2 diffusion training (freeze encoder/CTDN/decoder exactly as the paper's own
   two-stage protocol already does — Sec 3.4 / 4.1 of the paper: 4×10^5 iterations, Adam,
   lr 2×10⁻⁵).
4. Evaluate on LOL test set and DICM, using the existing eval script, reporting **exactly the
   metrics LightenDiffusion's own Table 2 ablation reports**: on LOL (paired) — **PSNR, SSIM,
   LPIPS**; on DICM (unpaired, no ground truth) — **NIQE, PI**; plus **Time (s)** per image
   (400×600×3, same convention as their Table 2 caption). Do **not** expect an LOE column in the
   existing eval script — LightenDiffusion's paper never reports LOE. LOE is specific to the
   *Agaian* paper and is used here only as our own calibration objective in Step 4, not as part
   of the final comparison table.
5. Additionally report calibration-set LOE for each row (the value that was actually minimized
   in Step 4) as a *separate diagnostic column*, clearly labeled as "not in original paper," so
   it's obvious to a reader which numbers are directly comparable to Table 2 and which aren't.
6. Produce a comparison table with (at minimum) these rows, mirroring Table 2's exact format:

   | Method | Time (s)↓ | PSNR↑ | SSIM↑ | LPIPS↓ | NIQE↓ | PI↓ | LOE↓ (ours, not in orig. paper) |
   |---|---|---|---|---|---|---|---|
   | Baseline (classical, τ=λ=ϑ=1) | | | | | | | |
   | Generalized, LOE-calibrated (scc only) | | | | | | | |
   | Generalized, LOE-calibrated (scc + x0) | | | | | | | |

   The first six columns should be directly comparable to LightenDiffusion's published Table 1
   (row "Ours": PSNR 20.453, SSIM 0.803, LPIPS 0.192 on LOL; NIQE 3.724, PI 3.144 on DICM) and
   Table 2 row 11 "Default" (Time 0.314s) — use those published numbers as the sanity-check
   target for your own re-run of the classical baseline before trusting the generalized rows.

6. Report training stability (any NaNs, loss curve behavior) and qualitative visual comparisons
   on a handful of images (reproduce a Fig. 8/Fig. 9-style panel if easy).

---

## 8. Acceptance criteria

- [ ] `generalized_retinex_compose` implemented, unit-tested against the classical special case.
- [ ] `compute_loe` implemented, unit-tested (self-comparison ≈ 0).
- [ ] Config toggle (`retinex.mode`) added; `classical` mode reproduces original behavior
      bit-for-bit (up to `eps`-induced floating point noise) when compared to the unmodified repo.
- [ ] `calibrate_retinex_params.py` runs end-to-end on a calibration subset and prints/saves a
      ranked table of candidate `(τ, λ, ϑ)` with both LOE components.
- [ ] Best constants documented (in the config and in a short `RESULTS.md` or similar) with the
      calibration LOE values that justified the choice.
- [ ] Stage-2 retrained with generalized composition; full evaluation table produced on LOL test
      + DICM, in the same metric format as the paper's Table 2.
- [ ] No regression: `retinex.mode: classical` path is unchanged from upstream behavior.

---

## 9. Known risks / things to flag back to the user rather than silently deciding

- If CTDN's `R`/`L` outputs are *not* already non-negative, note this clearly — don't silently
  add a new activation function without flagging it, since that changes the pretrained
  checkpoint's behavior beyond just the compose step.
- If decoding candidate compositions during calibration is too slow for a full grid (each grid
  point requires N decoder forward passes), reduce `n_calib_images` and/or the grid resolution
  first, and report actual wall-clock cost before scaling up — don't silently truncate the search
  in a way that changes which optimum gets reported.
- LOE is O(downsample_size²) per image pair — fine at 50×50, but flag if the existing codebase's
  image resolution conventions differ from the 400×600 used in the paper's own Table 2 timing.
- Two-stage weighting (`stage1_weight`) in the grid search is a design choice not explicitly
  pinned down by Agaian et al.'s two-step argmin (their Eq. 14 then Eq. 15 are sequential, not a
  single weighted sum) — implement the sequential two-step version instead if it's not
  meaningfully more expensive: first restrict the grid to candidates within some tolerance of the
  best `LOE(input)`, then pick the best `LOE(ref)` among those. Note whichever version you pick.
