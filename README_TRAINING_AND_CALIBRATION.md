# Training, Calibration & Evaluation Guide for LightenDiffusion (Generalized Retinex)

This repository includes the implementation of **Agaian's Generalized Retinex Composition** ($f(R, L) = \tau \cdot R^\lambda \cdot L^\vartheta$) calibrated via **Lightness Order Error (LOE)** search, as described in your advisor's specification.

---

## 1. Environment Setup (On the target training machine)

### 1.1 Create Conda Environment
```bash
conda create -n lightendiff python=3.9 -y
conda activate lightendiff
```

### 1.2 Install Dependencies
Install PyTorch for your CUDA version (e.g. CUDA 11.8):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
Install the rest of the required packages:
```bash
pip install -r requirements.txt
pip install pyyaml einops tqdm opencv-python pytest
```

---

## 2. Running Unit Tests
To verify numerical stability and LOE calculation:
```bash
python tests/test_generalized_retinex.py
python tests/test_loe.py
```

---

## 3. Step-by-Step Workflow

### Step 1: Calibrate $(\tau, \lambda, \vartheta)$ via LOE Search
Run the calibration script on the LOL training set to find the optimal composition parameters:

```bash
# 1. Calibrate SCC pseudo-label composition (F_tilde_low = compose(R_low, L_low))
python calibrate_retinex_params.py \
    --config configs/unsupervised.yml \
    --checkpoint stage1_weight.pth.tar \
    --data_root LOLdataset \
    --n_calib_images 100 \
    --target scc \
    --save_json calibration_scc_results.json

# 2. (Optional) Calibrate forward diffusion target (x0 = compose(R_low, L_high))
python calibrate_retinex_params.py \
    --config configs/unsupervised.yml \
    --checkpoint stage1_weight.pth.tar \
    --data_root LOLdataset \
    --n_calib_images 100 \
    --target x0 \
    --save_json calibration_x0_results.json
```

The script will print the top-ranked parameter combinations minimizing LOE.

---

### Step 2: Configure `configs/unsupervised.yml`
Update the `retinex` block in `configs/unsupervised.yml` with your chosen calibrated values:

```yaml
retinex:
    mode: "generalized"     # "classical" for baseline control, "generalized" for proposed method
    tau_x0: 1.0
    lambda_x0: 1.0
    vartheta_x0: 1.0
    tau_scc: 1.25           # Put best calibrated tau here
    lambda_scc: 1.20        # Put best calibrated lambda here
    vartheta_scc: 0.10      # Put best calibrated vartheta here
    eps: 0.0001
```

---

### Step 3: Run Stage-2 Training
Launch diffusion model training (uses Stage 1 weights `stage1_weight.pth.tar` to freeze the decomposition network, and trains the diffusion model with the generalized Retinex targets):

```bash
python train.py --config unsupervised.yml
```

The model checkpoints will be saved into `ckpt/stage2/` during training.

---

### Step 4: Evaluate the Trained Model
Run inference on the LOL validation set:

```bash
python evaluate.py --config unsupervised.yml --resume ckpt/stage2/stage2_weight.pth.tar
```
Enhanced output images will be saved in `results/LOLv1/`.

---

## 4. Summary of Code Additions
- `models/generalized_retinex.py`: Contains `generalized_retinex_compose` and `classical_retinex_compose`.
- `metrics/loe.py`: Vectorized implementation of Lightness Order Error (`compute_loe`).
- `calibrate_retinex_params.py`: LOE grid search optimizer for $(\tau, \lambda, \vartheta)$.
- `tests/`: Automated unit tests for retinex primitive and LOE metric.
- `configs/unsupervised.yml`: Includes full `retinex` configuration section.
- `LOLdataset/unpaired_train.txt`: Prepared training pair index (498 pairs) for out-of-the-box training.
