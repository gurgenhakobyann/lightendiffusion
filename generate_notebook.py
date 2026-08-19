import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# LightenDiffusion: Generalized Retinex Calibration & Training on Google Colab GPU\n",
    "\n",
    "This notebook guides you through running **Agaian's Generalized Retinex Composition** calibrated via **Lightness Order Error (LOE)** on Google Colab with GPU acceleration."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Check GPU Availability"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!nvidia-smi"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Clone Repository & Install Dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clone your repository\n",
    "!git clone https://github.com/gurgenhakobyann/lightendiffusion.git\n",
    "%cd lightendiffusion\n",
    "\n",
    "# Install dependencies\n",
    "!pip install einops pyyaml opencv-python scikit-image pytest tqdm"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Upload or Download Dataset and Stage-1 Weights\n",
    "\n",
    "Place `LOL-v1.zip` and `stage1_weight.pth.tar` into the `lightendiffusion/` directory on Colab.\n",
    "You can upload them using the Colab file browser on the left, or mount Google Drive:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Option A: Mount Google Drive if you stored your files there\n",
    "# from google.colab import drive\n",
    "# drive.mount('/content/drive')\n",
    "# !cp /content/drive/MyDrive/LOL-v1.zip .\n",
    "# !cp /content/drive/MyDrive/stage1_weight.pth.tar .\n",
    "\n",
    "# Prepare dataset folders and indices\n",
    "!python prepare_dataset.py"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Run Unit Tests"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!python tests/test_generalized_retinex.py\n",
    "!python tests/test_loe.py"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Run LOE Calibration (Find Optimal Retinex Parameters)\n",
    "\n",
    "This runs LOE grid search on the LOL training set to find optimal $(\\tau, \\lambda, \\vartheta)$ exponents."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Calibrate SCC pseudo-label composition\n",
    "!python calibrate_retinex_params.py --target scc --n_calib_images 50 --save_json calib_scc.json\n",
    "\n",
    "# 2. Calibrate Forward Diffusion x0 composition\n",
    "!python calibrate_retinex_params.py --target x0 --n_calib_images 50 --save_json calib_x0.json"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Train Stage-2 Diffusion Model with Calibrated Retinex"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!python train.py --config unsupervised.yml"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Evaluate and Compute Metrics (PSNR, SSIM, LOE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run inference on the 15 LOL test images\n",
    "!python evaluate.py --config unsupervised.yml --resume ckpt/stage2/model_latest.pth.tar\n",
    "\n",
    "# Compute quantitative metrics\n",
    "!python compute_metrics.py"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("LightenDiffusion_Colab.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
print("Created LightenDiffusion_Colab.ipynb")
