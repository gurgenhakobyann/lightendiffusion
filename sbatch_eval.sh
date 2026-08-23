#!/bin/bash
#SBATCH --job-name=lightendiff_eval
#SBATCH --partition=research
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_eval_%j.out
#SBATCH --error=slurm_eval_%j.err

echo "=== Evaluation job started on $(hostname) at $(date) ==="
nvidia-smi

cd $SLURM_SUBMIT_DIR

# 1. Run diffusion inference on LOL test images
/mnt/weka/ghakobyan/.conda/envs/lightendiff/bin/python evaluate.py --config unsupervised.yml --resume ckpt/stage2/model_latest.pth.tar

# 2. Compute PSNR, SSIM, and LOE
/mnt/weka/ghakobyan/.conda/envs/lightendiff/bin/python compute_metrics.py

echo "=== Evaluation job finished at $(date) ==="
