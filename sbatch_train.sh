#!/bin/bash
#SBATCH --job-name=lightendiff_main
#SBATCH --partition=research
#SBATCH --time=34:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

echo "=== Job started on $(hostname) at $(date) ==="
echo "GPU allocation:"
nvidia-smi

# Navigate to project directory (main branch)
cd $SLURM_SUBMIT_DIR

# Run Stage-2 Diffusion Training (Generalized Retinex / main branch)
# --resume is optional: remove it for a fresh training run, add it to continue
/mnt/weka/ghakobyan/.conda/envs/lightendiff/bin/python train.py --config unsupervised.yml

echo "=== Job finished at $(date) ==="
