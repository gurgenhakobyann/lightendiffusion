#!/bin/bash
#SBATCH --job-name=lightendiff_train
#SBATCH --partition=research
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

echo "=== Job started on $(hostname) at $(date) ==="
echo "GPU allocation:"
nvidia-smi

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Run Stage-2 Diffusion Training, resuming from latest checkpoint
/mnt/weka/ghakobyan/.conda/envs/lightendiff/bin/python train.py --config unsupervised.yml --resume ckpt/stage2/model_latest.pth.tar

echo "=== Job finished at $(date) ==="
