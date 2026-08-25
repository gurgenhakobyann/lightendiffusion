#!/bin/bash
#SBATCH --job-name=lightendiff_eval
#SBATCH --partition=research
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_eval_%j.out
#SBATCH --error=slurm_eval_%j.err

echo "=== Evaluation job started on $(hostname) at $(date) ==="
nvidia-smi

cd $SLURM_SUBMIT_DIR

PYTHON=/mnt/weka/ghakobyan/.conda/envs/lightendiff/bin/python
CKPT=/mnt/weka/ghakobyan/ckpt_main/stage2/model_latest.pth.tar

mkdir -p results/LOLv1

# Run inference on LOLv1 test set
$PYTHON evaluate.py --config unsupervised.yml --resume $CKPT

# Compute quantitative metrics
$PYTHON compute_metrics.py

echo "=== Evaluation job finished at $(date) ==="
