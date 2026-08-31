#!/bin/bash
#SBATCH --job-name=lightendiff_eval
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --output=slurm_eval_%j.out
#SBATCH --error=slurm_eval_%j.err

echo "=== Evaluation job started on $(hostname) at $(date) ==="
nvidia-smi

cd $SLURM_SUBMIT_DIR

PYTHON=/mnt/weka/ghakobyan/.conda/envs/lightendiff/bin/python
CKPT=ckpt/stage2/model_latest.pth.tar

mkdir -p results/LOLv1 results/LSRW results/DICM results/NPE results/VV

# 1. Run diffusion inference on LOL test images
$PYTHON evaluate.py --config unsupervised.yml --resume $CKPT

# 2. Compute full benchmark table (LOL, LSRW, DICM, NPE, VV)
$PYTHON evaluate_full_benchmarks.py --results_root results --data_root .

echo "=== Evaluation job finished at $(date) ==="
