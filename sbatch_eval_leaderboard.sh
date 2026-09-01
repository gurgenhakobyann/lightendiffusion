#!/bin/bash
#SBATCH --job-name=eval_leaderboard
#SBATCH --partition=research
#SBATCH --time=00:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --output=slurm_leaderboard_%j.out
#SBATCH --error=slurm_leaderboard_%j.err

echo "=== Leaderboard Evaluation started on $(hostname) at $(date) ==="
nvidia-smi

cd $SLURM_SUBMIT_DIR

/mnt/weka/ghakobyan/.conda/envs/lightendiff/bin/python eval_all_checkpoints.py

echo "=== Finished at $(date) ==="
