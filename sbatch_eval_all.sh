#!/bin/bash
#SBATCH --job-name=eval_all_ckpts
#SBATCH --partition=research
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_eval_all_%j.out
#SBATCH --error=slurm_eval_all_%j.err

echo "=== Evaluating all checkpoints on $(hostname) at $(date) ==="
nvidia-smi

cd $SLURM_SUBMIT_DIR

PYTHON=/mnt/weka/ghakobyan/.conda/envs/lightendiff/bin/python

$PYTHON eval_all_checkpoints.py --config unsupervised.yml --ckpt_dir /mnt/weka/ghakobyan/ckpt_main/stage2

echo "=== Checkpoint evaluation finished at $(date) ==="
