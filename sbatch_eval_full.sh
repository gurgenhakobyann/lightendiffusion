#!/bin/bash
#SBATCH --job-name=lighten_eval_full
#SBATCH --partition=research
#SBATCH --time=00:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --output=slurm_eval_full_%j.out
#SBATCH --error=slurm_eval_full_%j.err

echo "=== Full Benchmark Evaluation started on $(hostname) at $(date) ==="
nvidia-smi

cd $SLURM_SUBMIT_DIR

/mnt/weka/ghakobyan/.conda/envs/lightendiff/bin/python evaluate_all_lighten.py --resume ckpt/stage2/model_latest.pth.tar

echo "=== Finished at $(date) ==="
