#!/bin/bash
#SBATCH --job-name=lightendiff_train
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

echo "=== Job started on $(hostname) at $(date) ==="
echo "GPU allocation:"
nvidia-smi

# Load Miniconda / Anaconda environment
if [ -f "/home/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/$USER/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/home/$USER/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/home/$USER/anaconda3/etc/profile.d/conda.sh"
fi

conda activate lightendiff

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Run Stage-2 Diffusion Training
python train.py --config configs/unsupervised.yml

echo "=== Job finished at $(date) ==="
