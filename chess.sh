#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8              # Increased CPUs for data loading
#SBATCH --gres=gpu:1      # 1 GPU per job
#SBATCH --time=12:00:00   # Sufficient for 3 epochs
#SBATCH --mem=32G         
#SBATCH --account=m25146
#SBATCH --job-name=chess_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_LAUNCH_BLOCKING=1

# Exit immediately if a command exits with a non-zero status
set -e

# Load modules (adjust based on your cluster, e.g., module load cuda/11.8)
# module load cuda/12.1  <-- Uncomment if needed

# Activate venv
if [ ! -d "venv" ]; then
    echo "Creating venv..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -e .
else
    source venv/bin/activate
fi

# Define paths
MODEL_DIR="./output_chess_model"
mkdir -p logs

echo "============================================"
echo "Starting Chess Training Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Output Dir: $MODEL_DIR"
echo "============================================"

# 1. Training
echo "[1/2] Starting Training..."
python -m src.train \
    --output_dir $MODEL_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 64 \
    --save_steps 2000

# 2. Evaluation
echo "--------------------------------------------"
echo "[2/2] Starting Evaluation..."

# Legal moves check
echo "Running legal move evaluation..."
python -m src.evaluate \
    --model_path "${MODEL_DIR}/final_model" \
    --mode legal \
    --n_positions 1000

# Win rate check
echo "Running win rate check vs Stockfish..."
python -m src.evaluate \
    --model_path "${MODEL_DIR}/final_model" \
    --mode winrate \
    --n_games 50 \
    --stockfish_level 1

echo "============================================"
echo "Pipeline Completed!"
