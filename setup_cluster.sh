#!/bin/bash
# Run this on the LOGIN NODE (where you have internet)

echo "[1/3] Cleaning old environment..."
rm -rf venv

echo "[2/3] Setting up new environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .

echo "[3/3] Downloading data to cache..."
python -m src.warmup

echo "-----------------------------------"
echo "Setup complete! Now you can submit the job:"
echo "sbatch chess.sh"
