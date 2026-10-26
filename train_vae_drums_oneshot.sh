#!/bin/bash
#SBATCH --job-name=vae_drums_oneshot
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -e

# --- 1. Environment Setup ---
ml load system ffmpeg/7.1 system libsndfile/1.2.2

export UV_CACHE_DIR="$SCRATCH/.uv_cache"
export PATH="$SCRATCH/bin:$HOME/.local/bin:$PATH"

REPO_DIR="/home/users/nlabate/VAE/vae-audio"
cd "$REPO_DIR"

VENV_PATH="$SCRATCH/envs/vae_audio"

echo "Creating/verifying venv at $VENV_PATH..."
uv venv "$VENV_PATH" --python 3.10 2>/dev/null || true
echo "Syncing dependencies..."
uv pip install -r requirements.txt --python "$VENV_PATH/bin/python"

echo "Job: $SLURM_JOB_ID"
nvidia-smi

# --- 2. Path Configuration ---
DATA_DIR="$SCRATCH/datasets/drums_oneshots"
OUT_DIR="$SCRATCH/runs/vae_drums_oneshot"
TMP_CONFIG="$SCRATCH/tmp/config_drums_oneshot_sherlock.json"

mkdir -p "$OUT_DIR" "$(dirname "$TMP_CONFIG")"

# --- 3. Patch config with Sherlock paths ---
"$VENV_PATH/bin/python" - <<PYEOF
import json
with open("config_raw_drums_oneshot.json") as f:
    cfg = json.load(f)
cfg["data_loader"]["args"]["data_dir"] = ["$DATA_DIR"]
cfg["data_loader"]["args"]["num_workers"] = 8
cfg["trainer"]["save_dir"] = "$OUT_DIR"
with open("$TMP_CONFIG", "w") as f:
    json.dump(cfg, f, indent=4)
print("Patched config written to $TMP_CONFIG")
PYEOF

# --- 4. Training ---
# To resume from a checkpoint, add: -r /path/to/checkpoint.pth
echo "Starting VAE drums one-shot training..."
"$VENV_PATH/bin/python" train.py -c "$TMP_CONFIG"
