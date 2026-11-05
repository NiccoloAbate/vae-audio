#!/bin/bash
#SBATCH --job-name=vae_drums
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# Generic one-shot VAE training job: the config is the first argument.
#   sbatch --job-name=snare    train_vae.sh config_raw_drums_22k_snare.json
#   sbatch --job-name=snare_ft train_vae.sh config_raw_drums_22k_snare_ft.json
#   sbatch --job-name=22k_steps train_vae.sh config_raw_drums_22k_steps.json
# Extra arguments are passed to train.py (e.g. -r <checkpoint> to resume).

set -e
CONFIG="$1"; shift || true
[ -f "$CONFIG" ] || { echo "usage: sbatch train_vae.sh <config.json> [train.py args]"; exit 1; }

# --- 1. Environment Setup ---
ml load system ffmpeg/7.1 system libsndfile/1.2.2

export UV_CACHE_DIR="$SCRATCH/.uv_cache"
export PATH="$SCRATCH/bin:$HOME/.local/bin:$PATH"

REPO_DIR="/home/users/nlabate/VAE/vae-audio"
cd "$REPO_DIR"

VENV_PATH="$SCRATCH/envs/vae_audio"
uv venv "$VENV_PATH" --python 3.10 2>/dev/null || true
uv pip install -r requirements.txt --python "$VENV_PATH/bin/python"

echo "Job: $SLURM_JOB_ID  config: $CONFIG"
nvidia-smi

# --- 2. Path Configuration ---
DATA_DIR="$SCRATCH/datasets/drums_oneshots"
OUT_DIR="$SCRATCH/runs/vae_drums"
TMP_CONFIG="$SCRATCH/tmp/$(basename "$CONFIG" .json)_${SLURM_JOB_ID:-local}.json"
mkdir -p "$OUT_DIR" "$(dirname "$TMP_CONFIG")"

# --- 3. Patch config with Sherlock paths (and check a fine-tune source exists) ---
"$VENV_PATH/bin/python" - <<PYEOF
import json, os, sys
with open("$CONFIG") as f:
    cfg = json.load(f)
cfg["data_loader"]["args"]["data_dir"] = ["$DATA_DIR"]
cfg["data_loader"]["args"]["num_workers"] = 8
cfg["trainer"]["save_dir"] = "$OUT_DIR"
ft = cfg["trainer"].get("finetune_from")
if ft and not os.path.isfile(ft):
    sys.exit(f"finetune_from checkpoint not found: {ft}")
with open("$TMP_CONFIG", "w") as f:
    json.dump(cfg, f, indent=4)
print("Patched config written to $TMP_CONFIG")
PYEOF

# --- 4. Training ---
"$VENV_PATH/bin/python" train.py -c "$TMP_CONFIG" "$@"
