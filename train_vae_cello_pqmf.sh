#!/bin/bash
#SBATCH --job-name=vae_cello_pqmf
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
RAW_DATA_DIR="$SCRATCH/datasets/CCBachCello"
NPY_PATH="$SCRATCH/datasets/CCBachCello_48k_32768.npy"
OUT_DIR="$SCRATCH/runs/vae_cello_pqmf"
TMP_CONFIG="$SCRATCH/tmp/config_cello_pqmf_sherlock.json"

mkdir -p "$OUT_DIR" "$(dirname "$TMP_CONFIG")"

# --- 3. Preprocessing: build .npy chunk array if not yet done ---
if [ ! -f "$NPY_PATH" ]; then
    echo "Building chunk array from $RAW_DATA_DIR..."
    "$VENV_PATH/bin/python" build_chunks.py \
        --src "$RAW_DATA_DIR" \
        --out "$NPY_PATH" \
        --sr 48000 \
        --chunk_size 32768 \
        --drop_silence 0.005
fi

# --- 4. Patch config with Sherlock paths ---
"$VENV_PATH/bin/python" - <<PYEOF
import json
with open("config_raw_cello_pqmf_48k.json") as f:
    cfg = json.load(f)
cfg["data_loader"]["args"]["npy_path"] = "$NPY_PATH"
cfg["trainer"]["save_dir"] = "$OUT_DIR"
with open("$TMP_CONFIG", "w") as f:
    json.dump(cfg, f, indent=4)
print("Patched config written to $TMP_CONFIG")
PYEOF

# --- 5. Training ---
CKPT="/scratch/users/nlabate/runs/vae_cello_pqmf/models/RawAudioVAE_PQMF_Cello_48k/0628_063747/checkpoint-epoch650.pth"

echo "Resuming VAE cello PQMF 48k training from epoch 650..."
"$VENV_PATH/bin/python" train.py -c "$TMP_CONFIG" -r "$CKPT"
