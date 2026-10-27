#!/bin/bash
#SBATCH --job-name=vae_drums
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# Train one drum one-shot run. The config is the only argument:
#
#   sbatch -J drums_22k      train_vae_drums.sh config_raw_drums_22k.json
#   sbatch -J drums_44k      train_vae_drums.sh config_raw_drums_44k.json
#   sbatch -J drums_44k_pqmf train_vae_drums.sh config_raw_drums_44k_pqmf.json
#   sbatch -J drums_9th_44k  train_vae_drums.sh config_raw_drums_9th_44k.json
#
# Resubmitting the same command continues the run from its latest checkpoint
# (model, optimizer, discriminator and LR schedule). To queue a continuation
# ahead of time:  sbatch -d afterany:<jobid> -J <name> train_vae_drums.sh <config>

set -e
CONFIG="${1:?usage: sbatch train_vae_drums.sh <config.json>}"

# --- 1. Environment Setup ---
ml load system ffmpeg/7.1 system libsndfile/1.2.2

export UV_CACHE_DIR="$SCRATCH/.uv_cache"
export UV_LINK_MODE=copy    # cache (scratch) and venv (group home) are on different filesystems
export PATH="$SCRATCH/bin:$HOME/.local/bin:$PATH"

REPO_DIR="/home/users/nlabate/VAE/vae-audio"
cd "$REPO_DIR"

# The venv lives in $GROUP_HOME, not $SCRATCH: scratch is purged after 90 days
# and that breaks environments (pyvenv.cfg goes first).
VENV_PATH="$GROUP_HOME/nlabate/envs/VAE/vae-audio"
PY="$VENV_PATH/bin/python"
mkdir -p "$(dirname "$VENV_PATH")"

# Several jobs may start at once and share this venv, so sync it under a lock.
(
    flock 9
    echo "Creating/verifying venv at $VENV_PATH..."
    uv venv "$VENV_PATH" --python 3.10 2>/dev/null || true
    echo "Syncing dependencies..."
    uv pip install -r requirements.txt -c requirements-lock.txt --python "$PY"
) 9>"$VENV_PATH.lock"

echo "Job: $SLURM_JOB_ID  config: $CONFIG  commit: $(git rev-parse --short HEAD)"
nvidia-smi

# --- 2. Path Configuration ---
DATA_DIR="$SCRATCH/datasets/drums_oneshots"
OUT_DIR="$SCRATCH/runs/vae_drums"
NAME=$("$PY" -c "import json; print(json.load(open('$CONFIG'))['name'])")
TMP_CONFIG="$SCRATCH/tmp/${NAME}_sherlock.json"

mkdir -p "$OUT_DIR" "$(dirname "$TMP_CONFIG")"

# --- 3. Patch config with Sherlock paths ---
"$PY" - <<PYEOF
import json
with open("$CONFIG") as f:
    cfg = json.load(f)
cfg["data_loader"]["args"]["data_dir"] = ["$DATA_DIR"]
cfg["data_loader"]["args"]["num_workers"] = 8
cfg["trainer"]["save_dir"] = "$OUT_DIR"
with open("$TMP_CONFIG", "w") as f:
    json.dump(cfg, f, indent=4)
print("Patched config written to $TMP_CONFIG")
PYEOF

# --- 4. Resume from the latest checkpoint of this run, if any ---
CKPT=$("$PY" - <<PYEOF
import glob, re
ckpts = glob.glob("$OUT_DIR/models/$NAME/*/checkpoint-epoch*.pth")
print(max(ckpts, key=lambda p: int(re.search(r"epoch(\d+)\.pth$", p).group(1))) if ckpts else "")
PYEOF
)

# --- 5. Training ---
if [ -n "$CKPT" ]; then
    echo "Resuming $NAME from $CKPT"
    "$PY" train.py -c "$TMP_CONFIG" -r "$CKPT"
else
    echo "Starting $NAME from scratch"
    "$PY" train.py -c "$TMP_CONFIG"
fi
