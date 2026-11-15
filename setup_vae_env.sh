#!/bin/bash
#SBATCH --job-name=setup_vae_env
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Build the training venv in $GROUP_HOME (not purged) from requirements-lock.txt.
#
#   sbatch setup_vae_env.sh
#
# Why a lock file: requirements.txt is unpinned, and today's newest releases
# (numba/llvmlite/pillow/...) have no wheels for this node's old glibc, so uv
# tries to compile them (llvmlite then fails on a missing cmake). The lock file
# pins the versions of the June venv, which are known to install here.
# Packages without a compatible wheel (e.g. pandas, soxr) are compiled from
# source, which is why this runs as a job.

set -e

ml load system ffmpeg/7.1 system libsndfile/1.2.2

export UV_CACHE_DIR="$SCRATCH/.uv_cache"
export UV_LINK_MODE=copy    # cache (scratch) and venv (group home) are on different filesystems
export PATH="$SCRATCH/bin:$HOME/.local/bin:$PATH"
export MAKEFLAGS="-j$SLURM_CPUS_PER_TASK"
export CMAKE_BUILD_PARALLEL_LEVEL="$SLURM_CPUS_PER_TASK"

cd /home/users/nlabate/VAE/vae-audio

VENV_PATH="$GROUP_HOME/nlabate/envs/VAE/vae-audio"
PY="$VENV_PATH/bin/python"
mkdir -p "$(dirname "$VENV_PATH")"

uv venv "$VENV_PATH" --python 3.10 2>/dev/null || true
uv pip install -r requirements.txt -c requirements-lock.txt --python "$PY"

echo "--- verify ---"
"$PY" -c "
import torch, torchaudio, librosa, soundfile, numba, pandas, soxr, gradio
print('torch', torch.__version__, '| torchaudio', torchaudio.__version__, '| librosa', librosa.__version__)
print('numba', numba.__version__, '| pandas', pandas.__version__, '| soxr', soxr.__version__)
"
echo OK
