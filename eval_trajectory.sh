#!/bin/bash
#SBATCH --job-name=traj
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Val-split FAD / STFT for every saved checkpoint of one run (eval_trajectory.py).
# One job per run, so the jobs share nothing but the environments:
#
#   for r in Drums_44k_b005 Drums_44k_b002 Drums_44k_b01 Drums_44k Drums_22k \
#            Drums_22k_steps Drums_44k_PQMF; do
#       sbatch -J traj_$r eval_trajectory.sh $r
#   done
#
# Optional 2nd argument: the epoch stride (default 50). Resubmitting skips scored
# checkpoints. Results: $SCRATCH/eval/trajectory/<run>/{results_val,steps}.csv, small
# enough to download along with $SCRATCH/runs/vae_drums/log/ for plot_trajectory.py.
#
# Env check (run once): sbatch -J traj_check eval_trajectory.sh --check
#   scores Drums_44k_b005 epoch 1500 on the TEST split; locally that is FAD 0.062,
#   STFT 0.628, so a big gap means the Sherlock FAD env differs from the local one.

set -e
RUN="${1:?usage: sbatch eval_trajectory.sh <run name> [every] | --check}"
EVERY="${2:-50}"

ml load system ffmpeg/7.1 system libsndfile/1.2.2
export UV_CACHE_DIR="$SCRATCH/.uv_cache"
export UV_LINK_MODE=copy
export OPENBLAS_NUM_THREADS=1
export PATH="$SCRATCH/bin:$HOME/.local/bin:$PATH"

REPO_DIR="/home/users/nlabate/VAE/vae-audio"
cd "$REPO_DIR"

# Training env (eval_pipeline.py), as in train_vae_drums.sh
VENV_PATH="$GROUP_HOME/nlabate/envs/VAE/vae-audio"
PY="$VENV_PATH/bin/python"
(
    flock 9
    uv venv "$VENV_PATH" --python 3.10 2>/dev/null || true
    uv pip install -r requirements.txt -c requirements-lock.txt --python "$PY"
) 9>"$VENV_PATH.lock"

# FAD env: fadtk needs Python <= 3.13. Versions pinned to the local env that produced
# the milestone numbers. The CLAP weights download into the package on first use, so
# fetch them here under the lock rather than racing in parallel jobs.
FAD_VENV="$GROUP_HOME/nlabate/envs/fad"
(
    flock 9
    uv venv "$FAD_VENV" --python 3.12 2>/dev/null || true
    uv pip install --python "$FAD_VENV/bin/python" fadtk==1.1.0 laion_clap==1.1.7 \
        transformers==4.57.6 numpy==1.26.4 torch torchaudio torchvision
    "$FAD_VENV/bin/python" -c "from fadtk.model_loader import CLAPLaionModel; CLAPLaionModel('audio').load_model()"
) 9>"$FAD_VENV.lock"

DATA_DIR="$SCRATCH/datasets/drums_oneshots"
RUNS_DIR="$SCRATCH/runs/vae_drums/models"
OUT_ROOT="$SCRATCH/eval/trajectory"
echo "Job: $SLURM_JOB_ID  run: $RUN  commit: $(git rev-parse --short HEAD)"
nvidia-smi

if [ "$RUN" = "--check" ]; then
    CKPT=$(ls "$RUNS_DIR"/Drums_44k_b005/*/checkpoint-epoch1500.pth | tail -1)
    "$PY" eval_pipeline.py -r "$CKPT" --data_dir "$DATA_DIR" --name check_44k_b005_ep1500 \
        -o "$OUT_ROOT/check/work" --results "$OUT_ROOT/check/results_test.csv" \
        --ref_dir "$OUT_ROOT/check/reference" --no_truncation --fadtk "$FAD_VENV/bin/fadtk"
    rm -rf "$OUT_ROOT/check/work/audio"
    exit 0
fi

"$PY" eval_trajectory.py --runs_dir "$RUNS_DIR" --run "$RUN" --data_dir "$DATA_DIR" \
    --every "$EVERY" -o "$OUT_ROOT/$RUN" --fadtk "$FAD_VENV/bin/fadtk"
