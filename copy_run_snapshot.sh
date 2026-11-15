#!/bin/bash
#SBATCH --job-name=copy_snapshot
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

# Copy selected checkpoints (and each run's config.json) from $SCRATCH to
# $GROUP_HOME, keeping the same layout under the repo's group-home folder:
#
#   $SCRATCH/runs/vae_drums/models/<run>/<timestamp>/...
#   -> $GROUP_HOME/nlabate/VAE/vae-audio/runs/vae_drums/models/<run>/<timestamp>/...
#
#   sbatch copy_run_snapshot.sh
#
# Pre-GAN = last checkpoint before adversarial training starts (adv_start_epoch
# 500, or 2000 for the 9th-Wonder run); post-GAN = a recent checkpoint. Training
# keeps running; the post-GAN files are not the newest ones, so none is mid-write.

set -e

SRC="$SCRATCH/runs/vae_drums"
DST="$GROUP_HOME/nlabate/VAE/vae-audio/runs/vae_drums"

# Paths relative to $SRC. Pass your own as arguments to copy something else:
#   sbatch copy_run_snapshot.sh models/Drums_22k/1027_154437/checkpoint-epoch1500.pth ...
FILES=(
    models/Drums_44k/1027_125950/checkpoint-epoch490.pth
    models/Drums_44k/1027_125950/checkpoint-epoch900.pth
    models/Drums_44k/1027_125950/config.json
    models/Drums_44k_PQMF/1027_130315/checkpoint-epoch490.pth
    models/Drums_44k_PQMF/1027_130315/checkpoint-epoch1180.pth
    models/Drums_44k_PQMF/1027_130315/config.json
    models/Drums_22k/1027_154437/checkpoint-epoch490.pth
    models/Drums_22k/1027_154437/checkpoint-epoch1110.pth
    models/Drums_22k/1027_154437/config.json
    models/Drums_9th_44k/1027_154911/checkpoint-epoch1960.pth
    models/Drums_9th_44k/1027_154911/checkpoint-epoch3440.pth
    models/Drums_9th_44k/1027_154911/config.json
)

[ $# -gt 0 ] && FILES=("$@")

mkdir -p "$DST"
cd "$SRC"
for f in "${FILES[@]}"; do
    [ -f "$f" ] || { echo "ERROR: missing source $SRC/$f" >&2; exit 1; }
done

rsync -avR "${FILES[@]}" "$DST/"

# --- Verify: every file present at the destination with identical content ---
for f in "${FILES[@]}"; do
    cmp -s "$SRC/$f" "$DST/$f" || { echo "ERROR: $f differs after copy" >&2; exit 1; }
done
echo "Copied and verified ${#FILES[@]} files:"
du -ch $(for f in "${FILES[@]}"; do echo "$DST/$f"; done) | tail -1
echo OK
