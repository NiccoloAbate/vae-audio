#!/bin/bash
#SBATCH --job-name=restore_drums
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# Restore the drum one-shot dataset from its permanent copy in $GROUP_HOME to
# $SCRATCH (which is purged after 90 days without modification), where the
# training jobs read it.
#
#   sbatch restore_drums_dataset.sh
#
# The relative paths must stay identical to the source: the train/val/test
# split (oneshot_split) is a hash of each file's path relative to the dataset
# root. Safe to re-run; files already present with the same size are skipped.
#
# Timestamps are deliberately NOT preserved (--no-times): the copies get their
# real modification time, rather than the original extraction date.

set -e

SRC="$GROUP_HOME/nlabate/datasets/DrumOneShots/"
DST="$SCRATCH/datasets/drums_oneshots/"

count() { find "$1" -type f | grep -ciE '\.(wav|aif|aiff|flac)$' || true; }

# --delete makes scratch an exact mirror of the permanent copy (files removed or renamed
# there disappear here too). Refuse to run it against an empty or missing source.
N_SRC=$(count "$SRC")
[ "$N_SRC" -gt 1000 ] || { echo "ERROR: only $N_SRC audio files under $SRC; refusing to mirror" >&2; exit 1; }

mkdir -p "$DST"
echo "Mirroring $SRC -> $DST"
rsync -rlv --delete --no-times --no-perms --omit-dir-times --size-only "$SRC" "$DST"

# --- Verify ---
N_DST=$(count "$DST")
echo "audio files: source=$N_SRC  scratch=$N_DST"
if [ "$N_SRC" -ne "$N_DST" ] || [ "$N_DST" -eq 0 ]; then
    echo "ERROR: file counts do not match" >&2
    exit 1
fi
echo "OK"
