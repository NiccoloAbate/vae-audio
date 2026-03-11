"""
Prepare a balanced subset of Medley-solos-DB for RawAudioVAE training.

Samples up to MAX_PER_CLASS clips per instrument from the 'training' split,
copies them into dataset/medley-solos/organized/trainingdata/{instrument}/
to match the ESC-50 directory structure expected by RawAudioDataLoader.

Usage:
    python prepare_medley.py
"""

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

METADATA   = Path("dataset/medley-solos/metadata.csv")
AUDIO_DIR  = Path("dataset/medley-solos/Medley-solos-DB")
OUT_DIR    = Path("dataset/medley-solos/organized/trainingdata")
MAX_PER_CLASS = 500
SEED       = 42

random.seed(SEED)

# ── Read metadata ────────────────────────────────────────
by_class = defaultdict(list)
with open(METADATA) as f:
    for row in csv.DictReader(f):
        # use all splits — dataloader handles its own validation split
        pass
        by_class[row["instrument"]].append(row)

# ── Sample and copy ──────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
total = 0
for instrument, rows in sorted(by_class.items()):
    sample = random.sample(rows, min(MAX_PER_CLASS, len(rows)))
    dest   = OUT_DIR / instrument
    dest.mkdir(exist_ok=True)
    for row in sample:
        # filename: Medley-solos-DB_{subset}-{instrument_id}_{uuid4}.wav
        fname = f"Medley-solos-DB_{row['subset']}-{row['instrument_id']}_{row['uuid4']}.wav"
        src   = AUDIO_DIR / fname
        if not src.exists():
            print(f"  WARNING: missing {src}")
            continue
        shutil.copy2(src, dest / fname)
    print(f"  {instrument:30s} {len(sample):4d} clips  ->  {dest}")
    total += len(sample)

print(f"\nTotal clips copied: {total}")
print(f"Output: {OUT_DIR}")
