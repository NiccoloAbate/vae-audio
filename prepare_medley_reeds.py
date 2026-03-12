"""
Prepare a balanced reed-instrument subset of Medley-solos-DB.

Instruments: clarinet, tenor saxophone
Clips per class: up to MAX_PER_CLASS (limited by tenor sax @ 477 total)

Output structure matches ESC-50 / RawAudioDataLoader:
    dataset/medley-solos/organized_reeds/trainingdata/{instrument}/

Usage:
    python prepare_medley_reeds.py
"""

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

METADATA      = Path("dataset/medley-solos/metadata.csv")
AUDIO_DIR     = Path("dataset/medley-solos/Medley-solos-DB")
OUT_DIR       = Path("dataset/medley-solos/organized_reeds/trainingdata")
INSTRUMENTS   = {"clarinet", "tenor saxophone"}
MAX_PER_CLASS = 477   # match tenor sax total
SEED          = 42

random.seed(SEED)

by_class = defaultdict(list)
with open(METADATA) as f:
    for row in csv.DictReader(f):
        if row["instrument"] in INSTRUMENTS:
            by_class[row["instrument"]].append(row)

OUT_DIR.mkdir(parents=True, exist_ok=True)
total = 0
for instrument, rows in sorted(by_class.items()):
    sample = random.sample(rows, min(MAX_PER_CLASS, len(rows)))
    dest   = OUT_DIR / instrument
    dest.mkdir(exist_ok=True)
    for row in sample:
        fname = f"Medley-solos-DB_{row['subset']}-{row['instrument_id']}_{row['uuid4']}.wav"
        src   = AUDIO_DIR / fname
        if not src.exists():
            print(f"  WARNING: missing {src}")
            continue
        shutil.copy2(src, dest / fname)
    print(f"  {instrument:20s} {len(sample):4d} clips  ->  {dest}")
    total += len(sample)

# symlink testdata → trainingdata so evaluate_raw.py can find files
test_dir = OUT_DIR.parent / "testdata"
if not test_dir.exists():
    test_dir.symlink_to(OUT_DIR)
    print(f"  Symlinked testdata -> trainingdata")

print(f"\nTotal clips: {total}")
print(f"Output: {OUT_DIR}")
