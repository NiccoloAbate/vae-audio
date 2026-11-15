"""
List the training files whose audio duplicates a val or test file, so training can
skip them (data_loader arg "exclude"). The split is a hash of each file's path, so
the same sample saved in two packs can land in train and test; this keeps the test
set unchanged and removes the leak from training instead.

Duplicates are found by a content fingerprint: mono, peak-normalized, trimmed to the
onset, the first 0.5 s quantized to 8 bits (catches copies at other bit depths or
gains, not resampled copies).

Usage, on the final dataset (it writes paths relative to --data_dir, so the list
also works on Sherlock if the dataset there is the same):
    python dedupe_splits.py --data_dir dataset/DrumOneShots
"""

import argparse
import hashlib
import os
from collections import defaultdict

import numpy as np
import soundfile as sf

from dataset.datasets import OneShotAudioDataset

TYPES = ['kick', 'snare', 'clap', 'hat', 'cymbal', 'tom', 'perc']


def fingerprint(path):
    try:
        y, sr = sf.read(path, always_2d=True, dtype='float32')
    except Exception:
        return None
    y = y.mean(1)
    peak = np.abs(y).max()
    if peak <= 0:
        return None
    y = y / peak
    y = y[np.argmax(np.abs(y) > 0.05):][:sr // 2]
    return hashlib.md5(np.round(y * 64).astype(np.int8).tobytes()).hexdigest()


def main(args):
    held_out = defaultdict(list)                    # fingerprint -> val/test files
    for split in ('val', 'test'):
        for f in OneShotAudioDataset(args.data_dir, split=split, types=TYPES).files:
            h = fingerprint(f)
            if h:
                held_out[h].append(f)
    train = OneShotAudioDataset(args.data_dir, split='train', types=TYPES).files
    dup = sorted(os.path.relpath(f, args.data_dir).replace(os.sep, '/')
                 for f in train if fingerprint(f) in held_out)
    with open(args.out, 'w') as f:
        f.write(''.join(d + '\n' for d in dup))
    print(f'{len(dup)} of {len(train)} training files duplicate a val/test file -> {args.out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Training files that duplicate val/test audio')
    p.add_argument('--data_dir', required=True)
    p.add_argument('-o', '--out', default='splits/exclude_train.txt')
    main(p.parse_args())
