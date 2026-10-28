"""
Write the fixed train/val/test split of a one-shot dataset as file lists.

The split itself is defined by oneshot_split() (a hash of each file's path
relative to the dataset root), so OneShotAudioDataset(split=...) already
reproduces it. These lists make it explicit and portable: systems outside this
codebase (DSP baseline, DrumBlender, diffusion) are evaluated on exactly the
files in splits/test.txt, and comparing the printed md5s between machines
(laptop vs Sherlock) confirms both see the same split.

Writes <out_dir>/{train,val,test}.txt, one relative path per line, sorted.

Usage:
    python make_splits.py --data_dir dataset/DrumOneShots
    python make_splits.py --data_dir /scratch/users/nlabate/datasets/drums_oneshots
"""

import argparse
import hashlib
import os
from collections import Counter

from dataset.datasets import OneShotAudioDataset
from drum_types import classify

DEFAULT_TYPES = ['kick', 'snare', 'clap', 'hat', 'cymbal', 'tom', 'perc']


def main(data_dir, types, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for split in ('train', 'val', 'test'):
        ds = OneShotAudioDataset(data_dir, split=split, types=types)
        rel = sorted(os.path.relpath(f, data_dir).replace(os.sep, '/') for f in ds.files)
        text = '\n'.join(rel) + '\n'
        path = os.path.join(out_dir, f'{split}.txt')
        with open(path, 'w') as f:
            f.write(text)
        counts = Counter(classify(os.path.join(data_dir, r), root=data_dir) for r in rel)
        md5 = hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
        print(f'{split:<6}{len(rel):>6} files  md5 {md5}  -> {path}')
        print('      ' + '  '.join(f'{t}={counts.get(t, 0)}' for t in types))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export the fixed one-shot split as file lists')
    parser.add_argument('--data_dir', required=True, help='dataset root the hash split is relative to')
    parser.add_argument('--types', nargs='+', default=DEFAULT_TYPES,
                        help='drum types to keep (must match the training configs)')
    parser.add_argument('-o', '--out_dir', default='splits')
    args = parser.parse_args()
    main(args.data_dir, args.types, args.out_dir)
