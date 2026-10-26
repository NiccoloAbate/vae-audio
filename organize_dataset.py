"""
Build a by-type view of a one-shot dataset: <dst>/<type>/<original/relpath>.

Runs orthogonally to training/synthesis — it never touches the source. Files
are copied by default (a self-contained organized dataset, per request); use
--symlink for a zero-copy view. The relative path under the source is preserved
inside each type folder, so pack/subfolder provenance is kept and there are no
name collisions across packs.

Nothing is deleted — ambient/abstract hits land under 'fx' / 'texture', and
loops, fills and non-drum instruments land under 'exclude'. The labels are
also written to <dst>/labels.csv (relpath,type), which is what data splits
should be built from.

Usage:
    python organize_dataset.py --src dataset/DrumOneShots --dst dataset/DrumOneShots_bytype --dry_run
    python organize_dataset.py --src dataset/DrumOneShots --dst dataset/DrumOneShots_bytype           # copy
    python organize_dataset.py --src ... --dst ... --symlink                                          # link
"""
import argparse
import collections
import csv
import os
import shutil

from drum_types import classify, TYPES

EXTS = ('.wav', '.aif', '.aiff', '.flac')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--symlink', action='store_true', help='symlink instead of copy')
    ap.add_argument('--dry_run', action='store_true', help='report counts, write nothing')
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    counts = collections.Counter()
    labels = []
    n_written = 0

    for root, _, fnames in os.walk(src):
        for fn in sorted(fnames):
            if not fn.lower().endswith(EXTS):
                continue
            spath = os.path.join(root, fn)
            t = classify(spath, root=src)
            rel = os.path.relpath(spath, src)                 # pack/.../file.wav
            counts[t] += 1
            labels.append((rel, t))
            if args.dry_run:
                continue
            dpath = os.path.join(args.dst, t, rel)
            os.makedirs(os.path.dirname(dpath), exist_ok=True)
            if os.path.exists(dpath) or os.path.islink(dpath):
                continue
            if args.symlink:
                os.symlink(spath, dpath)
            else:
                shutil.copy2(spath, dpath)
            n_written += 1

    if not args.dry_run:
        os.makedirs(args.dst, exist_ok=True)
        with open(os.path.join(args.dst, 'labels.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['relpath', 'type'])
            w.writerows(sorted(labels))

    total = sum(counts.values())
    print(f'{"TYPE":>10} {"COUNT":>7}  {"%":>5}')
    for t in TYPES:
        n = counts.get(t, 0)
        if n:
            print(f'{t:>10} {n:>7}  {100*n/total:4.1f}')
    print(f'{"TOTAL":>10} {total:>7}')
    if args.dry_run:
        print(f'\n[dry run] would {"symlink" if args.symlink else "copy"} '
              f'{total} files into {args.dst}/<type>/<relpath>')
    else:
        print(f'\n{"Linked" if args.symlink else "Copied"} {n_written} files -> {args.dst}/')


if __name__ == '__main__':
    main()
