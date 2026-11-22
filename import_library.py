"""
One-time import of the cymbal and hi-hat one-shots from a sorted sample library
("Drum Samples by Instrument", with an index CSV) into the drum dataset, for the
metals-only runs.

Copies (never moves) the 'One-shot' rows of the Cymbal and Hi-hat folders into
<dataset>/Library - Drum Samples by Instrument/{Cymbals,Hats}/, renamed
'<type>_<pack>_<name>.wav' so drum_types.classify labels them from the name. Skips:
  - loops and unspecified forms (from the CSV's Form column),
  - files already in the dataset (same SHA-256),
  - a second hit inside the training window (import_freesound.second_hit),
  - non-audio and unreadable files.
Writes manifest.csv (source, pack, sha256, dest, status, reason) and
exclude_library.txt (dataset-relative paths, for runs that must not see it).
The library's licensing is not settled: keep it out of anything distributed.

  python import_library.py --src ~/Downloads/"Drum Samples by Instrument" --dry_run
  python import_library.py --src ~/Downloads/"Drum Samples by Instrument"
"""
import argparse
import csv
import glob
import hashlib
import os
import re
import shutil
from collections import Counter

import soundfile as sf

from drum_types import classify
from import_freesound import second_hit, clean

DEST_NAME = 'Library - Drum Samples by Instrument'
KEEP = {'Cymbal': ('cymbal', 'Cymbals', 'cymbal'), 'Hi-hat': ('hat', 'Hats', 'hihat')}
AUDIO = ('.wav', '.aif', '.aiff', '.flac')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main(args):
    dest = os.path.join(args.dest, DEST_NAME)
    if not args.dry_run and os.path.exists(dest):
        raise SystemExit(f'{dest} exists; remove it first to re-import')
    index = glob.glob(os.path.join(args.src, '*.csv'))
    assert index, 'no index CSV in --src'
    rows = list(csv.DictReader(open(index[0], encoding='utf-8-sig')))

    print('hashing the existing dataset (to skip duplicates)...')
    ours = set()
    for f in glob.glob(os.path.join(args.dest, '**', '*'), recursive=True):
        if f.lower().endswith(AUDIO) and DEST_NAME not in f:
            ours.add(sha256(f))

    out, used = [], set()
    for r in rows:
        if r['Instrument folder'] not in KEEP:
            continue
        t, sub, prefix = KEEP[r['Instrument folder']]
        src = os.path.join(args.src, r['Path in this ZIP'])
        row = dict(source=r['Path in this ZIP'], pack=r['Pack or library'], sha256=r['SHA-256'],
                   type=t, dest='', status='excluded', reason='')
        if r['Form'] != 'One-shot':
            row['reason'] = f"form: {r['Form']}"
        elif not src.lower().endswith(AUDIO) or not os.path.isfile(src):
            row['reason'] = 'not an audio file'
        elif r['SHA-256'] in ours:
            row['reason'] = 'already in the dataset'
        else:
            try:
                sf.info(src)
                hit = second_hit(src)
            except Exception as e:                      # unreadable / odd format
                hit, row['reason'] = None, f'unreadable: {e}'
            if row['reason']:
                pass
            elif hit is not None:
                row['reason'] = f'second hit at {hit:.2f} s'
            else:
                stem = re.sub(r'_+', '_', f"{prefix}_{clean(r['Pack or library'])}_{clean(os.path.splitext(r['Filename'])[0])}")
                stem = stem.lower()[:120]
                base, k = stem, 1
                while stem in used:
                    k += 1
                    stem = f'{base}-{k}'
                used.add(stem)
                rel = os.path.join(DEST_NAME, sub, stem + '.wav')
                got = classify(rel)
                if got != t:
                    row['reason'] = f'classified as {got}'
                else:
                    row.update(dest=rel, status='kept')
                    ours.add(r['SHA-256'])                 # duplicates inside the library
        out.append(row)

    kept = [r for r in out if r['status'] == 'kept']
    print(f'{len(kept)} kept of {len(out)}:', dict(Counter(r['type'] for r in kept)))
    for k, v in Counter(re.sub(r'at [\d.]+ s|: .*', '', r['reason']) for r in out if r['status'] != 'kept').most_common():
        print(f'  excluded {v:5d}: {k}')
    print('  kept by pack:', Counter(r['pack'] for r in kept).most_common(8))
    if args.dry_run:
        return
    for r in kept:
        o = os.path.join(args.dest, r['dest'])
        os.makedirs(os.path.dirname(o), exist_ok=True)
        x, sr = sf.read(os.path.join(args.src, r['source']), always_2d=True)
        sf.write(o, x, sr, subtype='PCM_24')           # aif / flac -> wav
    with open(os.path.join(dest, 'manifest.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    with open(args.exclude_out, 'w') as fh:
        fh.write('\n'.join(r['dest'] for r in kept) + '\n')
    print(f'copied to {dest}; exclude list {args.exclude_out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', required=True, help='the library folder (with its index CSV)')
    p.add_argument('--dest', default='dataset/DrumOneShots', help='dataset root')
    p.add_argument('--exclude_out', default='splits/exclude_library.txt')
    p.add_argument('--dry_run', action='store_true')
    main(p.parse_args())
