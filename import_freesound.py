"""
One-time import of two Freesound packs into the drum dataset, with clean names.

Copies (never moves) the kept files into one new folder per pack under the
dataset root, renamed '<type>_<description>_<n>.wav' so drum_types.classify
labels them from the name alone. Files are excluded when:
  - no keep-rule matches (FX, rhythms, sizzle chains, drumsticks, ...),
  - the licence is non-commercial (CC BY-NC),
  - a second hit starts inside the training window (1.49 s after the first onset).
Every source file gets a row in <dest>/<pack>/manifest.csv (kept or not, with
the reason), with its Freesound id, URL, author and licence: CC BY needs
attribution, so the manifest is also the credit list for the report.

  python import_freesound.py --dry_run        # report only
  python import_freesound.py                  # copy into dataset/DrumOneShots
"""
import argparse
import csv
import os
import re
import shutil
from collections import Counter

import numpy as np
import soundfile as sf

from drum_types import classify

DOWNLOADS = os.path.expanduser('~/Downloads')

# (source folder, destination pack folder, author)
PACKS = [
    ('43343__cvltiv8r__custom-cymbals-vintage-percussion-pack',
     'Freesound - CVLTIV8R Custom Cymbals', 'CVLTIV8R'),
    ('16665__theriavirra__hi-hats-charlestons-drumsticks-snares-and-cymbals-that-sounds-good',
     'Freesound - Theriavirra Cymbals and Snares', 'Theriavirra'),
]

SUBDIR = {'cymbal': 'Cymbals', 'hat': 'Hats', 'snare': 'Snares', 'tom': 'Toms', 'kick': 'Kicks'}

# Keep-rules on the core name (Freesound id/user prefix and '.wav' removed,
# lower case). Each maps to (type, new stem built from the match groups).
# Anything unmatched is excluded.
RULES = [
    # CVLTIV8R
    (r'zildjian-16-inch-crash-(\d+)',               'cymbal', 'crash_zildjian-16in_{0}'),
    (r'sabian-15-inch-custom-crash-(\d+)',          'cymbal', 'crash_sabian-15in_{0}'),
    (r'paiste-22-inch-custom-ride-(\d+)',           'cymbal', 'ride_paiste-22in_{0}'),
    (r'paiste-custom-22inch-ride-delicate-sharp-oneshot-?(\d*)', 'cymbal', 'ride_paiste-22in-sharp_{0}'),
    (r'cymbal-grab-stop-mute-(\d+)',                'cymbal', 'cymbal_choke_{0}'),
    (r'cymbal-muted-(\d+)',                         'cymbal', 'cymbal_muted_{0}'),
    (r'random-cymbal-(\d+)',                        'cymbal', 'cymbal_misc_{0}'),
    (r'vintage-custom-14-inch-hi-hat-(\d+)',        'hat',    'hihat_vintage-14in_{0}'),
    (r'(high|mid-high|mid-low|low-floor)-tom-sonor-force-3007-(\d+)', 'tom', 'tom_sonor-{0}_{1}'),
    (r'kick-sonor-force-3007-(\d+)',                'kick',   'kick_sonor_{0}'),
    (r'rim-hit-perc-oneshot-(\d+)',                 'snare',  'snare_rim_{0}'),
    # Theriavirra ('double' and 'tri(plets)' variants are several strikes)
    (r'(\d+[a-z]*)_hat_((?!.*(?:double|tri)).+)_cymbals__snares',   'hat',    'hihat_{1}_{0}'),
    (r'(\d+[a-z]*)_ride_((?!.*(?:double|tri)).+)_cymbals__snares',  'cymbal', 'ride_{1}_{0}'),
    (r'(\d+[a-z]*)_crash_((?!.*(?:double|tri)).+)_cymbals__snares', 'cymbal', 'crash_{1}_{0}'),
    (r'(\d+[a-z]*)_snare_((?!.*(?:double|tri)).+)_cymbals__snares', 'snare',  'snare_{1}_{0}'),
    (r'(\d+[a-z]*)_tom_+(?:-_)?(.*?)_?44100',       'tom',    'tom_{1}_{0}'),
    (r'snare_sample_-_2015_-_(.+)',                 'snare',  'snare_2015_{0}'),
    (r'snare_sample_-_chuzhbinov_-_(.+)',           'snare',  'snare_chuzhbinov_{0}'),
    (r'bass_drumevans(_softer)?',                   'kick',   'kick_evans{0}'),
]
RULES = [(re.compile(p + r'$'), t, s) for p, t, s in RULES]

WINDOW_S = 65536 / 44100          # training window after the first onset


def clean(s):
    """'low-floor' / '_-_' / '' -> tidy dash-joined words."""
    s = re.sub(r'[\s_\-]+', '-', (s or '').strip('_- ')).strip('-')
    return s


def parse_readme(path):
    """{freesound id: (url, licence)} from the pack's _readme_and_license.txt."""
    out, cur = {}, None
    for line in open(path, encoding='utf-8', errors='replace'):
        m = re.match(r'\s*\*\s+(\d+)__', line)
        if m:
            cur = m.group(1); out[cur] = ['', '']
        elif cur and '* url:' in line:
            out[cur][0] = line.split('url:', 1)[1].strip()
        elif cur and '* license:' in line:
            out[cur][1] = line.split('license:', 1)[1].strip()
    return out


def second_hit(path, rise_db=8.0, near_db=12.0, attack_s=0.15, guard_s=0.05):
    """Seconds from the first onset to a second strike inside the training
    window, or None. Envelope = 20 ms moving RMS on a 5 ms hop. The first
    strike is the loudest point within attack_s of the onset (cymbals bloom);
    after guard_s more, a strike = the envelope rising > rise_db above its
    running minimum, to within near_db of the file's peak."""
    y, sr = sf.read(path, always_2d=True)
    y = y.mean(1)
    hop, win = max(1, int(0.005 * sr)), int(0.02 * sr)
    p = np.convolve(y ** 2, np.ones(win) / win, mode='same')[::hop]
    db = 10 * np.log10(p / p.max() + 1e-12)
    start = int(np.argmax(db > -30))                         # first onset
    pk = start + int(np.argmax(db[start:start + int(attack_s * sr / hop) + 1]))
    stop = min(len(db), start + int(WINDOW_S * sr / hop))
    lo = np.inf
    for i in range(pk + int(guard_s * sr / hop), stop):
        lo = min(lo, db[i])
        if db[i] - lo > rise_db and db[i] > -near_db:
            return (i - start) * hop / sr
    return None


def main(args):
    for src_name, dest_name, author in PACKS:
        src = os.path.join(DOWNLOADS, src_name)
        dest = os.path.join(args.dest, dest_name)
        if not args.dry_run and os.path.exists(dest):
            raise SystemExit(f'{dest} exists; remove it first to re-import')
        lic = parse_readme(os.path.join(src, '_readme_and_license.txt'))
        rows, used = [], set()
        for f in sorted(os.listdir(src)):
            if not f.lower().endswith('.wav'):
                continue
            fid = f.split('__', 1)[0]
            core = re.sub(r'^\d+__[^_]+__', '', f)[:-4].lower()
            url, licence = lic.get(fid, ('', 'unknown'))
            row = dict(source=f, freesound_id=fid, url=url, author=author, license=licence,
                       type='', dest='', status='excluded', reason='')
            rule = next(((t, s, m) for p, t, s in RULES if (m := p.match(core))), None)
            if rule is None:
                row['reason'] = 'no keep-rule (fx / rhythm / sizzle / drumsticks / other)'
            elif 'noncommercial' in licence.replace(' ', '').lower():
                row['reason'] = 'licence ' + licence
            else:
                t, stem, m = rule
                hit = second_hit(os.path.join(src, f))
                if hit is not None:
                    row['reason'] = f'second hit at {hit:.2f} s'
                else:
                    stem = stem.format(*[clean(g) for g in m.groups()]).strip('_')
                    stem = re.sub(r'_+', '_', stem)
                    base, k = stem, 1
                    while stem in used:
                        k += 1; stem = f'{base}-{k}'
                    used.add(stem)
                    rel = os.path.join(dest_name, SUBDIR[t], stem + '.wav')
                    got = classify(rel)
                    assert got == t, f'{rel}: drum_types says {got}, expected {t}'
                    row.update(type=t, dest=rel, status='kept')
            rows.append(row)

        kept = [r for r in rows if r['status'] == 'kept']
        print(f'\n=== {dest_name}: {len(kept)} kept of {len(rows)}')
        print('  kept by type:', dict(Counter(r['type'] for r in kept)))
        reasons = Counter(re.sub(r'at [\d.]+ s', 'in window', r['reason'])
                          for r in rows if r['status'] != 'kept')
        for k, v in reasons.most_common():
            print(f'  excluded {v:3d}: {k}')
        if args.verbose:
            for r in rows:
                print(f"  {r['status'][:4]} {r['source'][:70]:70s} -> {r['dest'] or r['reason']}")
        if args.dry_run:
            continue
        for r in kept:
            out = os.path.join(args.dest, r['dest'])
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.copy2(os.path.join(src, r['source']), out)
        with open(os.path.join(dest, 'manifest.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f'  copied to {dest}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dest', default='dataset/DrumOneShots', help='dataset root')
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true', help='one line per source file')
    main(p.parse_args())
