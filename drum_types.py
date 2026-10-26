"""
Heuristic drum-type classifier from a file path.

One-shot libraries are organised by *pack*, not by drum type, and carry no type
metadata — but the type is almost always in the file/folder name ("Kick",
"Snr", "OH", "Clap", "Sub Impact", ...). This maps a path to a canonical type
by keyword.

The path is read from the most specific component to the least: file name
first, then each parent folder, then the pack name. The first component that
names a type decides it, so "Hats and Percs/Glitch Percs/x.wav" is perc, not
hat. Within one component:
  * drum words beat fx/texture words ("Clap Foley 3" is a clap);
  * a file name naming two drum types takes the first in TIE_ORDER;
  * a folder naming two ("Snares and Claps") defers to the file name, and if
    the file name is silent, prefers perc, then TIE_ORDER.

Folders that hold loops, fills or non-drum instruments are labelled 'exclude'
(kept on disk, left out of training). Anything unmatched is 'unsorted'.

Shared by build_latent_bank.py (representative-spread selection) and
organize_dataset.py (the by-type folder view and labels.csv).
"""
import os
import re

# (canonical_type, keyword regexes). Keywords are matched against a
# lower-cased component with '_', '-', '.', digits etc. turned into spaces.
# A leading \b stops 'hat' matching 'what' and 'tom' matching 'bottom'.
_RULES = [
    ('kick',    [r'kick', r'\bkck', r'\bbd\b', r'bassdrum', r'bass drum', r'808']),
    ('snare',   [r'snare', r'\bsnr', r'rimshot', r'rim shot', r'\brim\b', r'\bbrush']),
    ('clap',    [r'clap', r'\bclp']),
    ('hat',     [r'hihat', r'hi hat', r'\bhats?\b', r'\bhat\b', r'\bhh\b', r'\bohat',
                 r'\bchat\b', r'open hat', r'closed hat', r'high hat']),
    ('cymbal',  [r'\bcrash', r'\bride\b', r'cymbal', r'\bchina\b', r'\bsplash', r'\bgong']),
    ('tom',     [r'\btoms?\b']),
    ('perc',    [r'perc', r'conga', r'bongo', r'\bbngo', r'tabla', r'\btab\b', r'djembe',
                 r'shaker', r'\btamb', r'\btmb\b', r'\btam\b', r'tumbadora', r'\btim\b',
                 r'timbal', r'cowbell', r'clave', r'woodblock', r'wood block', r'\bblocks?\b',
                 r'triangle', r'agogo', r'cabasa', r'caxixi', r'guiro', r'cajon', r'darbuka',
                 r'doumbek', r'\bdoum\b', r'dhol', r'bendir', r'\briqq', r'frame drum',
                 r'\bbells?\b', r'chim', r'\btribal']),
    ('fx',      [r'impact', r'\bboom', r'braam', r'riser', r'downer', r'whoosh', r'sweep',
                 r'\bhits?\b', r'\bstab', r'\bzap', r'glitch', r'noise', r'\bsub\b', r'\bs?fx\b',
                 r'foley', r'laser', r'\bwub']),
    ('texture', [r'ambient', r'atmos', r'drone', r'\bpad', r'scrape', r'texture', r'soundscape',
                 r'swell', r'\bfield', r'\bwind\b', r'water', r'granular']),
]
_COMPILED = [(t, [re.compile(k) for k in kws]) for t, kws in _RULES]
_DRUM = {'kick', 'snare', 'clap', 'hat', 'cymbal', 'tom', 'perc'}
TIE_ORDER = ['snare', 'kick', 'clap', 'hat', 'tom', 'cymbal', 'perc', 'fx', 'texture']

# Not one-shot drums: loops, fills, rolls, melodic instruments, impulse
# responses. Matched against folder names exactly (lower-cased), plus the
# word 'loop' anywhere in the file name.
EXCLUDE_DIRS = {
    'bass loops', 'bass one shots', 'brass', 'drum fills', 'drum loops (stem seperated)',
    'hat loops', 'lead loops', 'lead one shots', 'pecussion loops', 'percussion loops',
    'snare rolls', 'vocal loops', 'impulse responses', 'midi', 'serum presets',
}
EXCLUDE_FILES = {'soundpacks.com'}    # pack watermark / promo file
_LOOP = re.compile(r'(?<!b)loops?\b')   # 'PercLoop1', not 'Basic Bloop'

# Hand corrections where the name misleads, keyed by path relative to the
# dataset root (no extension).
OVERRIDES = {
    'Glitch With Friends/vol 1/Drums/Kicks/Man From Sol - Boomf': 'kick',
}

# Canonical types in a stable display order ('unsorted' and 'exclude' last).
TYPES = [r[0] for r in _RULES] + ['unsorted', 'exclude']


def _norm(s):
    s = re.sub(r'\.(wav|aiff?|flac)$', '', s.lower())
    s = re.sub(r'(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])', ' ', s)    # 'tom1' -> 'tom 1'
    return ' ' + re.sub(r'[^a-z0-9]+', ' ', s).strip() + ' '


def _types_in(component):
    s = _norm(component)
    found = [t for t, pats in _COMPILED if any(p.search(s) for p in pats)]
    if any(t in _DRUM for t in found):
        found = [t for t in found if t in _DRUM]
    return found


def _components(path, root=None):
    if root is not None:
        path = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    parts = [p for p in path.replace('\\', '/').split('/') if p]
    return parts[::-1]                      # file name first, pack name last


def classify(path, root=None):
    """Return a canonical type for a file path, 'exclude', or 'unsorted'.

    Pass `root` (the dataset directory) so that folders above the dataset do
    not take part in matching.
    """
    comps = _components(path, root)
    key = re.sub(r'\.(wav|aiff?|flac)$', '', '/'.join(comps[::-1]), flags=re.I)
    for k, t in OVERRIDES.items():
        if key.endswith(k):
            return t
    stem = re.sub(r'\.(wav|aiff?|flac)$', '', comps[0].lower())
    if stem in EXCLUDE_FILES or _LOOP.search(_norm(stem)):
        return 'exclude'
    if any(c.lower() in EXCLUDE_DIRS for c in comps[1:]):
        return 'exclude'

    pending = None                          # candidates from an ambiguous folder
    for i, comp in enumerate(comps):
        found = _types_in(comp)
        if not found:
            continue
        if len(found) == 1:
            return found[0]
        if i == 0:                          # file name naming two drum types
            return min(found, key=TIE_ORDER.index)
        if pending is None:                 # folder naming two: file name was silent
            pending = found
            break
    if pending:
        return 'perc' if 'perc' in pending else min(pending, key=TIE_ORDER.index)
    return 'unsorted'


if __name__ == '__main__':
    import sys
    for p in sys.argv[1:]:
        print(f'{classify(p):>10}  {p}')
