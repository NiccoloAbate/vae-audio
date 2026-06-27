#!/usr/bin/env python
"""Pre-extract fixed-length raw-audio chunks from a directory of audio files
into a single .npy array (+ _meta.pkl), for use with NumpyChunkDataLoader.

This mirrors the fast-loader workflow used by the proven cello run, but lets you
re-extract at a new sample rate / chunk size (e.g. for the 48 kHz PQMF config).

Example (48 kHz cello, 32768-sample chunks for config_raw_cello_pqmf_48k.json):

    python build_chunks.py \
        --src /home/niccoloabate/Datasets/CCBachCello \
        --out /home/niccoloabate/Datasets/CCBachCello_48k_32768.npy \
        --sr 48000 --chunk_size 32768

chunk_size must be divisible by the model's total downsample factor
(n_bands * product(strides)); for the PQMF config that is 16 * 32 = 512,
and 32768 = 64 * 512.
"""
import argparse
import os
import pickle

import numpy as np
import torchaudio


def find_audio(src, exts=('.wav', '.flac', '.aif', '.aiff')):
    files = []
    for root, _, fnames in os.walk(src):
        for fn in sorted(fnames):
            if fn.lower().endswith(exts):
                files.append(os.path.join(root, fn))
    return sorted(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='dir of source audio files (recursive)')
    ap.add_argument('--out', required=True, help='output .npy path')
    ap.add_argument('--sr', type=int, default=48000)
    ap.add_argument('--chunk_size', type=int, default=32768)
    ap.add_argument('--label', default='cello')
    ap.add_argument('--drop_silence', type=float, default=0.0,
                    help='drop chunks whose peak |amp| < this (0 = keep all)')
    args = ap.parse_args()

    files = find_audio(args.src)
    if not files:
        raise SystemExit(f'No audio found under {args.src}')
    print(f'Found {len(files)} files; resampling to {args.sr} Hz, '
          f'chunk_size={args.chunk_size}')

    chunks, items = [], []
    for fi, f in enumerate(files):
        y, sr = torchaudio.load(f)
        if y.shape[0] > 1:
            y = y.mean(0, keepdim=True)
        if sr != args.sr:
            y = torchaudio.functional.resample(y, sr, args.sr)
        y = y.squeeze(0).numpy().astype(np.float32)

        n = len(y) // args.chunk_size
        for c in range(n):
            seg = y[c * args.chunk_size:(c + 1) * args.chunk_size]
            if args.drop_silence > 0 and np.abs(seg).max() < args.drop_silence:
                continue
            chunks.append(seg)
            items.append((args.label, f, c * args.chunk_size, args.sr))

        if (fi + 1) % 50 == 0:
            print(f'  {fi + 1}/{len(files)} files -> {len(chunks)} chunks')

    if not chunks:
        raise SystemExit('No chunks produced — is chunk_size larger than every file?')

    data = np.stack(chunks).astype(np.float32)
    np.save(args.out, data)
    meta_path = args.out.replace('.npy', '_meta.pkl')
    with open(meta_path, 'wb') as fh:
        pickle.dump({'items': items, 'sr': args.sr,
                     'chunk_size': args.chunk_size}, fh)
    print(f'Wrote {data.shape} -> {args.out}')
    print(f'Wrote metadata -> {meta_path}')


if __name__ == '__main__':
    main()
