"""
Generate a Stable Audio 3 one-shot set matched to the test split's drum-type mix.

Runs in the Stable Audio 3 env (its own uv venv), not the training env:
    ~/Documents/CCRMA/tools/stable-audio-3/.venv/bin/python gen_stable_audio.py \
        --data_dir dataset/DrumOneShots -o eval/gen/sa3_small_sfx/raw

For every test hit of type t it generates one hit from a prompt for t (cycling
through a few wordings so the set isn't one prompt's stereotype). Output is the
model's raw stereo audio; eval_pipeline.py --gen_dir applies the shared
preprocessing (mono, onset alignment, peak norm, 1.49 s at 44.1 kHz) before FAD.

Writes <out>/<type>_<i>.wav, manifest.csv (file, type, prompt, seed) and
timing.json (seconds per hit in batches, and batch-1 latency, on --device).
"""

import argparse
import csv
import json
import os
import time
from collections import Counter

import numpy as np
import soundfile as sf
import torch
from stable_audio_3 import StableAudioModel

from drum_types import classify

PROMPTS = {
    'kick':   ['kick drum one-shot', 'punchy kick drum hit', '808 kick drum one-shot',
               'acoustic kick drum hit'],
    'snare':  ['snare drum one-shot', 'tight snare drum hit', 'acoustic snare drum hit',
               'electronic snare drum one-shot'],
    'clap':   ['hand clap one-shot', 'electronic clap one-shot', 'layered clap hit'],
    'hat':    ['closed hi-hat one-shot', 'open hi-hat one-shot', 'hi-hat hit'],
    'cymbal': ['crash cymbal hit', 'ride cymbal hit'],
    'tom':    ['tom drum hit', 'floor tom one-shot'],
    'perc':   ['percussion one-shot', 'shaker one-shot', 'conga hit', 'rimshot one-shot',
               'cowbell hit', 'tambourine hit', 'woodblock hit', 'bongo hit'],
}


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def main(args):
    with open(args.split_file) as f:
        rel = [l.rstrip('\n') for l in f if l.strip()]
    types = [classify(os.path.join(args.data_dir, r), root=args.data_dir) for r in rel]
    counts = Counter(types)
    print('type mix:', dict(counts))
    jobs = []                                   # (type, index, prompt)
    for t in sorted(counts):
        for i in range(counts[t]):
            jobs.append((t, i, PROMPTS[t][i % len(PROMPTS[t])]))

    model = StableAudioModel.from_pretrained(args.model, device=args.device)
    sr = model.model.sample_rate
    os.makedirs(args.out_dir, exist_ok=True)
    model.generate(prompt=jobs[0][2], duration=args.duration, seed=0)      # warm-up

    # batch-1 latency, the drum-machine setting
    lat = []
    for k in range(args.n_latency):
        sync(args.device); t0 = time.perf_counter()
        model.generate(prompt=jobs[k][2], duration=args.duration, seed=10_000 + k)
        sync(args.device); lat.append(time.perf_counter() - t0)

    rows, t_total = [], 0.0
    for b in range(0, len(jobs), args.batch_size):
        chunk = jobs[b:b + args.batch_size]
        seed = args.seed + b
        sync(args.device); t0 = time.perf_counter()
        audio = model.generate(prompt=[j[2] for j in chunk], duration=args.duration,
                               batch_size=len(chunk), seed=seed)
        sync(args.device); t_total += time.perf_counter() - t0
        audio = audio.float().cpu().numpy()
        if audio.ndim == 2:
            audio = audio[None]
        for (t, i, p), a in zip(chunk, audio):
            name = f'{t}_{i:03d}.wav'
            sf.write(os.path.join(args.out_dir, name), a.T, sr, subtype='FLOAT')
            rows.append({'file': name, 'type': t, 'prompt': p, 'seed': seed})
        print(f'  {min(b + args.batch_size, len(jobs))}/{len(jobs)}', end='\r', flush=True)
    print()

    with open(os.path.join(args.out_dir, 'manifest.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'type', 'prompt', 'seed'])
        w.writeheader(); w.writerows(rows)
    timing = {'model': args.model, 'device': args.device, 'duration_s': args.duration,
              'steps': 8, 'n': len(jobs), 'batch_size': args.batch_size,
              'sec_per_hit_batched': t_total / len(jobs),
              'sec_per_hit_batch1_median': float(np.median(lat)), 'sample_rate': sr}
    with open(os.path.join(args.out_dir, 'timing.json'), 'w') as f:
        json.dump(timing, f, indent=1)
    print(json.dumps(timing, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Stable Audio 3 one-shots matched to the test type mix')
    p.add_argument('--data_dir', required=True)
    p.add_argument('--split_file', default='splits/test.txt')
    p.add_argument('-o', '--out_dir', default='eval/gen/sa3_small_sfx/raw')
    p.add_argument('--model', default='small-sfx')
    p.add_argument('--device', default='mps')
    p.add_argument('--duration', type=float, default=2.0)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--n_latency', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    main(p.parse_args())
