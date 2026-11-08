"""
Generate a one-shot set matched to the test split's drum-type mix, for unpaired
(generation) FAD. Nothing here touches test audio: every hit comes from the
training split.

Modes:
    interp   VAE: lerp between the latents of two training hits of the same
             type, at t ~ U(0.25, 0.75) so neither endpoint dominates
    perturb  VAE: one training hit's latent + N(0, sigma) noise on the frames
             where the hit sounds (silent frames left alone, as for the knobs)
    real     no model: the training hits themselves (the "sample library"
             reference: the realism of just picking real samples)

Writes <out>/<type>_<i>.wav at the model's native rate (or the file's, for
real), manifest.csv, and timing.json (batch-1 decode time on --device, VAE
modes). Score with: eval_pipeline.py --gen_dir <out> --name <name>.

Usage:
    python generate_set.py --mode interp -r saved/models/DrumOneShots/checkpoint-epoch1500.pth \
        --data_dir dataset/DrumOneShots -o eval/gen/prototype_interp/raw
    python generate_set.py --mode real --data_dir dataset/DrumOneShots -o eval/gen/real_train/raw
"""

import argparse
import csv
import json
import os
import random
import shutil
import time
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf
import torch

from drum_types import classify
from drum_engine import frame_activity
from eval_pipeline import load_checkpoint, read_split, make_dataset


def type_mix(data_dir, split_file):
    return Counter(classify(os.path.join(data_dir, r), root=data_dir) for r in read_split(split_file))


def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def main(args):
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    counts = type_mix(args.data_dir, args.split_file)
    train = read_split(args.train_split)
    by_type = defaultdict(list)
    for r in train:
        by_type[classify(os.path.join(args.data_dir, r), root=args.data_dir)].append(r)
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []

    if args.mode == 'real':
        for t in sorted(counts):
            for i, r in enumerate(rng.sample(by_type[t], counts[t])):
                name = f'{t}_{i:03d}{os.path.splitext(r)[1]}'
                shutil.copy(os.path.join(args.data_dir, r), os.path.join(args.out_dir, name))
                rows.append({'file': name, 'type': t, 'sources': r})
        timing = None
    else:
        device = torch.device(args.device)
        model, cfg, epoch = load_checkpoint(args.resume)
        model.to(device)
        dl = cfg['data_loader']['args']
        sr, chunk = dl.get('sr', 22050), dl.get('chunk_size', 110080)
        pre = dict(onset_thresh=dl.get('onset_thresh', 0.02), pre_pad=dl.get('pre_pad', 256),
                   peak_norm=dl.get('peak_norm', True))
        lat = []
        for t in sorted(counts):
            for i in range(counts[t]):
                srcs = rng.sample(by_type[t], 2 if args.mode == 'interp' else 1)
                ds = make_dataset(args.data_dir, srcs, sr, chunk, **pre)
                x = torch.stack([ds[k][2] for k in range(len(srcs))]).to(device)
                with torch.no_grad():
                    mu = model.encode(x)[0]
                    if args.mode == 'interp':
                        a = rng.uniform(0.25, 0.75)
                        z = (1 - a) * mu[0:1] + a * mu[1:2]
                        info = f'{srcs[0]} | {srcs[1]} | t={a:.3f}'
                    else:
                        y0 = model.decode(mu).squeeze(1).cpu()
                        m = frame_activity(y0, mu.shape[-1]).to(device)
                        z = mu + args.sigma * torch.randn_like(mu) * m[:, None, :]
                        info = f'{srcs[0]} | sigma={args.sigma}'
                    sync(device); t0 = time.perf_counter()
                    y = model.decode(z).squeeze().cpu().numpy()
                    sync(device); lat.append(time.perf_counter() - t0)
                name = f'{t}_{i:03d}.wav'
                sf.write(os.path.join(args.out_dir, name), y, sr, subtype='FLOAT')
                rows.append({'file': name, 'type': t, 'sources': info})
        timing = {'checkpoint': args.resume, 'epoch': epoch, 'device': args.device,
                  'mode': args.mode, 'n': len(rows), 'sample_rate': sr,
                  'chunk_s': chunk / sr, 'decode_sec_per_hit_median': float(np.median(lat[3:]))}
        with open(os.path.join(args.out_dir, 'timing.json'), 'w') as f:
            json.dump(timing, f, indent=1)

    with open(os.path.join(args.out_dir, 'manifest.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'type', 'sources'])
        w.writeheader(); w.writerows(rows)
    print(f'{args.mode}: wrote {len(rows)} hits -> {args.out_dir}' +
          (f'  (decode {timing["decode_sec_per_hit_median"]*1000:.1f} ms/hit on {args.device})'
           if timing else ''))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Matched generation set for unpaired FAD')
    p.add_argument('--mode', required=True, choices=['interp', 'perturb', 'real'])
    p.add_argument('-r', '--resume', help='checkpoint (interp / perturb)')
    p.add_argument('--data_dir', required=True)
    p.add_argument('--split_file', default='splits/test.txt', help='type mix to match')
    p.add_argument('--train_split', default='splits/train.txt', help='source hits')
    p.add_argument('-o', '--out_dir', required=True)
    p.add_argument('--sigma', type=float, default=0.5)
    p.add_argument('-d', '--device', default='cpu')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.mode != 'real' and not args.resume:
        p.error('--resume is required for interp / perturb')
    main(args)
