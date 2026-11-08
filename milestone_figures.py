"""
Figure for the CS230 milestone: (a) per-dimension KL on the test split, sorted,
for a few checkpoints (latent usage / the 44k collapse); (b) the Tone knob:
mean change in measured brightness vs knob setting, per model, on test hits.

Usage:
    python milestone_figures.py --data_dir dataset/DrumOneShots \
        -o ../cs230-milestone/figures/latent_knobs.pdf
"""

import argparse
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from control_sweeps import desc_array, encode_decode, decode
from drum_engine import frame_activity
from eval_pipeline import load_checkpoint, read_split, make_dataset

RUNS = 'runs/vae_drums/models'
MODELS = {   # label: (checkpoint, sweep directions, Tone knob key)
    'Prototype (22k)': ('saved/models/DrumOneShots/checkpoint-epoch1500.pth',
                        'sweeps/prototype/directions.pt', 'active/brightness/pca'),
    '22k': (f'{RUNS}/Drums_22k/1027_154437/checkpoint-epoch1500.pth', None, None),
    '44k': (f'{RUNS}/Drums_44k/1027_125950/checkpoint-epoch1500.pth',
            'sweeps/44k_ep1500/directions.pt', 'active/brightness/kl'),
}


def main(args):
    rel_all = read_split(args.split_file)
    rel_knob = random.Random(args.seed).sample(rel_all, args.n_knob)
    fig, (ax_kl, ax_k) = plt.subplots(1, 2, figsize=(7.0, 2.1))
    for c, (label, (ck, dirs, key)) in enumerate(MODELS.items()):
        color = f'C{c}'
        model, cfg, _ = load_checkpoint(ck)
        model.to(args.device)
        dl = cfg['data_loader']['args']
        sr, chunk = dl.get('sr', 22050), dl.get('chunk_size', 110080)
        pre = dict(onset_thresh=dl.get('onset_thresh', 0.02), pre_pad=dl.get('pre_pad', 256),
                   peak_norm=dl.get('peak_norm', True))
        ds = make_dataset(args.data_dir, rel_all, sr, chunk, **pre)
        mu, lv, _ = encode_decode(model, ds, list(range(len(ds))), args.device)
        kl = (0.5 * (mu.pow(2) + lv.exp() - 1 - lv)).mean((0, 2)).numpy()
        ax_kl.plot(np.arange(1, len(kl) + 1), np.sort(kl)[::-1], 'o-', ms=2.5, lw=1, color=color, label=label)
        print(f'{label}: KL total {kl.sum():.2f}, dims > 0.2: {(kl > 0.2).sum()}')
        if dirs is None:
            continue
        d = torch.load(dirs, map_location='cpu', weights_only=False)
        knob = d['directions'][key]
        v, std, grid = knob['vector'], knob['std'], d['grid']
        idx = [rel_all.index(r) for r in rel_knob]
        mu_k = mu[idx]
        y0 = decode(model, mu_k, args.device)
        m = frame_activity(y0, mu_k.shape[-1])
        b0 = desc_array(y0.numpy(), sr)[:, 0]
        means = []
        for s in grid:
            y = y0 if s == 0 else decode(model, mu_k + s * v[None, :, None] * m[:, None, :], args.device)
            means.append(np.nanmean((desc_array(y.numpy(), sr)[:, 0] - b0) / std))
        ax_k.plot(grid, means, 'o-', ms=3, color=color, label=f'{label}, {key.split("/")[-1]}')

    ax_kl.axhline(0.15, color='k', ls=':', lw=0.8)
    ax_kl.text(32, 0.17, 'free bits', ha='right', va='bottom', fontsize=7)
    ax_kl.set_yscale('log')
    ax_kl.set_xlabel('latent dimension (sorted)')
    ax_kl.set_ylabel('mean KL per frame (nats)')
    ax_kl.set_title('(a) latent usage, test split', fontsize=9)
    ax_kl.legend(fontsize=7)
    ax_k.plot(grid, grid, 'k:', lw=0.8, label='target')
    ax_k.set_xlabel('Tone knob $s$ (descriptor std)')
    ax_k.set_ylabel('$\\Delta$ brightness (std)')
    ax_k.set_title(f'(b) Tone knob, {args.n_knob} test hits', fontsize=9)
    ax_k.legend(fontsize=7)
    for ax in (ax_kl, ax_k):
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out)
    print('saved', args.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Milestone figure: latent usage + Tone knob')
    p.add_argument('--data_dir', default='dataset/DrumOneShots')
    p.add_argument('--split_file', default='splits/test.txt')
    p.add_argument('-o', '--out', default='../cs230-milestone/figures/latent_knobs.pdf')
    p.add_argument('--n_knob', type=int, default=64)
    p.add_argument('-d', '--device', default='cpu')
    p.add_argument('--seed', type=int, default=0)
    main(p.parse_args())
