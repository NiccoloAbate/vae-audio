"""
Time-varying controls for decay and attack, which constant offsets on every
sounding frame (control_sweeps.py) fail to steer. Decay is a property of the
tail and attack of the first frames, but the constant offsets are fit on, and
applied to, the latent averaged over the whole hit.

Per hit, the pivot is c = peak frame + --head (peak = loudest frame of the
hit's reconstruction). Head frames are t <= c, tail frames t > c; both are
restricted to sounding frames as in control_sweeps.py.

    static   the control_sweeps baseline: fit on, and offset, all sounding frames
    tail     fit decay on the latent averaged over tail frames; offset the tail only
    ramp     the tail direction, offset weighted by a ramp that rises from 0 after
             the pivot (normalised to mean 1 over the tail, so the calibration of
             'tail' still holds): pushes later frames further
    stretch  no fit: resample the latent after the pivot by r = 10^(s * std), so s
             promises s std of log10 decay time if the decoder maps a stretched
             latent to a stretched tail; past the end it fades into the fit set's
             silence latent (drum_engine.stretch_tail, as in the app). 'stretch' uses the common
             s grid (+-3 std is up to 35x here); 'stretch_x' the --stretch_factors
             a slider offers (x0.25 .. x4), still reported in std units
    head     fit attack on the latent averaged over head frames; offset the head only

Fitted directions use control_sweeps.fit_directions (kinds: ridge in the top
--pcs PCs 'pca', the KL-active dims 'kl', all dims 'ridge'). Metrics are
control_sweeps.sweep_metrics, in descriptor std on the fit set, so the table
compares directly with summary.txt there.

Outputs under sweeps/<name>/time/: summary.txt, summary.json, sweeps.png,
audio/<control>/<hit>.wav (one hit swept from -max to +max s).

Usage:
    python time_controls.py -r runs/.../checkpoint-epoch1500.pth \
        --data_dir dataset/DrumOneShots --name 44k_b005_ep1500 -d mps
"""

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from control_sweeps import KNOBS, desc_array, encode_decode, decode, pool, fit_directions, \
    sweep_metrics
from drum_engine import frame_activity, frame_rms, stretch_tail, silence_latent
from eval_pipeline import load_checkpoint, read_split, make_dataset
from evaluate_raw import get_device


def head_tail(y, T, head):
    """Sounding mask, pivot frame and head / tail masks, each (N, T) (pivot (N,))."""
    act = frame_activity(y, T)
    pivot = frame_rms(y, T).argmax(1) + head
    t = torch.arange(T)[None, :]
    is_head = (t <= pivot[:, None]).float()
    return act, pivot, act * is_head, act * (1 - is_head)


def ramp(tail):
    """Tail mask -> weights rising linearly over each hit's tail, mean 1 over it."""
    r = tail.cumsum(1) * tail
    return r / (r.sum(1, keepdim=True) / tail.sum(1, keepdim=True).clamp_min(1)).clamp_min(1e-8)


def main(args):
    device = torch.device(args.device) if args.device else get_device()
    model, cfg, epoch = load_checkpoint(args.resume)
    model.to(device)
    dl = cfg['data_loader']['args']
    sr, chunk = dl.get('sr', 22050), dl.get('chunk_size', 110080)
    pre = dict(onset_thresh=dl.get('onset_thresh', 0.02), pre_pad=dl.get('pre_pad', 256),
               peak_norm=dl.get('peak_norm', True))
    out_dir = args.out_dir or os.path.join('sweeps', args.name, 'time')
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(args.seed)          # same draws as control_sweeps.py with the same seed
    fit_rel = read_split(args.fit_split)
    fit_rel = rng.sample(fit_rel, min(args.max_fit, len(fit_rel)))
    test_rel = read_split(args.split_file)
    test_rel = rng.sample(test_rel, min(args.n_sweep, len(test_rel)))
    fit_ds = make_dataset(args.data_dir, fit_rel, sr, chunk, **pre)
    test_ds = make_dataset(args.data_dir, test_rel, sr, chunk, **pre)
    print(f'{args.resume} (epoch {epoch}) on {device}: fit {len(fit_rel)}, sweep {len(test_rel)}')

    mu_fit, lv_fit, y_fit = encode_decode(model, fit_ds, list(range(len(fit_ds))), device)
    T = mu_fit.shape[2]
    kl_dim = (0.5 * (mu_fit.pow(2) + lv_fit.exp() - 1 - lv_fit)).mean((0, 2)).numpy()
    active = np.nonzero(kl_dim > args.kl_thresh)[0]
    if len(active) == 0:
        active = np.argsort(kl_dim)[-2:]
    del lv_fit
    Y_raw = desc_array(y_fit.numpy(), sr)
    keep = ~np.isnan(Y_raw).any(1)
    mean, std = Y_raw[keep].mean(0), Y_raw[keep].std(0)
    Y = (Y_raw - mean) / std
    mu_t, _, y_t = encode_decode(model, test_ds, list(range(len(test_ds))), device)
    D0 = (desc_array(y_t.numpy(), sr) - mean) / std

    a_fit, _, h_fit, t_fit = head_tail(y_fit, T, args.head)
    a_t, piv_t, h_t, t_t = head_tail(y_t, T, args.head)
    print(f'  tail frames per test hit: median {t_t.sum(1).median():.0f}; '
          f'head: median {h_t.sum(1).median():.0f}')
    fit = lambda m: fit_directions(pool(mu_fit, m)[keep].numpy(), Y[keep], args.pcs, active)
    dirs = {'static': fit(a_fit), 'tail': fit(t_fit), 'head': fit(h_fit)}
    jd, ja = KNOBS.index('decay'), KNOBS.index('attack')

    # control -> (descriptor index, fn(s) -> latent (N, D, T), cv R^2 or None[, own grid])
    controls = {}
    for kind in args.kinds:
        v = lambda reg, k: torch.tensor(dirs[reg][k][kind], dtype=torch.float32)[None, :, None]
        r2 = lambda reg, k: dirs[reg][k]['r2_cv'][kind]
        controls[f'decay/static/{kind}'] = (jd, lambda s, v=v('static', 'decay'): mu_t + s * v * a_t[:, None],
                                            r2('static', 'decay'))
        controls[f'decay/tail/{kind}'] = (jd, lambda s, v=v('tail', 'decay'): mu_t + s * v * t_t[:, None],
                                          r2('tail', 'decay'))
        controls[f'decay/ramp/{kind}'] = (jd, lambda s, v=v('tail', 'decay'): mu_t + s * v * ramp(t_t)[:, None],
                                          r2('tail', 'decay'))
        controls[f'attack/static/{kind}'] = (ja, lambda s, v=v('static', 'attack'): mu_t + s * v * a_t[:, None],
                                             r2('static', 'attack'))
        controls[f'attack/head/{kind}'] = (ja, lambda s, v=v('head', 'attack'): mu_t + s * v * h_t[:, None],
                                           r2('head', 'attack'))
    z_sil = silence_latent(mu_fit, y_fit)          # compressed tails fade into it, as in the app
    stretch = lambda s: stretch_tail(mu_t, piv_t, 10 ** (s * std[jd]), pad=z_sil)
    controls['decay/stretch'] = (jd, stretch, None)
    # the same operator over the factors a slider would offer: s = log10(r) / std, so
    # slope and leakage stay in the units of every other row
    controls['decay/stretch_x'] = (jd, stretch, None,
                                   [float(np.log10(r) / std[jd]) for r in args.stretch_factors])

    summary = {}
    if args.only:                    # a subset, merged into an earlier run's summary
        controls = {k: c for k, c in controls.items() if any(k.startswith(o) for o in args.only)}
        prev = os.path.join(out_dir, 'summary.json')
        if os.path.isfile(prev):
            with open(prev) as f:
                summary = json.load(f)['controls']
    fig, axes = plt.subplots(2, (len(controls) + 1) // 2, figsize=(3.6 * ((len(controls) + 1) // 2), 7),
                             sharey=True, squeeze=False)
    for ax, (name, (j, make, r2, *own)) in zip(axes.flat, controls.items()):
        grid = own[0] if own else args.grid
        i0 = int(np.argmin(np.abs(grid)))
        Ds = np.stack([D0 if s == 0 else
                       (desc_array(decode(model, make(s), device).numpy(), sr) - mean) / std
                       for s in grid])
        met = sweep_metrics(Ds, grid, j)
        summary[name] = {**met, 'r2_cv': r2}
        if name.startswith('decay/stretch'):                      # safe range as factors
            summary[name]['safe_factors'] = [float(10 ** (x * std[jd])) for x in met['safe_range']]
        print(f'  {name:<22} slope {met["slope"]:5.2f}  monotone {met["monotone"]:4.0%}  '
              f'leak {met["leakage"]:.2f}', flush=True)
        dD = Ds - Ds[i0]
        for q, k in enumerate(KNOBS):
            ax.plot(grid, np.nanmean(dD[:, :, q], 1), lw=2.5 if q == j else 1,
                    alpha=1 if q == j else 0.6, label=k)
        ax.plot(grid, grid, 'k:', lw=1)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('s (std units)')
        if args.n_audio:
            write_strips(model, make, grid, test_rel, test_ds,
                         os.path.join(out_dir, 'audio', name.replace('/', '_')), sr, device,
                         args.n_audio)
    axes[0, 0].set_ylabel('mean change (std units)')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f'{args.name}: time-varying controls (pivot = peak + {args.head} frames)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'sweeps_' + '_'.join(o.replace('/', '-') for o in args.only)
                             + '.png' if args.only else 'sweeps.png'), dpi=110)

    lines = [f'{"control":<24}{"cv_R2":>7}{"slope":>7}{"r2":>6}{"monot":>7}{"leak":>6}  safe_range']
    for key, m in summary.items():
        cv = f'{m["r2_cv"]:>7.2f}' if m['r2_cv'] is not None else f'{"-":>7}'
        lines.append(f'{key:<24}{cv}{m["slope"]:>7.2f}{m["r2"]:>6.2f}{m["monotone"]:>7.0%}'
                     f'{m["leakage"]:>6.2f}  [{m["safe_range"][0]:+.1f}, {m["safe_range"][1]:+.1f}]'
                     + (f'  = x{m["safe_factors"][0]:.2f} .. x{m["safe_factors"][1]:.2f}'
                        if 'safe_factors' in m else ''))
    text = '\n'.join(lines)
    print('\n' + text)
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write(text + '\n')
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump({'head': args.head, 'controls': summary}, f, indent=1)
    print(f'Saved {out_dir}/summary.txt, sweeps.png')


def write_strips(model, make, grid, rel, ds, out_dir, sr, device, n):
    """As control_sweeps.write_strips: one hit per type, swept low to high, 0.3 s gaps."""
    os.makedirs(out_dir, exist_ok=True)
    by_type = defaultdict(list)
    for i, l in enumerate(ds.labels):
        by_type[l].append(i)
    picks = [ix[0] for _, ix in sorted(by_type.items())][:n]
    picks += [i for i in range(len(ds.labels)) if i not in picks][:n - len(picks)]
    Z = torch.stack([make(s)[picks] for s in grid], 1)                # (n, S, D, T)
    gap = np.zeros(int(0.3 * sr), dtype=np.float32)
    for i, z in zip(picks, Z):
        parts = []
        for y in decode(model, z, device).numpy():
            nz = np.nonzero(np.abs(y) > np.abs(y).max() * 1e-3)[0]
            parts += [y[:(nz[-1] if len(nz) else 0) + int(0.05 * sr)], gap]
        stem = os.path.splitext(os.path.basename(rel[i]))[0][:32].replace(' ', '_')
        w = np.concatenate(parts)
        sf.write(os.path.join(out_dir, f'{ds.labels[i]}_{stem}.wav'),
                 w / (np.abs(w).max() + 1e-8) * 0.95, sr, subtype='PCM_16')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Time-varying decay / attack controls')
    p.add_argument('-r', '--resume', required=True)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('-o', '--out_dir', help='default sweeps/<name>/time')
    p.add_argument('--fit_split', default='splits/train.txt')
    p.add_argument('--split_file', default='splits/test.txt')
    p.add_argument('--max_fit', type=int, default=1200)
    p.add_argument('--n_sweep', type=int, default=96)
    p.add_argument('--grid', type=float, nargs='+', default=[-3, -2, -1, 0, 1, 2, 3])
    p.add_argument('--only', nargs='+', help='run only controls starting with these, e.g. '
                   'decay/stretch; merged into an existing summary')
    p.add_argument('--stretch_factors', type=float, nargs='+',
                   default=[0.25, 0.354, 0.5, 0.707, 1, 1.414, 2, 2.828, 4],
                   help='tail stretch factors for decay/stretch_x (half-octave steps)')
    p.add_argument('--head', type=int, default=2, help='pivot = peak frame + head')
    p.add_argument('--kinds', nargs='+', default=['pca', 'kl'], choices=['pca', 'kl', 'ridge'])
    p.add_argument('--pcs', type=int, default=8)
    p.add_argument('--kl_thresh', type=float, default=0.2)
    p.add_argument('--n_audio', type=int, default=7)
    p.add_argument('-d', '--device')
    p.add_argument('--seed', type=int, default=0)
    main(p.parse_args())
