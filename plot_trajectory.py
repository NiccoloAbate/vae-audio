"""
Plot val metrics over training (eval_trajectory.py) next to the training losses
(info.log), one colour per run, to diagnose the adversarial stage: does it never
help, or help and then degrade? Also prints, per run, val FAD before the
adversarial stage, at its best checkpoint and at the end.

Inputs, downloaded from Sherlock:
    <traj_root>/<run>/results_val.csv, steps.csv     ($SCRATCH/eval/trajectory)
    <log_root>/<run>/<timestamp>/info.log            ($SCRATCH/runs/vae_drums/log)

Usage:
    python plot_trajectory.py --traj_root eval/trajectory --log_root runs/vae_drums/log
    python plot_trajectory.py ... --runs Drums_44k_b005 Drums_44k_b002 -o eval/trajectory/b.png
"""

import argparse
import csv
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_training import parse_log


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_traj(run_dir):
    if not os.path.isfile(os.path.join(run_dir, 'results_val.csv')):
        return []
    steps = {r['name']: r for r in read_csv(os.path.join(run_dir, 'steps.csv'))}
    rows = []
    for r in read_csv(os.path.join(run_dir, 'results_val.csv')):
        s = steps.get(r['name'])
        if s and r.get('fad'):
            rows.append(dict(epoch=int(s['epoch']), step=int(s['step']),
                             adv=int(s['adv_start_epoch'] or 0), fad=float(r['fad']),
                             stft=float(r['stft']), lat=float(r['lat_within2'])))
    rows.sort(key=lambda r: r['epoch'])
    return rows


def load_logs(log_dir):
    """Per-epoch training metrics over every info.log of the run (later dirs win)."""
    merged = {}
    for p in sorted(glob.glob(os.path.join(log_dir, '*', 'info.log'))):
        log = parse_log(Path(p))
        if 'loss_disc' in log and 'disc_steps' not in log:
            # older logs averaged loss_disc over all steps with 0 on the skipped ones
            # (disc_update_every 2 in every such run): rescale to the per-update value
            ep, val = log['loss_disc']
            log['loss_disc'] = (ep, val * 2)
        for k, (ep, val) in log.items():
            merged.setdefault(k, {}).update(zip(ep.tolist(), val.tolist()))
    return {k: (np.array(sorted(d)), np.array([d[e] for e in sorted(d)])) for k, d in merged.items()}


def smooth(y, w):
    if w <= 1 or len(y) < w:
        return y
    return np.convolve(np.pad(y, (w // 2, w - 1 - w // 2), mode='edge'), np.ones(w) / w, 'valid')


def summarize(run, rows):
    adv = rows[0]['adv']
    pre = [r for r in rows if r['epoch'] < adv]
    post = [r for r in rows if r['epoch'] >= adv]
    pre_fad = pre[-1]['fad'] if pre else float('nan')
    best = min(post, key=lambda r: r['fad']) if post else None
    final = rows[-1]
    if best is None:
        verdict = 'no adversarial checkpoints'
    elif best['fad'] > 0.9 * pre_fad:
        verdict = 'never clearly helped'
    elif final['fad'] > 1.3 * best['fad']:
        verdict = 'helped, then degraded'
    else:
        verdict = 'helped and held'
    b = f"{best['fad']:.3f} @ {best['step'] / 1e3:.0f}k" if best else '-'
    return (f'{run:22s} pre-adv {pre_fad:.3f}   best {b:>15s}   final {final["fad"]:.3f} '
            f'@ {final["step"] / 1e3:.0f}k   {verdict}')


def main(args):
    root = args.traj_root if os.path.isdir(args.traj_root) else args.log_root
    runs = args.runs or sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    panels = [('fad', 'val FAD (CLAP)'), ('stft', 'val STFT distance'),
              ('loss_recon', 'recon loss (train; val dashed)'), ('loss_disc', 'discriminator loss'),
              ('loss_gen_adv', 'generator adversarial loss'), ('loss_fm', 'feature matching loss')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    lines = []
    for c, run in enumerate(runs):
        color = f'C{c % 10}'
        rows = load_traj(os.path.join(args.traj_root, run))
        logs = load_logs(os.path.join(args.log_root, run)) if args.log_root else {}
        if not rows and not logs:
            continue
        if rows:
            lines.append(summarize(run, rows))
            spe = np.median([r['step'] / r['epoch'] for r in rows if r['epoch']])   # steps per epoch
            adv = rows[0]['adv']
        else:          # logs only: first epoch that logged a discriminator loss
            spe = args.steps_per_epoch
            adv = int(logs['loss_disc'][0][0]) if 'loss_disc' in logs else 0
        for ax, (key, title) in zip(axes.flat, panels):
            if key in ('fad', 'stft'):
                ax.plot([r['step'] / 1e3 for r in rows], [r[key] for r in rows], 'o-', ms=3,
                        color=color, label=run)
            elif key in logs:
                ep, y = logs[key]
                ax.plot(ep * spe / 1e3, smooth(y, args.smooth), color=color, lw=1, label=run)
                if key == 'loss_recon' and 'val_loss_recon' in logs:     # val: dashed
                    ep, y = logs['val_loss_recon']
                    ax.plot(ep * spe / 1e3, smooth(y, args.smooth), color=color, lw=1, ls='--')
            if adv:
                ax.axvline(adv * spe / 1e3, color=color, ls=':', lw=0.8)
            ax.set_title(title, fontsize=10)
    for ax in axes.flat:
        ax.set_xlabel('optimizer steps (k)')
        ax.grid(alpha=0.3)
    if lines:                      # some run has val FAD
        axes[0, 0].set_yscale('log')
    axes[0, 2].legend(fontsize=7)
    fig.suptitle(f'Dotted: adversarial start. Losses smoothed over {args.smooth} epochs.', fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print('\n'.join(lines))
    with open(os.path.splitext(args.out)[0] + '_summary.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('saved', args.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Val metrics + training losses over steps, per run')
    p.add_argument('--traj_root', default='eval/trajectory')
    p.add_argument('--log_root', help='dir of <run>/<timestamp>/info.log (training losses)')
    p.add_argument('--runs', nargs='+', help='default: every run under --traj_root')
    p.add_argument('--steps_per_epoch', type=float, default=388,
                   help='without trajectory csvs (logs only): 388 for the 7-type runs, 78 snare')
    p.add_argument('--smooth', type=int, default=10, help='epochs of moving average on losses')
    p.add_argument('-o', '--out', default='eval/trajectory/trajectory.png')
    main(p.parse_args())
