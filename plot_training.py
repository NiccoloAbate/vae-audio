#!/usr/bin/env python
"""
Plot training curves for a RawAudioVAE run from its info.log file.

Usage:
    python plot_training.py <run_id>
    python plot_training.py 0309_233328
    python plot_training.py          # uses most recent run
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LOG_BASE = Path('saved/log/RawAudioVAE')


def latest_run(log_base):
    for run in sorted(log_base.iterdir(), reverse=True):
        if (run / 'info.log').exists():
            return run.name
    raise FileNotFoundError(f'No info.log found under {log_base}')


def parse_log(log_path):
    """Parse info.log into a dict of metric_name → list of (epoch, value)."""
    text   = log_path.read_text()
    blocks = re.split(r'epoch\s+: (\d+)', text)

    metrics = {}
    for i in range(1, len(blocks), 2):
        epoch  = int(blocks[i])
        block  = blocks[i + 1]
        for key, val in re.findall(r'(\w+)\s+: ([0-9eE+\-.]+)', block):
            if key == 'epoch':
                continue
            metrics.setdefault(key, []).append((epoch, float(val)))

    return {k: (np.array([e for e, _ in v]), np.array([v_ for _, v_ in v]))
            for k, v in metrics.items()}


def plot_training(run_id, out_dir):
    log_path = LOG_BASE / run_id / 'info.log'
    if not log_path.exists():
        raise FileNotFoundError(f'Log not found: {log_path}')

    m = parse_log(log_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Figure 1 — Reconstruction + KL + Total loss (train vs val)
    # ------------------------------------------------------------------ #
    pairs = [
        ('loss_recon',  'val_loss_recon', 'Reconstruction loss'),
        ('loss_kl',     'val_loss_kl',    'KL loss'),
        ('loss',        'val_loss',       'Total loss  (recon + β·KL)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Training curves — run {run_id}', fontsize=11)

    for ax, (train_key, val_key, title) in zip(axes, pairs):
        if train_key in m:
            ax.plot(*m[train_key], label='train', linewidth=1.2)
        if val_key in m:
            ax.plot(*m[val_key],   label='val',   linewidth=1.2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / 'training_curves.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved  {out_path}')

    # ------------------------------------------------------------------ #
    # Figure 2 — Beta schedule
    # ------------------------------------------------------------------ #
    if 'beta' in m:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(*m['beta'], linewidth=1.5, color='#E8724C')
        ax.set_title(f'KL weight (β) schedule — run {run_id}', fontsize=10)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('β')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = out_dir / 'beta_schedule.png'
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'Saved  {out_path}')

    # ------------------------------------------------------------------ #
    # Figure 3 — val_loss_recon with best epoch marked
    # ------------------------------------------------------------------ #
    if 'val_loss_recon' in m:
        epochs, vals = m['val_loss_recon']
        best_idx     = np.argmin(vals)
        fig, ax      = plt.subplots(figsize=(8, 3.5))
        ax.plot(epochs, vals, linewidth=1.2, label='val_loss_recon')
        ax.axvline(epochs[best_idx], color='red', linestyle='--', linewidth=1.2,
                   label=f'best epoch {epochs[best_idx]:.0f}  ({vals[best_idx]:.4f})')
        ax.set_title('Validation reconstruction loss — best checkpoint indicator', fontsize=10)
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = out_dir / 'val_recon_best.png'
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'Saved  {out_path}')
        print(f'\nBest val_loss_recon: {vals[best_idx]:.4f}  at epoch {epochs[best_idx]:.0f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot RawAudioVAE training curves')
    parser.add_argument('run_id', nargs='?', default=None,
                        help='Run ID (e.g. 0309_233328). Defaults to most recent run with a log.')
    parser.add_argument('--out', default=None,
                        help='Output directory. Defaults to eval/<run_id>/.')
    args = parser.parse_args()

    run_id  = args.run_id or latest_run(LOG_BASE)
    out_dir = Path(args.out) if args.out else Path('eval') / run_id
    print(f'Run:    {run_id}')
    print(f'Output: {out_dir}\n')

    plot_training(run_id, out_dir)
