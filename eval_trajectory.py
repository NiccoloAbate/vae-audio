"""
Score every saved checkpoint of a run on the validation split: FAD, STFT and
attack error over training, to see when (and whether) the adversarial stage
helps or breaks a run, and to pick checkpoints on val instead of test.

Each checkpoint goes through eval_pipeline.py unchanged (same metrics as the
results tables, minus the truncation curve), so the numbers are comparable.
Rows go to <out_dir>/results_val.csv; <out_dir>/steps.csv adds each
checkpoint's optimizer step and the run's adversarial start. Already-scored
checkpoints are skipped, so a killed job can simply be resubmitted.

Usage (Sherlock: eval_trajectory.sh wraps this):
    python eval_trajectory.py --runs_dir runs/vae_drums/models --run Drums_44k_b005 \
        --data_dir dataset/DrumOneShots --fadtk <fad env>/bin/fadtk -o eval/trajectory/Drums_44k_b005
Then plot locally with plot_trajectory.py.
"""

import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys

import torch


def checkpoints(runs_dir, run, every):
    """{epoch: path} over all of the run's timestamp dirs (resumes start new ones; for an
    epoch saved twice the later dir wins), every `every` epochs plus the last one."""
    found = {}
    for p in sorted(glob.glob(os.path.join(runs_dir, run, '*', 'checkpoint-epoch*.pth'))):
        found[int(re.search(r'epoch(\d+)\.pth$', p).group(1))] = p
    if not found:
        raise FileNotFoundError(f'no checkpoints under {runs_dir}/{run}/*/')
    last = max(found)
    return {e: found[e] for e in sorted(found) if e % every == 0 or e == last}


def done_names(path):
    if not os.path.isfile(path):
        return set()
    with open(path) as f:
        return {r['name'] for r in csv.DictReader(f) if r.get('fad')}


def ckpt_info(path):
    """Adam's step count (= optimizer steps taken) and the run's adversarial start epoch."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    steps = [s['step'] for s in ck['optimizer']['state'].values() if 'step' in s]
    cfg = ck['config'].config if hasattr(ck['config'], 'config') else ck['config']
    return int(max(float(s) for s in steps)) if steps else '', cfg['trainer'].get('adv_start_epoch', '')


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    results = os.path.join(args.out_dir, 'results_val.csv')
    steps_csv = os.path.join(args.out_dir, 'steps.csv')
    ref_dir = args.ref_dir or os.path.join(args.out_dir, 'reference')
    work = os.path.join(args.out_dir, 'work')
    done = done_names(results)
    cks = checkpoints(args.runs_dir, args.run, args.every)
    print(f'{args.run}: {len(cks)} checkpoints, {len(done)} already scored')
    new_steps = not os.path.isfile(steps_csv)
    for e, path in cks.items():
        name = f'{args.run}_ep{e}' + (f'_gate{-args.gate_db:g}' if args.gate_db is not None else '') \
            + (f'_zs{args.sample_z}' if args.sample_z is not None else '')
        if name in done:
            continue
        print(f'=== {name}', flush=True)
        cmd = [sys.executable, 'eval_pipeline.py', '-r', path, '--data_dir', args.data_dir,
               '--split_file', args.split_file, '--name', name, '-o', work,
               '--results', results, '--ref_dir', ref_dir, '--no_truncation',
               '--n_plot', '1', '--fadtk', args.fadtk]
        if args.device:
            cmd += ['-d', args.device]
        if args.gate_db is not None:
            cmd += ['--gate_db', str(args.gate_db)]
        if args.sample_z is not None:
            cmd += ['--sample_z', str(args.sample_z)]
        subprocess.run(cmd, check=True)
        shutil.rmtree(work, ignore_errors=True)      # ~50 MB of wavs per checkpoint
        step, adv = ckpt_info(path)
        with open(steps_csv, 'a', newline='') as f:
            w = csv.writer(f)
            if new_steps:
                w.writerow(['name', 'run', 'epoch', 'step', 'adv_start_epoch'])
                new_steps = False
            w.writerow([name, args.run, e, step, adv])
    print('done', args.run)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Val-split metrics for every checkpoint of a run')
    p.add_argument('--runs_dir', required=True, help='parent of <run>/<timestamp>/checkpoint-*.pth')
    p.add_argument('--run', required=True, help='run name, e.g. Drums_44k_b005')
    p.add_argument('--data_dir', required=True)
    p.add_argument('--split_file', default='splits/val.txt')
    p.add_argument('-o', '--out_dir', required=True)
    p.add_argument('--ref_dir', help='default <out_dir>/reference (one per job: no races)')
    p.add_argument('--every', type=int, default=50, help='score epochs divisible by this (+ the last)')
    p.add_argument('--fadtk', required=True)
    p.add_argument('-d', '--device')
    p.add_argument('--gate_db', type=float, help='passed to eval_pipeline.py; rows get a _gate<dB> suffix')
    p.add_argument('--sample_z', type=int, help='passed to eval_pipeline.py; rows get a _zs<seed> suffix')
    main(p.parse_args())
