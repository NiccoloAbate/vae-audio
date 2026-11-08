"""
Control sweeps: can linear latent directions act like drum-machine knobs?

Classic drum-machine controls map onto measurable descriptors:
    Tone   -> brightness  log2 spectral centroid (octaves)
    Snappy -> noisiness   log10 spectral flatness
    Decay  -> decay       log10 time from the peak to the last sample within
                          30 dB of it
    Attack -> attack      log10 attack time (eval_pipeline.log_attack_time)
All are computed on the model's own decoded audio, energy-weighted over the
frames within 40 dB of the loudest one.

1. Fit: encode + decode training hits, pool each hit's latent over time, and
   fit a ridge regression descriptor ~ pooled latent (cross-validated R^2).
   The knob direction is v = beta / |beta|^2, the smallest latent offset that
   moves the *predicted* descriptor by one unit; descriptors are standardised
   on the fit set, so a knob value s means "+s standard deviations".
   Variants: 'ridge' uses all latent dims; 'pca' fits only within the top
   --pcs principal components of the pooled latent (where the decoder's real
   variation lives; dims that correlate with a descriptor but that the
   decoder ignores can't absorb the direction); 'kl' fits only within the
   dims whose mean KL exceeds --kl_thresh (above the free-bits floor, i.e.
   the dims the decoder uses; needed when the latent has partly collapsed);
   'single' is the one latent dim most correlated with the descriptor.
2. Sweep: on test hits, add s*v to the latent for s in --grid, decode, and
   re-measure every descriptor.
     slope       measured change in the target per unit s (1 = as promised)
     r2          how linearly the target follows s
     monotone    share of hits whose target moves the right way at every step
     leakage     largest |slope| of the other descriptors (0 = disentangled)
     safe_range  largest |s| on each side where >= 75% of hits still move
                 the right way at each step; the app's slider range
   The offset goes either on every latent frame ('all') or only where the hit
   is sounding ('active': frames of the recon within 50 dB of its loudest,
   dilated by 2); pooling for the fit matches.

Outputs under <out_dir>/ (default sweeps/<name>):
    summary.txt, summary.json   the table above, per mode x knob x direction
    sweeps_<mode>.png           mean descriptor change vs s
    directions.pt               knob vectors + calibration + safe ranges (app)
    audio/<kind>/<hit>_<knob>.wav  one hit swept from -max to +max s, with gaps
                                   ('active' mode only)

Usage:
    python control_sweeps.py -r saved/models/DrumOneShots/checkpoint-epoch1500.pth \
        --data_dir dataset/DrumOneShots --name prototype
"""

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict

from eval_pipeline import load_checkpoint, read_split, make_dataset, log_attack_time
from drum_engine import frame_activity
from evaluate_raw import get_device

KNOBS = ['brightness', 'noisiness', 'decay', 'attack']
ANALOGUE = {'brightness': 'Tone', 'noisiness': 'Snappy', 'decay': 'Decay', 'attack': 'Attack'}


# ---------------------------------------------------------------------------
# descriptors
# ---------------------------------------------------------------------------

def descriptors(w, sr):
    """Dict of the KNOBS descriptors for one waveform (see module docstring)."""
    w = np.asarray(w, dtype=np.float64)
    n_fft = 2048 if sr > 32000 else 1024
    f, _, Z = scipy.signal.stft(w, sr, nperseg=n_fft, noverlap=3 * n_fft // 4,
                                boundary=None, padded=False)
    P = np.abs(Z) ** 2 + 1e-12                                   # (F, T)
    e = P.sum(0)
    wts = e * (e > e.max() * 1e-4)                               # frames within 40 dB
    centroid = (f[:, None] * P).sum(0) / P.sum(0)
    flat = np.exp(np.log(P).mean(0)) / P.mean(0)

    env = np.abs(scipy.signal.hilbert(w))
    k = max(1, int(0.01 * sr))
    env = np.convolve(env, np.ones(k) / k, mode='same')
    pk = int(env.argmax())
    above = np.nonzero(env[pk:] >= env[pk] * 10 ** (-30 / 20))[0]
    decay_s = (above[-1] + 1) / sr if len(above) else 1 / sr

    return {'brightness': float(np.log2(max(np.average(centroid, weights=wts), 1.0))),
            'noisiness': float(np.log10(np.average(flat, weights=wts))),
            'decay': float(np.log10(decay_s)),
            'attack': log_attack_time(w, sr)}


def desc_array(waves, sr):
    """(N, L) -> (N, len(KNOBS))"""
    return np.array([[descriptors(w, sr)[k] for k in KNOBS] for w in waves])


# ---------------------------------------------------------------------------
# model helpers
# ---------------------------------------------------------------------------

def encode_decode(model, ds, idxs, device, bs=16):
    """Posterior means and log-variances (N, D, T), decoded recon (N, L), on CPU."""
    mus, lvs, ys = [], [], []
    with torch.no_grad():
        for s in range(0, len(idxs), bs):
            x = torch.stack([ds[i][2] for i in idxs[s:s + bs]]).to(device)
            mu, lv, _ = model.encode(x)
            mus.append(mu.cpu()); lvs.append(lv.cpu())
            ys.append(model.decode(mu).squeeze(1).cpu())
    return torch.cat(mus), torch.cat(lvs), torch.cat(ys)


def decode(model, z, device, bs=16):
    with torch.no_grad():
        return torch.cat([model.decode(z[s:s + bs].to(device)).squeeze(1).cpu()
                          for s in range(0, len(z), bs)])


def pool(mu, mask):
    """(N, D, T), (N, T) -> (N, D) masked time-mean."""
    return (mu * mask[:, None, :]).sum(2) / mask.sum(1, keepdim=True).clamp_min(1)


# ---------------------------------------------------------------------------
# fit + sweep
# ---------------------------------------------------------------------------

def cv_r2(X, y):
    pred = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), X, y, cv=5)
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def fit_directions(X, Y, pcs, active):
    """X (N, D) pooled latents, Y (N, K) standardised descriptors -> per knob:
    ridge (all dims), pca (ridge in the top `pcs` PCs), kl (ridge on the
    `active` dims only), and best single dim."""
    out = {}
    Xc = X - X.mean(0)
    U = np.linalg.svd(Xc, full_matrices=False)[2][:pcs].T          # (D, pcs)
    for j, k in enumerate(KNOBS):
        y = Y[:, j]
        beta = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X, y).coef_
        gamma = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xc @ U, y).coef_
        beta_pca = U @ gamma
        beta_kl = np.zeros_like(beta)
        beta_kl[active] = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X[:, active], y).coef_
        corr = np.array([np.corrcoef(X[:, d], y)[0, 1] for d in range(X.shape[1])])
        d = int(np.nanargmax(np.abs(corr)))
        a = np.polyfit(X[:, d], y, 1)[0]                          # y_std per unit of dim d
        single = np.zeros_like(beta); single[d] = 1.0 / a
        out[k] = {'ridge': beta / (beta @ beta), 'pca': beta_pca / (beta_pca @ beta_pca),
                  'kl': beta_kl / (beta_kl @ beta_kl),
                  'single': single, 'single_dim': d, 'single_r': float(corr[d]),
                  'r2_cv': {'ridge': cv_r2(X, y), 'pca': cv_r2(Xc @ U, y),
                            'kl': cv_r2(X[:, active], y), 'single': cv_r2(X[:, [d]], y)}}
    return out


def sweep_metrics(D, grid, j):
    """D (S, N, K) standardised descriptors over the s grid for knob index j."""
    grid = np.asarray(grid, dtype=float)
    i0 = int(np.argmin(np.abs(grid)))
    dD = D - D[i0]                                                # change vs s = 0
    s_all = np.repeat(grid, D.shape[1])
    slopes = [np.polyfit(s_all, dD[:, :, q].ravel(), 1)[0] for q in range(D.shape[2])]
    t = dD[:, :, j].ravel(); fit = np.polyval(np.polyfit(s_all, t, 1), s_all)
    r2 = 1 - ((t - fit) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-12)
    steps = np.diff(D[:, :, j], axis=0) > 0                       # (S-1, N) moved up
    monotone = float(steps.all(0).mean())
    frac = steps.mean(1)                                          # per step
    hi = 0.0
    for i in range(i0, len(grid) - 1):
        if frac[i] < 0.75: break
        hi = grid[i + 1]
    lo = 0.0
    for i in range(i0 - 1, -1, -1):
        if frac[i] < 0.75: break
        lo = grid[i]
    others = [abs(slopes[q]) for q in range(len(slopes)) if q != j]
    return {'slope': float(slopes[j]), 'r2': float(r2), 'monotone': monotone,
            'leakage': float(max(others)), 'leak_slopes': [float(x) for x in slopes],
            'safe_range': [float(lo), float(hi)]}


def main(args):
    device = torch.device(args.device) if args.device else get_device()
    model, cfg, epoch = load_checkpoint(args.resume)
    model.to(device)
    dl = cfg['data_loader']['args']
    sr, chunk = dl.get('sr', 22050), dl.get('chunk_size', 110080)
    pre = dict(onset_thresh=dl.get('onset_thresh', 0.02), pre_pad=dl.get('pre_pad', 256),
               peak_norm=dl.get('peak_norm', True))
    out_dir = args.out_dir or os.path.join('sweeps', args.name)
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    fit_rel = read_split(args.fit_split)
    fit_rel = rng.sample(fit_rel, min(args.max_fit, len(fit_rel)))
    test_rel = read_split(args.split_file)
    test_rel = rng.sample(test_rel, min(args.n_sweep, len(test_rel)))
    fit_ds = make_dataset(args.data_dir, fit_rel, sr, chunk, **pre)
    test_ds = make_dataset(args.data_dir, test_rel, sr, chunk, **pre)
    print(f'{args.resume} (epoch {epoch}) on {device}: fit on {len(fit_rel)}, sweep {len(test_rel)} hits')

    print('Encoding fit set…')
    mu_fit, lv_fit, y_fit = encode_decode(model, fit_ds, list(range(len(fit_ds))), device)
    T = mu_fit.shape[2]
    kl_dim = (0.5 * (mu_fit.pow(2) + lv_fit.exp() - 1 - lv_fit)).mean((0, 2)).numpy()
    active = np.nonzero(kl_dim > args.kl_thresh)[0]
    if len(active) == 0:
        active = np.argsort(kl_dim)[-2:]
    print(f'  dims above KL {args.kl_thresh}: {list(active)} (of {len(kl_dim)})')
    del lv_fit
    Y_raw = desc_array(y_fit.numpy(), sr)
    keep = ~np.isnan(Y_raw).any(1)
    mean, std = Y_raw[keep].mean(0), Y_raw[keep].std(0)
    Y = (Y_raw - mean) / std
    print('  descriptor mean/std on fit set: ' +
          ', '.join(f'{k}={m:.2f}±{s:.2f}' for k, m, s in zip(KNOBS, mean, std)))

    print('Encoding sweep set…')
    mu_t, _, y_t = encode_decode(model, test_ds, list(range(len(test_ds))), device)
    grid = args.grid
    i0 = grid.index(0.0)

    summary, directions = {}, {}
    for mode in args.modes:
        m_fit = frame_activity(y_fit, T) if mode == 'active' else torch.ones(len(mu_fit), T)
        m_t = frame_activity(y_t, T) if mode == 'active' else torch.ones(len(mu_t), T)
        dirs = fit_directions(pool(mu_fit, m_fit)[keep].numpy(), Y[keep], args.pcs, active)
        D0 = (desc_array(y_t.numpy(), sr) - mean) / std
        fig, axes = plt.subplots(
            4, len(KNOBS), figsize=(4 * len(KNOBS), 13), sharey=True, squeeze=False)
        for j, k in enumerate(KNOBS):
            for row, kind in enumerate(['ridge', 'pca', 'kl', 'single']):
                v = torch.tensor(dirs[k][kind], dtype=torch.float32)
                Ds = []
                for s in grid:
                    if s == 0.0:
                        Ds.append(D0); continue
                    z = mu_t + s * v[None, :, None] * m_t[:, None, :]
                    Ds.append((desc_array(decode(model, z, device).numpy(), sr) - mean) / std)
                Ds = np.stack(Ds)                                  # (S, N, K)
                met = sweep_metrics(Ds, grid, j)
                summary[f'{mode}/{k}/{kind}'] = {**met, 'r2_cv': dirs[k]['r2_cv'][kind],
                                                 'single_dim': dirs[k]['single_dim']}
                ax = axes[row, j]
                dD = Ds - Ds[i0]
                for q, kk in enumerate(KNOBS):
                    mu_q = np.nanmean(dD[:, :, q], 1)
                    ax.plot(grid, mu_q, lw=2.5 if q == j else 1, alpha=1 if q == j else 0.6,
                            label=kk)
                ax.plot(grid, grid, 'k:', lw=1)
                ax.set_title(f'{ANALOGUE[k]} ({k}) - {kind}'
                             + (f' d{dirs[k]["single_dim"]}' if kind == 'single' else ''), fontsize=9)
                ax.set_xlabel('knob s (std units)')
                if j == 0: ax.set_ylabel('mean change (std units)')
                if j == 0 and row == 0: ax.legend(fontsize=7)
                directions[f'{mode}/{k}/{kind}'] = {
                    'vector': v, 'mean': float(mean[j]), 'std': float(std[j]), **met,
                    'r2_cv': dirs[k]['r2_cv'][kind], 'analogue': ANALOGUE[k]}
                if kind != 'single':
                    if args.n_audio and mode == 'active':
                        write_strips(model, mu_t, m_t, v, grid, test_rel, test_ds,
                                     os.path.join(out_dir, 'audio', kind), k, sr, device,
                                     args.n_audio)
        fig.suptitle(f'{args.name}: knob sweeps (offset on {mode} frames)', fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'sweeps_{mode}.png'), dpi=110)
        plt.close(fig)

    lines = [f'{"mode/knob/direction":<30}{"cv_R2":>7}{"slope":>7}{"r2":>6}{"monot":>7}'
             f'{"leak":>6}  safe_range']
    for key, m in summary.items():
        lines.append(f'{key:<30}{m["r2_cv"]:>7.2f}{m["slope"]:>7.2f}{m["r2"]:>6.2f}'
                     f'{m["monotone"]:>7.0%}{m["leakage"]:>6.2f}  [{m["safe_range"][0]:+.0f}, '
                     f'{m["safe_range"][1]:+.0f}]')
    text = '\n'.join(lines)
    print('\n' + text)
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write(text + '\n')
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    torch.save({'checkpoint': args.resume, 'knobs': KNOBS, 'grid': grid,
                'directions': directions, 'sr': sr, 'chunk_size': chunk},
               os.path.join(out_dir, 'directions.pt'))
    print(f'Saved  {out_dir}/directions.pt, summary.txt, sweeps_*.png')


def write_strips(model, mu, mask, v, grid, rel, ds, out_dir, knob, sr, device, n):
    """For the first n hits (one per type where possible): the hit swept over
    the grid, low to high, each step trimmed at -60 dB, 0.3 s gaps."""
    os.makedirs(out_dir, exist_ok=True)
    by_type = defaultdict(list)
    for i, l in enumerate(ds.labels):
        by_type[l].append(i)
    picks = [ix[0] for _, ix in sorted(by_type.items())][:n]
    rest = [i for i in range(len(ds.labels)) if i not in picks]   # single-type sets: more hits
    picks += rest[:n - len(picks)]
    gap = np.zeros(int(0.3 * sr), dtype=np.float32)
    for i in picks:
        z = torch.stack([mu[i] + s * v[:, None] * mask[i][None, :] for s in grid])
        ys = decode(model, z, device).numpy()
        parts = []
        for y in ys:
            nz = np.nonzero(np.abs(y) > np.abs(y).max() * 1e-3)[0]
            parts += [y[:(nz[-1] if len(nz) else 0) + int(0.05 * sr)], gap]
        stem = os.path.splitext(os.path.basename(rel[i]))[0][:32].replace(' ', '_')
        w = np.concatenate(parts)
        sf.write(os.path.join(out_dir, f'{ds.labels[i]}_{stem}_{knob}.wav'),
                 w / (np.abs(w).max() + 1e-8) * 0.95, sr, subtype='PCM_16')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Latent knob sweeps for a one-shot RawAudioVAE')
    p.add_argument('-r', '--resume', required=True)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('-o', '--out_dir')
    p.add_argument('--fit_split', default='splits/train.txt')
    p.add_argument('--split_file', default='splits/test.txt')
    p.add_argument('--max_fit', type=int, default=1200)
    p.add_argument('--n_sweep', type=int, default=128)
    p.add_argument('--grid', type=float, nargs='+', default=[-3, -2, -1, 0, 1, 2, 3])
    p.add_argument('--modes', nargs='+', default=['active', 'all'], choices=['active', 'all'])
    p.add_argument('--pcs', type=int, default=8,
                   help="principal components for the 'pca' directions (per-hit PCA "
                        "needed ~7 for 95%% variance on the prototype)")
    p.add_argument('--kl_thresh', type=float, default=0.2,
                   help="nats; 'kl' directions use only dims with mean KL above this")
    p.add_argument('--n_audio', type=int, default=7, help='hits per knob to write as sweep strips')
    p.add_argument('-d', '--device')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    assert 0.0 in args.grid, '--grid must include 0'
    main(args)
