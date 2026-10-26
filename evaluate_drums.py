"""
Evaluate a trained one-shot RawAudioVAE checkpoint (e.g. drum hits).

Why a separate script from evaluate_raw.py:
  evaluate_raw.py assumes long recordings sliced into interchangeable chunks
  that are stitched back with a crossfade — correct for sustained instruments
  (cello) but wrong for one-shots. Here each file is ONE coherent event, so:

    * reconstruction is a single forward pass over the padded/attack-aligned
      chunk (no chunking, no crossfade),
    * generation does NOT sample the i.i.d. N(0,1) prior. For a *temporal*
      latent (D, T_lat), independent per-frame samples scatter little
      transients across the whole 5 s instead of producing one clean hit
      (verified on this checkpoint). Instead we generate by:
        - interpolating between real encoded hits (slerp on the latent seq),
        - perturbing a real encoded hit with small latent noise,
        - a decay-envelope baseline (random latent vector × exp decay).

Produces under <out_dir>/:
    recon_waveforms.png     input vs recon, zoomed on the attack
    recon_spectrograms.png  log-STFT input / recon / difference
    recon_metrics.txt       per-class multi-res STFT loss + onset-time error
    latent_pca.png          2-D PCA of pooled latent means, coloured by class
    kl_per_dim.png          KL per latent dim (active vs collapsed)
    audio/                  orig/recon .wav pairs
    interp/                 latent-interpolation audio + spectrogram grids
    gen/                    generation audio + spectrograms

Usage (run where the dataset lives, e.g. Sherlock):
    python evaluate_drums.py -r saved/models/DrumOneShots/checkpoint-epoch1500.pth
    python evaluate_drums.py -r <ckpt> -c <config.json> -n 8 -o eval/drums
"""

import argparse
import os
import random
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import soundfile as sf

import data_loader.data_loaders as module_data
import model.model as module_arch
from model.loss import multi_res_stft_loss
from parse_config import ConfigParser

# Reuse the pure (dataset-agnostic) helpers from evaluate_raw
from evaluate_raw import (get_device, log_magnitude_stft, slerp,
                          plot_kl_per_dim, plot_latent_pca)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_model_and_data(config, resume, device, seed=42):
    dl_args = dict(config['data_loader']['args'])
    dl_args.update({'validation_split': 0.0, 'shuffle': True, 'num_workers': 0})
    torch.manual_seed(seed)
    data_loader = getattr(module_data, config['data_loader']['type'])(**dl_args)

    model = config.initialize('arch', module_arch)
    ckpt = torch.load(resume, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    return model, data_loader


# ---------------------------------------------------------------------------
# collection (one item == one hit)
# ---------------------------------------------------------------------------

def pack_label(path, roots):
    """Group label = the top-level folder under the dataset root (the sample
    pack), not the immediate parent — deeply-nested one-shot libraries would
    otherwise yield 100s of messy per-folder labels. Falls back to parent dir."""
    p = os.path.abspath(path)
    for r in roots:
        r = os.path.abspath(r)
        if p.startswith(r + os.sep):
            return os.path.relpath(p, r).split(os.sep)[0]
    return os.path.basename(os.path.dirname(path))


def collect(model, data_loader, roots, device, max_items=256):
    """Run hits through the model. Returns pooled latents + waveforms + meta.

    mu/logvar are mean-pooled over time -> one D-vector per hit (for PCA / KL).
    Group label is the top-level pack (see pack_label). Stops after ~max_items.
    """
    dataset = data_loader.dataset
    mu_l, lv_l, inp_l, rec_l, lbl_l, fn_l = [], [], [], [], [], []

    with torch.no_grad():
        for data_idx, label, data in data_loader:
            x = data.float().to(device)
            y_hat, mu, logvar, _ = model(x)
            mu_l.append(mu.mean(dim=2).cpu())
            lv_l.append(logvar.mean(dim=2).cpu())
            inp_l.append(x.squeeze(1).cpu())
            rec_l.append(y_hat.squeeze(1).cpu())
            for idx in data_idx.tolist():
                path = dataset.files[idx]
                fn_l.append(os.path.basename(path))
                lbl_l.append(pack_label(path, roots))
            if len(fn_l) >= max_items:
                break

    return (torch.cat(mu_l), torch.cat(lv_l),
            torch.cat(inp_l), torch.cat(rec_l), lbl_l, fn_l)


def pick_per_class(labels, n_samples, seed=0):
    """One representative index per class, up to n_samples (round-robins classes)."""
    by_class = defaultdict(list)
    for i, l in enumerate(labels):
        by_class[l].append(i)
    rng = random.Random(seed)
    picks, classes = [], sorted(by_class)
    while len(picks) < n_samples and any(by_class.values()):
        for c in classes:
            if by_class[c]:
                picks.append(by_class[c].pop(rng.randrange(len(by_class[c]))))
                if len(picks) >= n_samples:
                    break
    return picks


# ---------------------------------------------------------------------------
# reconstruction metrics + plots
# ---------------------------------------------------------------------------

def onset_sample(wave, thresh=0.1):
    """Index of first sample exceeding thresh*peak (crude attack-time proxy)."""
    e = np.abs(wave)
    pk = e.max()
    if pk <= 0:
        return 0
    above = np.nonzero(e > thresh * pk)[0]
    return int(above[0]) if len(above) else int(e.argmax())


def recon_metrics(inputs, recons, labels, out_path, sr=22050):
    """Per-class multi-res STFT loss + onset-time error; writes + prints a table."""
    per_class_stft = defaultdict(list)
    per_class_onset = defaultdict(list)
    for i in range(inputs.size(0)):
        x = inputs[i:i+1].unsqueeze(1)
        y = recons[i:i+1].unsqueeze(1)
        loss = float(multi_res_stft_loss(x, y))
        on_err = abs(onset_sample(inputs[i].numpy()) -
                     onset_sample(recons[i].numpy())) / sr * 1000.0  # ms
        per_class_stft[labels[i]].append(loss)
        per_class_onset[labels[i]].append(on_err)

    lines = [f'{"class":<20}{"n":>5}{"stft_loss":>12}{"onset_err_ms":>14}']
    all_stft, all_onset = [], []
    for c in sorted(per_class_stft):
        s = per_class_stft[c]; o = per_class_onset[c]
        all_stft += s; all_onset += o
        lines.append(f'{c:<20}{len(s):>5}{np.mean(s):>12.4f}{np.mean(o):>14.2f}')
    lines.append('-' * 51)
    lines.append(f'{"OVERALL":<20}{len(all_stft):>5}'
                 f'{np.mean(all_stft):>12.4f}{np.mean(all_onset):>14.2f}')
    text = '\n'.join(lines)
    with open(out_path, 'w') as f:
        f.write(text + '\n')
    print('\n' + text + '\n')
    print(f'Saved  {out_path}')
    return np.mean(all_stft)


def plot_recon_waveforms(inputs, recons, labels, filenames, picks, out_path,
                         sr=22050, zoom_s=0.5):
    """Overlay orig vs recon, zoomed on the first zoom_s seconds (the attack)."""
    n = len(picks)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.2 * n), squeeze=False)
    zoom = int(zoom_s * sr)
    t = np.arange(zoom) / sr
    for r, i in enumerate(picks):
        ax = axes[r, 0]
        ax.plot(t, inputs[i].numpy()[:zoom], lw=0.6, color='steelblue', label='orig', alpha=0.8)
        ax.plot(t, recons[i].numpy()[:zoom], lw=0.6, color='tomato',    label='recon', alpha=0.8)
        ax.set_title(f'[{labels[i]}] {filenames[i]}  (first {zoom_s}s)', fontsize=8)
        ax.set_xlabel('Time (s)', fontsize=7); ax.set_ylabel('Amp', fontsize=7)
        ax.tick_params(labelsize=6); ax.legend(fontsize=7, loc='upper right')
    plt.suptitle('One-shot reconstruction — attack region', fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=120, bbox_inches='tight'); plt.close(fig)
    print(f'Saved  {out_path}')


def plot_recon_spectrograms(inputs, recons, labels, filenames, picks, out_path,
                            n_fft=1024, hop=256):
    n = len(picks)
    fig = plt.figure(figsize=(14, 3 * n))
    gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.5, wspace=0.35)
    for r, i in enumerate(picks):
        S_in = log_magnitude_stft(inputs[i].numpy(), n_fft, hop)
        S_re = log_magnitude_stft(recons[i].numpy(), n_fft, hop)
        diff = S_in - S_re
        vmin, vmax = min(S_in.min(), S_re.min()), max(S_in.max(), S_re.max())
        for col, (S, title, cmap, v0, v1) in enumerate([
            (S_in, f'Input [{labels[i]}] {filenames[i]}', 'magma', vmin, vmax),
            (S_re, 'Reconstruction', 'magma', vmin, vmax),
            (diff, 'Difference', 'RdBu_r', -np.abs(diff).max(), np.abs(diff).max())]):
            ax = fig.add_subplot(gs[r, col])
            im = ax.imshow(S, aspect='auto', origin='lower', cmap=cmap, vmin=v0, vmax=v1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=8)
            ax.set_xlabel('Frame', fontsize=7); ax.set_ylabel('Freq bin', fontsize=7)
            ax.tick_params(labelsize=6)
    plt.suptitle('One-shot reconstruction — log-magnitude STFT', fontsize=10)
    fig.savefig(out_path, dpi=120, bbox_inches='tight'); plt.close(fig)
    print(f'Saved  {out_path}')


def save_recon_audio(inputs, recons, labels, filenames, picks, out_dir, sr=22050):
    os.makedirs(out_dir, exist_ok=True)
    for row, i in enumerate(picks):
        stem = os.path.splitext(filenames[i])[0]
        tag = f'{row:03d}_{labels[i]}_{stem}'
        for suffix, w in [('orig', inputs[i].numpy()), ('recon', recons[i].numpy())]:
            sf.write(os.path.join(out_dir, f'{tag}_{suffix}.wav'),
                     w / (np.abs(w).max() + 1e-8), sr)
    print(f'Saved  {len(picks)} orig/recon pairs -> {out_dir}/')


# ---------------------------------------------------------------------------
# generation (temporal-latent aware — NOT i.i.d. prior sampling)
# ---------------------------------------------------------------------------

def encode_hit(model, dataset, idx, device):
    """Encode one hit -> mu (D, T_lat) on CPU."""
    _, _, y = dataset[idx]
    with torch.no_grad():
        mu, _, _ = model.encode(y.float().unsqueeze(0).to(device))
    return mu.squeeze(0).cpu()


def decode_seq(model, z, device):
    with torch.no_grad():
        return model.decode(z.unsqueeze(0).to(device)).squeeze().cpu().numpy()


def interpolate_hits(model, dataset, labels, picks, device, out_dir,
                     n_steps=7, sr=22050, n_fft=1024, hop=256):
    """slerp between consecutive picked hits on the latent sequence."""
    os.makedirs(out_dir, exist_ok=True)
    zs = [(encode_hit(model, dataset, i, device), labels[i]) for i in picks[:4]]
    if len(zs) < 2:
        print('  interpolate: need >=2 hits, skipping'); return
    ts = np.linspace(0, 1, n_steps)
    for p in range(len(zs) - 1):
        (z1, l1), (z2, l2) = zs[p], zs[p + 1]
        specs, waves = [], []
        for t in ts:
            w = decode_seq(model, slerp(z1, z2, torch.tensor(float(t))), device)
            waves.append(w); specs.append(log_magnitude_stft(w, n_fft, hop))
        vmin, vmax = min(s.min() for s in specs), max(s.max() for s in specs)
        fig, axes = plt.subplots(1, n_steps, figsize=(2.1 * n_steps, 3), sharey=True)
        for col, (t, s) in enumerate(zip(ts, specs)):
            axes[col].imshow(s, aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
            axes[col].set_xticks([]); axes[col].set_yticks([])
            axes[col].set_title(f't={t:.2f}', fontsize=7)
        fig.suptitle(f'slerp: [{l1}] -> [{l2}]', fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'interp_{p:02d}_{l1}_to_{l2}.png'), dpi=120)
        plt.close(fig)
        for t, w in zip(ts, waves):
            sf.write(os.path.join(out_dir, f'interp_{p:02d}_{l1}_to_{l2}_t{t:.2f}.wav'),
                     w / (np.abs(w).max() + 1e-8), sr)
    print(f'Saved  interpolations -> {out_dir}/')


def generate(model, dataset, picks, device, out_dir, T_lat, latent_dim,
             sr=22050, n_fft=1024, hop=256, sigma=0.5, seed=0):
    """Generate new hits without i.i.d. prior sampling.

    (a) perturb: real hit latent + N(0, sigma) noise  (stays on the manifold)
    (b) decay : random latent vector held at t=0 and decayed over frames
                (imposes the attack->decay temporal structure by construction)
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    rows = []

    for k, i in enumerate(picks[:4]):
        z = encode_hit(model, dataset, i, device)
        rows.append((f'perturb_{k}', decode_seq(model, z + sigma * torch.randn_like(z), device)))

    env = torch.exp(-torch.linspace(0, 6, T_lat)).view(1, T_lat)  # (1, T_lat)
    for k in range(3):
        base = torch.randn(latent_dim, 1)
        rows.append((f'decay_{k}', decode_seq(model, base * env, device)))

    for name, w in rows:
        sf.write(os.path.join(out_dir, name + '.wav'), w / (np.abs(w).max() + 1e-8), sr)

    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(13, 1.9 * n))
    zoom = int(0.5 * sr)
    for r, (name, w) in enumerate(rows):
        axes[r, 0].plot(np.arange(zoom) / sr, w[:zoom], lw=0.5, color='seagreen')
        axes[r, 0].set_title(name + ' (first 0.5s)', fontsize=8); axes[r, 0].tick_params(labelsize=6)
        axes[r, 1].imshow(log_magnitude_stft(w, n_fft, hop), aspect='auto',
                          origin='lower', cmap='magma')
        axes[r, 1].set_title('log-STFT (5s)', fontsize=8); axes[r, 1].tick_params(labelsize=6)
    fig.suptitle('Generated hits (manifold perturbation + decay-envelope)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'generated.png'), dpi=120)
    plt.close(fig)
    print(f'Saved  {n} generated hits -> {out_dir}/')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(config, resume, n_samples, out_dir, seed=42, max_items=256):
    device = get_device()
    print(f'Using device: {device}')
    os.makedirs(out_dir, exist_ok=True)

    model, data_loader = load_model_and_data(config, resume, device, seed=seed)
    sr = config['data_loader']['args'].get('sr', 22050)
    chunk_size = config['data_loader']['args'].get('chunk_size', 110080)
    latent_dim = config['arch']['args']['latent_dim']
    roots = config['data_loader']['args']['data_dir']
    if isinstance(roots, str):
        roots = [roots]
    with torch.no_grad():
        T_lat = model.encode(torch.zeros(1, 1, chunk_size).to(device))[0].shape[2]
    print(f'chunk={chunk_size}  T_lat={T_lat}  latent_dim={latent_dim}  sr={sr}')

    print('Collecting reconstructions…')
    mu, logvar, inputs, recons, labels, filenames = collect(
        model, data_loader, roots, device, max_items=max_items)
    print(f'  {inputs.size(0)} hits, {len(set(labels))} classes')
    print(f'  input range [{inputs.min():.3f}, {inputs.max():.3f}]  '
          f'recon range [{recons.min():.3f}, {recons.max():.3f}]')

    picks = pick_per_class(labels, n_samples, seed=seed)

    recon_metrics(inputs, recons, labels, os.path.join(out_dir, 'recon_metrics.txt'), sr=sr)
    plot_recon_waveforms(inputs, recons, labels, filenames, picks,
                         os.path.join(out_dir, 'recon_waveforms.png'), sr=sr)
    plot_recon_spectrograms(inputs, recons, labels, filenames, picks,
                            os.path.join(out_dir, 'recon_spectrograms.png'))
    save_recon_audio(inputs, recons, labels, filenames, picks,
                     os.path.join(out_dir, 'audio'), sr=sr)

    plot_latent_pca(mu, labels, os.path.join(out_dir, 'latent_pca.png'))
    plot_kl_per_dim(mu, logvar, os.path.join(out_dir, 'kl_per_dim.png'),
                    free_bits=config['trainer'].get('free_bits', None))

    print('\nGeneration (temporal-latent aware)…')
    interpolate_hits(model, data_loader.dataset, labels, picks, device,
                     os.path.join(out_dir, 'interp'), sr=sr)
    generate(model, data_loader.dataset, picks, device,
             os.path.join(out_dir, 'gen'), T_lat, latent_dim, sr=sr)

    print('\nDone. Outputs in', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='One-shot RawAudioVAE evaluation')
    parser.add_argument('-r', '--resume', required=True, help='checkpoint .pth')
    parser.add_argument('-c', '--config', default=None,
                        help='config json (default: config.json beside checkpoint)')
    parser.add_argument('-d', '--device', default=None)
    parser.add_argument('-n', '--n_samples', type=int, default=6,
                        help='hits to visualise / save (one per class, round-robin)')
    parser.add_argument('-o', '--out_dir', default='eval/drums_run')
    parser.add_argument('-m', '--max_items', type=int, default=256,
                        help='number of hits to run through the model for metrics/PCA')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.out_dir == 'eval/drums_run':
        import re as _re
        ep = _re.search(r'epoch(\d+)', args.resume)
        args.out_dir = f'eval/drums_epoch{ep.group(1)}' if ep else 'eval/drums_run'

    config = ConfigParser(parser)
    main(config, args.resume, args.n_samples, args.out_dir,
         seed=args.seed, max_items=args.max_items)
