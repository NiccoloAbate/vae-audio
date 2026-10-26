"""
Standalone drum generator driving the latent synthesis engine (drum_engine.py).

Modes:
  sample   render one representative hit per type from the bank (a "kit")
  interp   morph between two bank hits over N steps (this is the velocity /
           automation primitive: t is what you'd drive from MIDI velocity)
  perturb  N on-manifold variations of one bank hit

Each mode writes .wav files and a summary spectrogram figure.

Usage:
  python generate_drums.py sample  -b bank/drums_bank.pt -r <ckpt> -o out/kit
  python generate_drums.py interp  -b bank/drums_bank.pt -r <ckpt> --a 0 --b 5 --steps 7
  python generate_drums.py perturb -b bank/drums_bank.pt -r <ckpt> --a 0 --sigma 0.5 -n 6
"""
import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf

from drum_engine import DrumEngine


def logspec(w, n_fft=1024, hop=256):
    t = torch.tensor(w, dtype=torch.float32)
    S = torch.stft(t, n_fft, hop, n_fft, torch.hann_window(n_fft), return_complex=True)
    return torch.log(S.abs() + 1e-5).numpy()


def montage(waves, titles, sr, out_png, suptitle):
    n = len(waves)
    fig, axes = plt.subplots(n, 2, figsize=(13, 1.8 * n), squeeze=False)
    zoom = int(0.5 * sr)
    for r, (w, title) in enumerate(zip(waves, titles)):
        axes[r, 0].plot(np.arange(min(zoom, len(w))) / sr, w[:zoom], lw=0.5, color='seagreen')
        axes[r, 0].set_title(title + ' (first 0.5s)', fontsize=8)
        axes[r, 0].set_ylim(-1.05, 1.05); axes[r, 0].tick_params(labelsize=6)
        axes[r, 1].imshow(logspec(w), aspect='auto', origin='lower', cmap='magma')
        axes[r, 1].set_title('log-STFT', fontsize=8); axes[r, 1].tick_params(labelsize=6)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f'Saved  {out_png}')


def save_wavs(waves, names, sr, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for w, name in zip(waves, names):
        sf.write(os.path.join(out_dir, name + '.wav'), w, sr)
    print(f'Saved  {len(waves)} wavs -> {out_dir}/')


def mode_sample(eng, args):
    """One representative hit per type = a starter kit."""
    seen, waves, names, titles = set(), [], [], []
    for i, t in enumerate(eng.bank['types']):
        if t in seen:
            continue
        seen.add(t)
        waves.append(eng.render(eng.z(i)))
        names.append(f'{len(names):02d}_{t}')
        titles.append(f'[{t}]')
    save_wavs(waves, names, eng.sr, args.out)
    montage(waves, titles, eng.sr, os.path.join(args.out, 'kit.png'),
            'Kit — one representative hit per type')


def mode_interp(eng, args):
    za, zb = eng.z(args.a), eng.z(args.b)
    ta, tb = eng.bank['types'][args.a], eng.bank['types'][args.b]
    fn = eng.slerp if args.slerp else eng.lerp
    ts = np.linspace(0, 1, args.steps)
    waves, names, titles = [], [], []
    for t in ts:
        w = eng.render(fn(za, zb, torch.tensor(float(t))))
        waves.append(w)
        names.append(f'interp_{args.a}_{args.b}_t{t:.2f}')
        titles.append(f't={t:.2f}  (vel {int(t*127):3d})')
    save_wavs(waves, names, eng.sr, args.out)
    montage(waves, titles, eng.sr, os.path.join(args.out, f'interp_{ta}_to_{tb}.png'),
            f'Morph [{ta}] (#{args.a}) -> [{tb}] (#{args.b})   — t is the velocity/automation axis')


def mode_perturb(eng, args):
    z = eng.z(args.a)
    t = eng.bank['types'][args.a]
    g = torch.Generator().manual_seed(args.seed)
    waves = [eng.render(z)] + [eng.render(eng.perturb(z, args.sigma, g)) for _ in range(args.n)]
    names = [f'{t}_original'] + [f'{t}_var{i}' for i in range(args.n)]
    titles = [f'[{t}] original'] + [f'[{t}] var {i} (sigma={args.sigma})' for i in range(args.n)]
    save_wavs(waves, names, eng.sr, args.out)
    montage(waves, titles, eng.sr, os.path.join(args.out, f'perturb_{t}.png'),
            f'Perturbations of [{t}] (#{args.a})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['sample', 'interp', 'perturb'])
    ap.add_argument('-b', '--bank', required=True)
    ap.add_argument('-r', '--resume', required=True, help='checkpoint for the decoder')
    ap.add_argument('--decoder_pt', default=None, help='optional traced decoder.pt')
    ap.add_argument('-o', '--out', default='out/drums')
    ap.add_argument('--a', type=int, default=0)
    ap.add_argument('--b', type=int, default=1)
    ap.add_argument('--steps', type=int, default=7)
    ap.add_argument('--slerp', action='store_true', help='spherical interp (default: linear)')
    ap.add_argument('--sigma', type=float, default=0.5)
    ap.add_argument('-n', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    eng = DrumEngine(checkpoint=args.resume, decoder_pt=args.decoder_pt, bank=args.bank)
    print(f'Engine: sr={eng.sr} chunk={eng.chunk_size} '
          f'bank={eng.bank["latents"].shape} types={sorted(set(eng.bank["types"]))}')
    {'sample': mode_sample, 'interp': mode_interp, 'perturb': mode_perturb}[args.mode](eng, args)


if __name__ == '__main__':
    main()
