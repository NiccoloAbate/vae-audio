#!/usr/bin/env python
"""
DSP-perspective visualisation of the RawAudioVAE decoder.

Three analyses:
  1. Latent basis functions  — decode each unit latent dim, show waveform + spectrum
  2. Output FIR responses    — frequency response of each of the 32 final-layer filters
  3. Decoder impulse response — temporal spread of a single latent-frame pulse

Usage:
    python visualize_decoder.py -r saved/models/RawAudioVAE/<run>/checkpoint-epochN.pth
"""

import argparse, os, sys
import numpy as np
import torch
import torchaudio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model.model as module_arch
from parse_config import ConfigParser

SR         = 22050
CHUNK_SIZE = 16384
T_LAT      = 64      # CHUNK_SIZE / (4^4)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def db(x, eps=1e-8):
    return 20 * np.log10(np.abs(x) + eps)

def freq_response(kernel_1d, n=8192):
    """Magnitude frequency response (dB) of a 1-D FIR kernel."""
    h = np.zeros(n)
    h[:len(kernel_1d)] = kernel_1d
    H = np.fft.rfft(h)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    return freqs, db(H)

def decode_z(model, device, z):
    with torch.no_grad():
        return model.decode(z.to(device)).squeeze(0).squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# 1. Latent basis functions
# ---------------------------------------------------------------------------

def plot_basis_functions(model, device, out_dir, n_dims=32, scale=3.0):
    """
    For each latent dim d, set z[d] = scale everywhere (others = 0) and decode.
    Shows what waveform + spectrum each latent dimension 'synthesises'.
    scale=3.0 ≈ 3σ — a strong but not extreme activation.
    """
    n_cols = 4
    n_rows = (n_dims + n_cols - 1) // n_cols

    fig_wave, axes_w = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 2.5))
    fig_spec, axes_s = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 2.5))

    for d in range(n_dims):
        z = torch.zeros(1, n_dims, T_LAT)
        z[0, d, :] = scale
        y = decode_z(model, device, z)

        ax_w = axes_w.flat[d]
        ax_s = axes_s.flat[d]

        t = np.arange(len(y)) / SR
        ax_w.plot(t * 1000, y, linewidth=0.6)
        ax_w.set_title(f'dim {d}', fontsize=8)
        ax_w.set_ylim(-1.05, 1.05)
        ax_w.set_xlabel('ms', fontsize=7)
        ax_w.tick_params(labelsize=6)

        # Log-magnitude spectrum
        Y  = np.fft.rfft(y * np.hanning(len(y)))
        ff = np.fft.rfftfreq(len(y), d=1.0 / SR)
        ax_s.semilogx(ff[1:], db(Y[1:]), linewidth=0.7)
        ax_s.set_title(f'dim {d}', fontsize=8)
        ax_s.set_xlim(20, SR / 2)
        ax_s.set_ylim(-80, 10)
        ax_s.set_xlabel('Hz', fontsize=7)
        ax_s.tick_params(labelsize=6)

    # Hide unused panels
    for ax in list(axes_w.flat[n_dims:]) + list(axes_s.flat[n_dims:]):
        ax.set_visible(False)

    for fig, name in [(fig_wave, 'basis_waveforms.png'), (fig_spec, 'basis_spectra.png')]:
        fig.suptitle(f'Latent basis functions  (z[d]=+{scale}, others=0)', fontsize=10)
        fig.tight_layout()
        p = out_dir / name
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f'Saved  {p}')

    # Also save audio for first 8 dims
    audio_dir = out_dir / 'basis_audio'
    audio_dir.mkdir(exist_ok=True)
    for d in range(min(8, n_dims)):
        z = torch.zeros(1, n_dims, T_LAT)
        z[0, d, :] = scale
        y = decode_z(model, device, z)
        y_t = torch.tensor(y).unsqueeze(0)
        torchaudio.save(str(audio_dir / f'dim_{d:02d}.wav'), y_t, SR)
    print(f'Saved  basis audio → {audio_dir}/')


# ---------------------------------------------------------------------------
# 2. Output FIR frequency responses
# ---------------------------------------------------------------------------

def plot_output_fir(model, out_dir):
    """
    The final Conv1d(32 → 1, kernel_size=7) is a bank of 32 seven-tap FIR filters
    summed to produce the output sample.  Plot each filter's frequency response.
    """
    # decoder[-2] is the final Conv1d (before Tanh)
    final_conv = model.decoder[-2]   # Conv1d(32, 1, 7)
    # weight shape: (out_ch=1, in_ch=32, kernel=7)
    kernels = final_conv.weight.detach().cpu().numpy()[0]   # (32, 7)

    n_cols = 4
    n_rows = (kernels.shape[0] + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 2.5))

    for d, kernel in enumerate(kernels):
        freqs, H_db = freq_response(kernel)
        ax = axes.flat[d]
        ax.semilogx(freqs[1:], H_db[1:], linewidth=0.8)
        ax.set_title(f'ch {d}  (max={kernel.max():.3f})', fontsize=8)
        ax.set_xlim(20, SR / 2)
        ax.set_ylim(-60, 20)
        ax.axhline(-3, color='red', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Hz', fontsize=7)
        ax.tick_params(labelsize=6)

    for ax in axes.flat[len(kernels):]:
        ax.set_visible(False)

    fig.suptitle('Output layer FIR frequency responses\n'
                 'Final Conv1d(32→1, k=7): each channel\'s 7-tap filter → summed to output',
                 fontsize=10)
    fig.tight_layout()
    p = out_dir / 'output_fir_responses.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'Saved  {p}')

    # Also plot the effective combined kernel (sum of all 32 filters weighted by bias)
    combined = kernels.sum(axis=0)   # rough combined impulse
    freqs, H_db = freq_response(combined)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.stem(np.arange(len(combined)), combined, markerfmt='C0o', basefmt='k-')
    ax1.set_title('Sum of output FIR taps (impulse response sketch)', fontsize=9)
    ax1.set_xlabel('Tap')
    ax2.semilogx(freqs[1:], H_db[1:], linewidth=1.2, color='C0')
    ax2.set_title('Frequency response of summed output FIR', fontsize=9)
    ax2.set_xlim(20, SR / 2); ax2.set_ylim(-60, 20)
    ax2.set_xlabel('Hz')
    ax2.axhline(-3, color='red', linewidth=0.8, linestyle='--', label='-3 dB')
    ax2.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / 'output_fir_combined.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'Saved  {p}')


# ---------------------------------------------------------------------------
# 3. Decoder impulse response
# ---------------------------------------------------------------------------

def plot_impulse_response(model, device, out_dir, n_dims=32):
    """
    Pulse a single latent frame (middle of the sequence) and decode.
    Shows how far in time a single latent event 'spreads' in the output waveform
    — i.e. the effective temporal support of the decoder filter stack.
    """
    mid = T_LAT // 2
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    for plot_idx, d in enumerate([0, 1, 2, 3]):
        z = torch.zeros(1, n_dims, T_LAT)
        z[0, d, mid] = 3.0   # single pulse at middle frame
        y = decode_z(model, device, z)

        t_ms = np.arange(len(y)) / SR * 1000
        ax = axes[plot_idx]
        ax.plot(t_ms, y, linewidth=0.7)
        ax.axvline((mid / T_LAT) * (CHUNK_SIZE / SR) * 1000,
                   color='red', linewidth=0.8, linestyle='--', label='pulse position')
        ax.set_title(f'Impulse response — latent dim {d}  (single frame pulse at t={mid})',
                     fontsize=9)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Decoder impulse responses\n'
                 'A single active latent frame → how far does it spread in the output?',
                 fontsize=10)
    fig.tight_layout()
    p = out_dir / 'decoder_impulse_response.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'Saved  {p}')

    # Save audio of impulse response for dim 0
    z = torch.zeros(1, n_dims, T_LAT)
    z[0, 0, mid] = 3.0
    y = decode_z(model, device, z)
    torchaudio.save(str(out_dir / 'impulse_response_dim0.wav'),
                    torch.tensor(y).unsqueeze(0), SR)
    print(f'Saved  {out_dir}/impulse_response_dim0.wav')


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSP visualisation of RawAudioVAE decoder')
    parser.add_argument('-r', '--resume', required=True, help='path to checkpoint (.pth)')
    parser.add_argument('-c', '--config', default=None)
    parser.add_argument('-d', '--device', default=None)
    parser.add_argument('--out', default=None, help='output directory')
    args = parser.parse_args()

    cfg    = ConfigParser(parser)
    device = (torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))
    print(f'Device: {device}')

    model = cfg.initialize('arch', module_arch)
    ckpt  = torch.load(args.resume, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()

    run_id  = Path(args.resume).parent.name
    out_dir = Path(args.out) if args.out else Path('eval') / run_id / 'dsp'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output: {out_dir}\n')

    print('1. Latent basis functions…')
    plot_basis_functions(model, device, out_dir)

    print('\n2. Output FIR frequency responses…')
    plot_output_fir(model, out_dir)

    print('\n3. Decoder impulse responses…')
    plot_impulse_response(model, device, out_dir)

    print('\nDone.')
