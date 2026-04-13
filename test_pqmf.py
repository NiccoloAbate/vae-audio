"""
test_pqmf.py — listen to PQMF reconstruction quality.

Usage:
    python3 test_pqmf.py                          # uses default test file
    python3 test_pqmf.py path/to/audio.wav        # use your own file
    python3 test_pqmf.py --n_bands 16 --taps 128  # override PQMF params

Outputs to pqmf_test/:
    original.wav          — input (resampled to sr if needed)
    reconstructed.wav     — after PQMF analysis → synthesis
    difference.wav        — error signal (scaled for audibility)
    subbands.png          — spectrogram of each subband
"""

import argparse
import os
import sys

import numpy as np
import torch
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from model.model import PQMF


DEFAULT_FILE = (
    "dataset/medley-solos/organized_reeds/trainingdata/"
    "tenor saxophone/Medley-solos-DB_training-5_8aff5fe1-620a-52b0-fae9-b39a4c4dea1c.wav"
)
SR = 22050


def run_test(audio_path, n_bands, taps, cutoff_ratio, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Load and resample
    audio, sr_orig = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr_orig != SR:
        audio = librosa.resample(audio, orig_sr=sr_orig, target_sr=SR)
        print(f"Resampled {sr_orig} Hz → {SR} Hz")

    audio = audio.astype(np.float32)
    print(f"Audio: {len(audio)/SR:.2f}s  ({len(audio)} samples at {SR} Hz)")

    # Instantiate PQMF
    pqmf = PQMF(n_bands=n_bands, taps=taps, cutoff_ratio=cutoff_ratio)
    delay_ms = (taps // 2) / SR * 1000
    print(f"PQMF:  N={n_bands}, taps={taps}, cutoff={cutoff_ratio}, "
          f"filter delay={delay_ms:.1f} ms")

    # Forward pass
    x = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)   # (1, 1, T)
    with torch.no_grad():
        subbands = pqmf.analysis(x)                          # (1, N, T//N)
        x_recon  = pqmf.synthesis(subbands)                  # (1, 1, T)

    orig  = x.squeeze().numpy()
    recon = x_recon.squeeze().numpy()

    # Align lengths (synthesis may differ by a sample or two)
    L = min(len(orig), len(recon))
    orig, recon = orig[:L], recon[:L]

    # Metrics
    err     = orig - recon
    sig_pow = np.mean(orig ** 2)
    err_pow = np.mean(err ** 2)
    snr     = 10 * np.log10(sig_pow / (err_pow + 1e-12))
    rms_err = np.sqrt(err_pow)
    rms_sig = np.sqrt(sig_pow)
    print(f"\nSNR:        {snr:.1f} dB")
    print(f"RMS signal: {rms_sig:.4f}")
    print(f"RMS error:  {rms_err:.4f}  ({rms_err/rms_sig*100:.2f}% of signal)")

    # Save audio
    sf.write(os.path.join(out_dir, "original.wav"),      orig,  SR)
    sf.write(os.path.join(out_dir, "reconstructed.wav"), recon, SR)

    # Scale error for audibility (bring up to same RMS as original)
    if rms_err > 0:
        err_scaled = err * (rms_sig / rms_err)
    else:
        err_scaled = err
    sf.write(os.path.join(out_dir, "difference_amplified.wav"), err_scaled, SR)
    print(f"\nSaved to {out_dir}/")
    print("  original.wav              — input")
    print("  reconstructed.wav         — PQMF roundtrip")
    print(f"  difference_amplified.wav  — error signal amplified {rms_sig/max(rms_err,1e-9):.0f}x")

    # Subband spectrograms
    subs = subbands.squeeze(0).numpy()   # (N, T//N)
    n_cols = min(n_bands, 8)
    n_rows = (n_bands + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2))
    axes = np.array(axes).flatten()
    for k in range(n_bands):
        s = subs[k]
        D = librosa.amplitude_to_db(np.abs(librosa.stft(s, n_fft=128, hop_length=32)),
                                    ref=np.max)
        axes[k].imshow(D, aspect='auto', origin='lower', vmin=-60, vmax=0, cmap='magma')
        axes[k].set_title(f'band {k}', fontsize=8)
        axes[k].axis('off')
    for k in range(n_bands, len(axes)):
        axes[k].axis('off')
    fig.suptitle(f'PQMF subbands  (N={n_bands}, taps={taps}, SNR={snr:.1f} dB)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "subbands.png"), dpi=120)
    plt.close()
    print("  subbands.png              — subband spectrograms")

    # Original vs reconstructed waveform overlay
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    t = np.arange(L) / SR
    axes[0].plot(t, orig,  lw=0.5, label='original')
    axes[0].plot(t, recon, lw=0.5, alpha=0.7, label='reconstructed')
    axes[0].legend(fontsize=8); axes[0].set_ylabel('amplitude')
    axes[0].set_title(f'PQMF reconstruction  (SNR={snr:.1f} dB)')
    axes[1].plot(t, err, lw=0.5, color='red', label='error')
    axes[1].set_ylabel('error'); axes[1].set_xlabel('time (s)')
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "waveform.png"), dpi=120)
    plt.close()
    print("  waveform.png              — original vs reconstructed overlay")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PQMF reconstruction quality')
    parser.add_argument('audio', nargs='?', default=DEFAULT_FILE,
                        help='input audio file (default: tenor sax from dataset)')
    parser.add_argument('--n_bands',     type=int,   default=16)
    parser.add_argument('--taps',        type=int,   default=128)
    parser.add_argument('--cutoff',      type=float, default=0.039,
                        help='prototype filter cutoff ratio (default: 0.039)')
    parser.add_argument('--out_dir',     default='pqmf_test',
                        help='output directory (default: pqmf_test/)')
    args = parser.parse_args()

    run_test(args.audio, args.n_bands, args.taps, args.cutoff, args.out_dir)
