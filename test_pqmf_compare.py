"""Compare our PQMF (with and without reverse_half) vs RAVE-style."""
import sys, os
sys.path.insert(0, '.')

import torch
import numpy as np
import soundfile as sf
import torchaudio
import glob

SR = 22050
N_SAMPLES = 65536

audio_files = sorted(glob.glob('data/medley_reeds/**/*.wav', recursive=True))
if audio_files:
    wav, orig_sr = sf.read(audio_files[0], always_2d=False)
    if wav.ndim > 1: wav = wav[:, 0]
    if orig_sr != SR:
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav.copy()).float().unsqueeze(0), orig_sr, SR
        ).squeeze(0).numpy()
    wav = wav[:N_SAMPLES]
    x = torch.from_numpy(wav.copy()).float().unsqueeze(0).unsqueeze(0)
    print(f"Audio: {os.path.basename(audio_files[0])}\n")
else:
    x = torch.randn(1, 1, N_SAMPLES)
    print("Using random noise\n")

def snr(sig, recon):
    n = min(sig.shape[-1], recon.shape[-1])
    s = sig[..., :n]
    e = s - recon[..., :n]
    return 10 * np.log10((s**2).mean().item() / (e**2).mean().item() + 1e-15)

# Import our PQMF
from model.model import PQMF, _reverse_half

print(f"{'Config':50s}  {'SNR (dB)':>10}")
print("-" * 65)

for n_bands, taps, cutoff in [(16, 128, 0.039)]:
    # Our PQMF as-is (with reverse_half)
    pqmf = PQMF(n_bands=n_bands, taps=taps, cutoff_ratio=cutoff)
    with torch.no_grad():
        xa = pqmf.analysis(x)
        yr = pqmf.synthesis(xa)
    label = f"Ours (N={n_bands}, taps={taps}, cutoff={cutoff}) + reverse_half"
    print(f"{label:50s}  {snr(x, yr):>10.1f}")

    # Without reverse_half (bypass it)
    with torch.no_grad():
        import torch.nn.functional as F
        # analysis without reverse_half
        xa2 = F.conv1d(pqmf.pad_fn(x), pqmf.analysis_filter)
        xa2 = F.conv1d(xa2, pqmf.updown_filter, stride=n_bands)
        # synthesis without reverse_half
        yr2 = F.conv_transpose1d(xa2, pqmf.updown_filter * n_bands, stride=n_bands)
        yr2 = F.conv1d(pqmf.pad_fn(yr2), pqmf.synthesis_filter)
    label = f"Ours (N={n_bands}, taps={taps}, cutoff={cutoff}) NO reverse_half"
    print(f"{label:50s}  {snr(x, yr2):>10.1f}")
