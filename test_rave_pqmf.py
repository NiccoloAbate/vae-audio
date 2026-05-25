"""Test RAVE's PQMF algorithm (functions ported verbatim from RAVE/rave/pqmf.py)."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import fmin
from scipy.signal import firwin, kaiserord
import soundfile as sf
import torchaudio
import glob, os


# ─── RAVE pqmf.py verbatim ────────────────────────────────────────────────

def reverse_half(x):
    mask = torch.ones_like(x)
    mask[..., 1::2, ::2] = -1
    return x * mask

def center_pad_next_pow_2(x):
    next_2 = 2**math.ceil(math.log2(x.shape[-1]))
    pad = next_2 - x.shape[-1]
    return nn.functional.pad(x, (pad // 2, pad // 2 + int(pad % 2)))

def get_qmf_bank(h, n_band):
    k = torch.arange(n_band).reshape(-1, 1)
    N = h.shape[-1]
    t = torch.arange(-(N // 2), N // 2 + 1)
    p = (-1)**k * math.pi / 4
    mod = torch.cos((2 * k + 1) * math.pi / (2 * n_band) * t + p)
    hk = 2 * h * mod
    return hk

def kaiser_filter(wc, atten, N=None):
    N_, beta = kaiserord(atten, wc / np.pi)
    N_ = 2 * (N_ // 2) + 1
    N = N if N is not None else N_
    h = firwin(N, wc, window=('kaiser', beta), scale=False, fs=2*np.pi)
    return h

def loss_wc(wc, atten, M, N):
    h = kaiser_filter(wc, atten, N)
    g = np.convolve(h, h[::-1], "full")
    g = abs(g[g.shape[-1] // 2::2 * M][1:])
    return np.max(g)

def get_prototype(atten, M, N=None):
    wc = fmin(lambda w: loss_wc(w, atten, M, N), 1 / M, disp=0)[0]
    return kaiser_filter(wc, atten, N)

def classic_forward(x, hk):
    x = F.conv1d(
        x,
        hk.unsqueeze(1),
        stride=hk.shape[0],
        padding=hk.shape[-1] // 2,
    )[..., :-1]
    return x

def classic_inverse(x, hk):
    hk = hk.flip(-1)
    y = torch.zeros(*x.shape[:2], hk.shape[0] * x.shape[-1]).to(x)
    y[..., ::hk.shape[0]] = x * hk.shape[0]
    y = F.conv1d(y, hk.unsqueeze(0), padding=hk.shape[-1] // 2)[..., 1:]
    return y

def pqmf_forward(x, hk):
    x = classic_forward(x, hk)
    x = reverse_half(x)
    return x

def pqmf_inverse(x, hk):
    x = reverse_half(x)
    return classic_inverse(x, hk)


# ─── Test ─────────────────────────────────────────────────────────────────

SR = 22050
N_SAMPLES = 65536

audio_files = sorted(glob.glob(
    '/home/niccoloabate/CCRMA/EE269/final/vae-audio/data/medley_reeds/**/*.wav',
    recursive=True))
audio_path = audio_files[0] if audio_files else None

if audio_path:
    wav, orig_sr = sf.read(audio_path, always_2d=False)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if orig_sr != SR:
        wav_t = torch.from_numpy(wav.copy()).float().unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, orig_sr, SR)
        wav = wav_t.squeeze(0).numpy()
    wav = wav[:N_SAMPLES]
    x = torch.from_numpy(wav.copy()).float().unsqueeze(0).unsqueeze(0)
    print(f"Audio: {os.path.basename(audio_path)}  ({x.shape[-1]} samples)\n")
else:
    x = torch.randn(1, 1, N_SAMPLES)
    print("Using random noise\n")

print(f"{'Label':45s}  {'filter_len':>10}  {'SNR (dB)':>10}")
print("-" * 70)

for atten in [80, 100, 120]:
    for n_band in [16]:
        h = get_prototype(atten, n_band)
        ht = torch.from_numpy(h).float()
        hk = get_qmf_bank(ht, n_band)
        hk = center_pad_next_pow_2(hk)

        with torch.no_grad():
            xa = pqmf_forward(x, hk)
            yr = pqmf_inverse(xa, hk)

        n = min(x.shape[-1], yr.shape[-1])
        sig = x[..., :n]
        err = sig - yr[..., :n]
        snr = 10 * np.log10((sig**2).mean().item() / (err**2).mean().item() + 1e-15)
        label = f"RAVE PQMF (atten={atten}, N={n_band})"
        print(f"{label:45s}  {hk.shape[-1]:>10}  {snr:>10.1f}")
