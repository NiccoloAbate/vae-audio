#!/usr/bin/env python
"""
Interactive latent-space interpolation demo for SpecVAE.

Upload 2–3 audio files, drag the slider to morph between them in latent space.

Usage:
    python demo.py -r saved/models/SpecVAE/<run>/model_best.pth
"""
import argparse
import os
import sys
import numpy as np
import torch
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model.model as module_arch
from parse_config import ConfigParser

SR     = 22050
N_FFT  = 2048
HOP    = 735
N_MELS = 64
CHUNK  = 15      # time frames per VAE chunk


# ---------------------------------------------------------------------------
# preprocessing  (mirrors the training pipeline exactly)
# ---------------------------------------------------------------------------

def wav_to_mel_norm(path):
    """Load a wav and return a normalised dB mel spectrogram (shape: N_MELS × T)."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    y -= y.mean()                                         # Zscore (no sigma)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2)
    mel_db = librosa.power_to_db(mel, ref=np.max)        # dB in (-80, 0)
    return mel_db / 40.0 + 1.0                           # NormalizeSpecDb → (-1, 1)


def encode_file(model, device, path):
    """Encode a wav to a sequence of per-chunk latent vectors (preserves temporal structure)."""
    mel_norm = wav_to_mel_norm(path)
    n_chunks = mel_norm.shape[1] // CHUNK
    if n_chunks == 0:
        raise ValueError('Audio too short — need at least 0.5 s')
    mus = []
    model.eval()
    with torch.no_grad():
        for i in range(n_chunks):
            chunk = mel_norm[:, i * CHUNK:(i + 1) * CHUNK]
            x = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            mu, _, _ = model.encode(x)
            mus.append(mu.squeeze(0).cpu())
    return mus   # list of (latent_dim,) tensors, one per 0.5s chunk


# ---------------------------------------------------------------------------
# latent interpolation
# ---------------------------------------------------------------------------

def slerp(z1, z2, t):
    """Spherical linear interpolation between two latent vectors."""
    z1_n = z1 / (z1.norm() + 1e-8)
    z2_n = z2 / (z2.norm() + 1e-8)
    omega = torch.acos(torch.clamp((z1_n * z2_n).sum(), -1.0, 1.0))
    if omega.abs() < 1e-6:
        return (1 - t) * z1 + t * z2
    return (torch.sin((1 - t) * omega) * z1 +
            torch.sin(t * omega) * z2) / torch.sin(omega)


def interpolate_chunks(z1_list, z2_list, z3_list, t_val):
    """
    Interpolate chunk-by-chunk between two (or three) latent sequences.
    Files may have different lengths — use the minimum chunk count.

    With 2 files:  t=0 → z1,  t=100 → z2
    With 3 files:  t=0 → z1,  t=50  → z2,  t=100 → z3
    """
    t = torch.tensor(float(t_val) / 100.0, dtype=torch.float32)

    if z3_list is None:
        n = min(len(z1_list), len(z2_list))
        return [slerp(z1_list[i], z2_list[i], t) for i in range(n)]

    n = min(len(z1_list), len(z2_list), len(z3_list))
    if t <= 0.5:
        return [slerp(z1_list[i], z2_list[i], t * 2) for i in range(n)]
    return [slerp(z2_list[i], z3_list[i], (t - 0.5) * 2) for i in range(n)]


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------

def decode_to_audio(model, device, z_list):
    """Decode a list of latent vectors → audio via Griffin-Lim.
    Concatenates all chunks into one mel spectrogram before running GL,
    avoiding phase discontinuities at chunk boundaries.
    """
    specs = []
    with torch.no_grad():
        for z in z_list:
            spec = model.decode(z.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            specs.append(spec)
    spec_full = np.concatenate(specs, axis=1)        # (N_MELS, CHUNK * n_chunks)
    mel_db    = (spec_full - 1.0) * 40.0             # denorm → dB
    mel_pow   = librosa.db_to_power(mel_db)
    y = librosa.feature.inverse.mel_to_audio(
        mel_pow, sr=SR, n_iter=256, n_fft=N_FFT, hop_length=HOP)
    y = y / (np.abs(y).max() + 1e-8)
    return y.astype(np.float32), spec_full


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def build_interface(model, device):

    # Cache encoded latents so re-dragging the slider is instant
    _cache = {}

    def get_latent(path, key):
        if path is None:
            return None
        if _cache.get(key) != path:
            _cache[key] = path
            _cache[key + '_z'] = encode_file(model, device, path)
        return _cache[key + '_z']

    def generate(f1, f2, f3, t_val):
        if f1 is None or f2 is None:
            gr.Warning('Please upload at least two audio files.')
            return None, None
        try:
            z1 = get_latent(f1, 'f1')
            z2 = get_latent(f2, 'f2')
            z3 = get_latent(f3, 'f3') if f3 is not None else None
        except Exception as e:
            gr.Warning(str(e))
            return None, None

        z_chunks = interpolate_chunks(z1, z2, z3, t_val)
        y, spec = decode_to_audio(model, device, z_chunks)

        n_files = 3 if z3 is not None else 2
        if n_files == 2:
            tick_label = f't={t_val/100:.2f}  (0=File1 · 1=File2)'
        else:
            tick_label = f't={t_val/100:.2f}  (0=File1 · 0.5=File2 · 1=File3)'

        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.imshow(spec, aspect='auto', origin='lower', cmap='magma', vmin=-1, vmax=1)
        ax.set_title(f'Decoded mel spectrogram — {tick_label}', fontsize=9)
        ax.set_xlabel('Time frame')
        ax.set_ylabel('Mel bin')
        plt.tight_layout()

        return (SR, y), fig

    with gr.Blocks(title='SpecVAE — Latent Interpolation') as demo:
        gr.Markdown(
            '## SpecVAE — Latent Space Interpolation\n'
            'Upload **2 or 3** audio files (wav/mp3/etc.), then drag the slider '
            'to morph between them in latent space using slerp. '
            'Hit **Generate** to synthesise audio at the current position.'
        )

        with gr.Row():
            f1 = gr.Audio(label='File 1', type='filepath')
            f2 = gr.Audio(label='File 2', type='filepath')
            f3 = gr.Audio(label='File 3 (optional)', type='filepath')

        slider = gr.Slider(
            minimum=0, maximum=100, value=0, step=1,
            label='Interpolation  [ 0 = File 1  ·  50 = File 2  ·  100 = File 3 (or File 2) ]')

        btn = gr.Button('Generate', variant='primary')

        with gr.Row():
            out_audio = gr.Audio(label='Interpolated audio', type='numpy', autoplay=True)
            out_spec  = gr.Plot(label='Decoded mel spectrogram')

        btn.click(fn=generate,
                  inputs=[f1, f2, f3, slider],
                  outputs=[out_audio, out_spec])

    return demo


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SpecVAE latent interpolation demo')
    parser.add_argument('-r', '--resume', required=True, help='path to checkpoint (.pth)')
    parser.add_argument('-c', '--config', default=None, help='config json (inferred if omitted)')
    parser.add_argument('-d', '--device', default=None)
    args = parser.parse_args()

    cfg = ConfigParser(parser)

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    model = cfg.initialize('arch', module_arch)
    ckpt  = torch.load(args.resume, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    print('Model loaded — launching demo…')

    iface = build_interface(model, device)
    iface.launch()
