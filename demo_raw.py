#!/usr/bin/env python
"""
Interactive latent-space interpolation demo for RawAudioVAE.

Upload 2–3 audio files, drag the slider to morph between them in latent space.

Usage:
    python demo_raw.py -r saved/models/RawAudioVAE/<run>/model_best.pth
"""
import argparse
import os
import sys
import numpy as np
import torch
import torchaudio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model.model as module_arch
from parse_config import ConfigParser

SR         = 22050
CHUNK_SIZE = 16384
CROSSFADE  = 256


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

def load_wav(path, sr=SR):
    """Load an audio file, resample if needed, return mono float32 numpy (T,)."""
    y, file_sr = torchaudio.load(path)
    if file_sr != sr:
        y = torchaudio.functional.resample(y, file_sr, sr)
    if y.shape[0] > 1:
        y = y.mean(0, keepdim=True)
    return y.squeeze(0).numpy()


def encode_file(model, device, path, chunk_size=CHUNK_SIZE):
    """Load a wav and return a list of per-chunk latent means, each (D, T_lat)."""
    y        = load_wav(path)
    n_chunks = len(y) // chunk_size
    if n_chunks == 0:
        raise ValueError(f'Audio too short — need at least {chunk_size / SR:.2f}s')

    latents = []
    model.eval()
    with torch.no_grad():
        for i in range(n_chunks):
            chunk = torch.tensor(y[i * chunk_size:(i + 1) * chunk_size],
                                 dtype=torch.float32)
            x   = chunk.unsqueeze(0).unsqueeze(0).to(device)   # (1, 1, T)
            mu, _, _ = model.encode(x)
            latents.append(mu.squeeze(0).cpu())                 # (D, T_lat)
    return latents


# ---------------------------------------------------------------------------
# latent interpolation
# ---------------------------------------------------------------------------

def slerp(z1, z2, t):
    """Spherical linear interpolation between two tensors of any shape."""
    z1_n  = z1 / (z1.norm() + 1e-8)
    z2_n  = z2 / (z2.norm() + 1e-8)
    omega = torch.acos(torch.clamp((z1_n * z2_n).sum(), -1.0, 1.0))
    if omega.abs() < 1e-6:
        return (1 - t) * z1 + t * z2
    return (torch.sin((1 - t) * omega) * z1 +
            torch.sin(t * omega) * z2) / torch.sin(omega)


def interpolate_chunks(z1_list, z2_list, z3_list, t_val):
    """
    Interpolate chunk-by-chunk between two or three latent sequences.
    Files may have different lengths — uses the minimum chunk count.

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

def decode_to_audio(model, device, z_list, crossfade=CROSSFADE):
    """Decode a list of latent sequences (D, T_lat) → waveform + STFT plot array.

    A short linear crossfade is applied at chunk boundaries to suppress clicks.
    Returns (wave float32 numpy (T,), log-mag STFT numpy (F, T_frames)).
    """
    chunks = []
    with torch.no_grad():
        for z in z_list:
            y_hat = model.decode(z.unsqueeze(0).to(device))
            chunks.append(y_hat.squeeze(0).squeeze(0).cpu().numpy())

    if crossfade > 0 and len(chunks) > 1:
        fade_out = np.linspace(1, 0, crossfade)
        fade_in  = np.linspace(0, 1, crossfade)
        result   = chunks[0].copy()
        for chunk in chunks[1:]:
            blend  = result[-crossfade:] * fade_out + chunk[:crossfade] * fade_in
            result = np.concatenate([result[:-crossfade], blend, chunk[crossfade:]])
        wave = result
    else:
        wave = np.concatenate(chunks)

    wave = wave / (np.abs(wave).max() + 1e-8)

    # Log-magnitude STFT for display
    n_fft, hop = 1024, 256
    win  = torch.hann_window(n_fft)
    S    = torch.stft(torch.tensor(wave, dtype=torch.float32),
                      n_fft, hop, n_fft, win, return_complex=True)
    spec = torch.log(S.abs() + 1e-5).numpy()

    return wave.astype(np.float32), spec


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
            _cache[key]        = path
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

        z_chunks     = interpolate_chunks(z1, z2, z3, t_val)
        wave, spec   = decode_to_audio(model, device, z_chunks)

        n_files    = 3 if z3 is not None else 2
        tick_label = (f't={t_val/100:.2f}  (0=File1 · 1=File2)'
                      if n_files == 2 else
                      f't={t_val/100:.2f}  (0=File1 · 0.5=File2 · 1=File3)')

        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.imshow(spec, aspect='auto', origin='lower', cmap='magma')
        ax.set_title(f'Decoded STFT — {tick_label}', fontsize=9)
        ax.set_xlabel('Time frame')
        ax.set_ylabel('Freq bin')
        plt.tight_layout()

        return (SR, wave), fig

    with gr.Blocks(title='RawAudioVAE — Latent Interpolation') as demo:
        gr.Markdown(
            '## RawAudioVAE — Latent Space Interpolation\n'
            'Upload **2 or 3** audio files (wav/mp3/etc.), then drag the slider '
            'to morph between them in latent space using slerp.  '
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
            out_spec  = gr.Plot(label='Log-magnitude STFT')

        btn.click(fn=generate,
                  inputs=[f1, f2, f3, slider],
                  outputs=[out_audio, out_spec])

    return demo


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RawAudioVAE latent interpolation demo')
    parser.add_argument('-r', '--resume', required=True, help='path to checkpoint (.pth)')
    parser.add_argument('-c', '--config', default=None,
                        help='config json (inferred from checkpoint dir if omitted)')
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
