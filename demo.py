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
from sklearn.decomposition import PCA

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
    Per-chunk slerp: interpolate each chunk independently.
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


def interpolate_chunks_mean_offset(z1_list, z2_list, z3_list, t_val):
    """
    Mean + offset interpolation:
      1. Compute global mean latent per file (mean over chunks) → (latent_dim,)
      2. Slerp between the global means at position t
      3. Blend each chunk's per-file offset from its mean: δ[i] = z[i] - μ_global
      4. Recombine: z_t[i] = μ_t + (1-t)*δ_a[i] + t*δ_b[i]

    This separates global character (morphed via slerp) from local temporal
    structure (blended linearly), reducing abrupt chunk-to-chunk jumps.
    """
    t = torch.tensor(float(t_val) / 100.0, dtype=torch.float32)

    def _mean(z_list):
        return torch.stack(z_list).mean(dim=0)   # (latent_dim,)

    def _interp(za_list, zb_list, t_scalar):
        n    = min(len(za_list), len(zb_list))
        mu_a = _mean(za_list)
        mu_b = _mean(zb_list)
        mu_t = slerp(mu_a, mu_b, t_scalar)
        result = []
        for i in range(n):
            delta_a = za_list[i] - mu_a
            delta_b = zb_list[i] - mu_b
            delta_t = (1 - t_scalar) * delta_a + t_scalar * delta_b
            result.append(mu_t + delta_t)
        return result

    if z3_list is None:
        return _interp(z1_list, z2_list, t)

    if t <= 0.5:
        return _interp(z1_list, z2_list, t * 2)
    return _interp(z2_list, z3_list, (t - 0.5) * 2)


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
# latent visualisation helpers
# ---------------------------------------------------------------------------

def pool_latents(z_list):
    """Pool a list of (latent_dim,) tensors → single (latent_dim,) numpy vector (mean over chunks)."""
    return torch.stack(z_list).mean(dim=0).numpy()


def plot_latent_position(z1_list, z2_list, z3_list, z_cur_list, t_val, labels):
    """Two-panel figure:
      Left  — PCA scatter: endpoints + slerp path + current position.
      Right — Latent heatmap: interpolated latent (latent_dim × n_chunks).
    """
    p1  = pool_latents(z1_list)
    p2  = pool_latents(z2_list)
    p3  = pool_latents(z3_list) if z3_list is not None else None
    cur = pool_latents(z_cur_list)

    endpoints = [p1, p2] + ([p3] if p3 is not None else [])
    all_pts   = np.stack(endpoints + [cur])

    n_components = min(2, all_pts.shape[0], all_pts.shape[1])
    pca      = PCA(n_components=n_components)
    proj     = pca.fit_transform(all_pts)
    ep_proj  = proj[:-1]
    cur_proj = proj[-1]

    ts = np.linspace(0, 1, 60)
    path_pts = []
    for t_f in ts:
        t_t = torch.tensor(t_f, dtype=torch.float32)
        if p3 is None:
            z_path = slerp(torch.tensor(p1), torch.tensor(p2), t_t).numpy()
        else:
            if t_f <= 0.5:
                z_path = slerp(torch.tensor(p1), torch.tensor(p2), t_t * 2).numpy()
            else:
                z_path = slerp(torch.tensor(p2), torch.tensor(p3), (t_t - 0.5) * 2).numpy()
        path_pts.append(z_path)
    path_proj = pca.transform(np.stack(path_pts))

    colours = ['#4C9BE8', '#E8724C', '#4CE87A']
    fig, (ax_pca, ax_heat) = plt.subplots(1, 2, figsize=(10, 3.5))

    # --- PCA scatter ---
    ax_pca.plot(path_proj[:, 0], path_proj[:, 1],
                '--', color='#aaaaaa', linewidth=1, zorder=1)
    for i, (pt, lbl) in enumerate(zip(ep_proj, labels)):
        ax_pca.scatter(*pt, color=colours[i], s=120, zorder=3, label=lbl)
        ax_pca.annotate(lbl, pt, textcoords='offset points',
                        xytext=(6, 4), fontsize=8, color=colours[i])
    ax_pca.scatter(*cur_proj, color='white', s=160, zorder=4,
                   edgecolors='black', linewidths=1.5, marker='*',
                   label=f't={t_val/100:.2f}')
    ax_pca.set_title('Latent PCA — interpolation position', fontsize=9)
    ax_pca.set_xlabel('PC 1', fontsize=8); ax_pca.set_ylabel('PC 2', fontsize=8)
    ax_pca.tick_params(labelsize=7)

    # --- Latent heatmap (latent_dim × n_chunks) ---
    heat = torch.stack(z_cur_list).numpy().T   # (latent_dim, n_chunks)
    vabs = np.abs(heat).max() + 1e-8
    im   = ax_heat.imshow(heat, aspect='auto', origin='lower',
                          cmap='RdBu_r', vmin=-vabs, vmax=vabs)
    ax_heat.set_title('Interpolated latent (dims × chunks)', fontsize=9)
    ax_heat.set_xlabel('Chunk', fontsize=8)
    ax_heat.set_ylabel('Latent dim', fontsize=8)
    ax_heat.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


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

    def generate(f1, f2, f3, t_val, mode):
        if f1 is None or f2 is None:
            gr.Warning('Please upload at least two audio files.')
            return None, None, None
        try:
            z1 = get_latent(f1, 'f1')
            z2 = get_latent(f2, 'f2')
            z3 = get_latent(f3, 'f3') if f3 is not None else None
        except Exception as e:
            gr.Warning(str(e))
            return None, None, None

        interp_fn = (interpolate_chunks_mean_offset
                     if mode == 'Mean + offset' else interpolate_chunks)
        z_chunks = interp_fn(z1, z2, z3, t_val)
        y, spec = decode_to_audio(model, device, z_chunks)

        n_files = 3 if z3 is not None else 2
        tick_label = (f't={t_val/100:.2f}  (0=File1 · 1=File2)'
                      if n_files == 2 else
                      f't={t_val/100:.2f}  (0=File1 · 0.5=File2 · 1=File3)')

        fig_spec, ax = plt.subplots(figsize=(8, 2.5))
        ax.imshow(spec, aspect='auto', origin='lower', cmap='magma', vmin=-1, vmax=1)
        ax.set_title(f'Decoded mel spectrogram — {tick_label}', fontsize=9)
        ax.set_xlabel('Time frame')
        ax.set_ylabel('Mel bin')
        fig_spec.tight_layout()

        f_labels = ['File 1', 'File 2', 'File 3'][:n_files]
        fig_lat  = plot_latent_position(z1, z2, z3, z_chunks, t_val, f_labels)

        return (SR, y), fig_spec, fig_lat

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

        mode = gr.Radio(
            choices=['Per-chunk slerp', 'Mean + offset'],
            value='Per-chunk slerp',
            label='Interpolation mode',
            info='"Per-chunk slerp" morphs each chunk independently. '
                 '"Mean + offset" slerps the global character while blending '
                 'each file\'s temporal structure — smoother transitions.')

        btn = gr.Button('Generate', variant='primary')

        with gr.Row():
            out_audio = gr.Audio(label='Interpolated audio', type='numpy', autoplay=True)
            out_spec  = gr.Plot(label='Decoded mel spectrogram')

        out_latent = gr.Plot(label='Latent space')

        btn.click(fn=generate,
                  inputs=[f1, f2, f3, slider, mode],
                  outputs=[out_audio, out_spec, out_latent])

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
