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
from sklearn.decomposition import PCA

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
# latent visualisation helpers
# ---------------------------------------------------------------------------

def pool_latents(z_list):
    """Pool a list of (D, T_lat) tensors → single (D,) numpy vector (mean over T and chunks)."""
    return torch.stack([z.mean(dim=1) for z in z_list]).mean(dim=0).numpy()


def plot_latent_position(z1_list, z2_list, z3_list, z_cur_list, t_val, labels):
    """Two-panel figure:
      Left  — PCA scatter: endpoints + slerp path + current position.
      Right — Latent heatmap: interpolated latent (D × T_lat) for the middle chunk.
    """
    # Pool each file's chunks to a single (D,) vector
    p1  = pool_latents(z1_list)
    p2  = pool_latents(z2_list)
    p3  = pool_latents(z3_list) if z3_list is not None else None
    cur = pool_latents(z_cur_list)

    endpoints = [p1, p2] + ([p3] if p3 is not None else [])
    all_pts   = np.stack(endpoints + [cur])

    # Fit PCA on endpoints + current position
    n_components = min(2, all_pts.shape[0], all_pts.shape[1])
    pca    = PCA(n_components=n_components)
    proj   = pca.fit_transform(all_pts)        # (n_pts, 2)
    ep_proj  = proj[:-1]                        # endpoints
    cur_proj = proj[-1]                         # current

    # Compute full slerp path for the curve
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

    # --- Latent heatmap (middle chunk, all dims × time frames) ---
    mid   = len(z_cur_list) // 2
    heat  = z_cur_list[mid].numpy()            # (D, T_lat)
    vabs  = np.abs(heat).max() + 1e-8
    im    = ax_heat.imshow(heat, aspect='auto', origin='lower',
                           cmap='RdBu_r', vmin=-vabs, vmax=vabs)
    ax_heat.set_title('Interpolated latent (middle chunk)', fontsize=9)
    ax_heat.set_xlabel('Latent time frame', fontsize=8)
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
            _cache[key]        = path
            _cache[key + '_z'] = encode_file(model, device, path)
        return _cache[key + '_z']

    def generate(f1, f2, f3, t_val):
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

        z_chunks   = interpolate_chunks(z1, z2, z3, t_val)
        wave, spec = decode_to_audio(model, device, z_chunks)

        n_files    = 3 if z3 is not None else 2
        tick_label = (f't={t_val/100:.2f}  (0=File1 · 1=File2)'
                      if n_files == 2 else
                      f't={t_val/100:.2f}  (0=File1 · 0.5=File2 · 1=File3)')

        fig_spec, ax = plt.subplots(figsize=(8, 2.5))
        ax.imshow(spec, aspect='auto', origin='lower', cmap='magma')
        ax.set_title(f'Decoded STFT — {tick_label}', fontsize=9)
        ax.set_xlabel('Time frame')
        ax.set_ylabel('Freq bin')
        fig_spec.tight_layout()

        f_labels  = ['File 1', 'File 2', 'File 3'][:n_files]
        fig_lat   = plot_latent_position(z1, z2, z3, z_chunks, t_val, f_labels)

        return (SR, wave), fig_spec, fig_lat

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

        out_latent = gr.Plot(label='Latent space')

        btn.click(fn=generate,
                  inputs=[f1, f2, f3, slider],
                  outputs=[out_audio, out_spec, out_latent])

        demo.load(js="""
        () => {
            const observer = new MutationObserver(() => {
                document.querySelectorAll('audio').forEach(a => { a.loop = true; });
            });
            observer.observe(document.body, { childList: true, subtree: true });
        }
        """)

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
