"""
Evaluate a trained RawAudioVAE checkpoint.

Produces:
  - waveforms.png        : input vs reconstruction waveforms for N test chunks
  - spectrograms.png     : log-magnitude STFT of input vs reconstruction
  - latent_pca.png       : 2-D PCA of latent means (pooled over time), coloured by class
  - kl_per_dim.png       : mean KL per latent dimension
  - audio/               : .wav files for original and reconstructed chunks
  - interp_*_grid.png    : latent interpolation spectrogram grids (lerp vs slerp)
  - interp_audio/        : .wav files for lerp interpolation steps
  - pca_traversal_pc*.png: decoded spectrograms along top PCA directions

Usage:
    python evaluate_raw.py -r saved/models/RawAudioVAE/<timestamp>/model_best.pth [-n 6] [-o eval/raw_run]
"""

import argparse
import os
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import soundfile as sf
from sklearn.decomposition import PCA

import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser


# ---------------------------------------------------------------------------
# device / model loading
# ---------------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_model_and_data(config, resume, device):
    dl_args = dict(config['data_loader']['args'])
    dl_args.update({'validation_split': 0.0, 'shuffle': False, 'subset': None})
    data_loader = getattr(module_data, config['data_loader']['type'])(**dl_args)

    model = config.initialize('arch', module_arch)
    ckpt = torch.load(resume, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    return model, data_loader


# ---------------------------------------------------------------------------
# data collection
# ---------------------------------------------------------------------------

def collect_latents_and_recons(model, data_loader, device, max_per_class=64, seed=0):
    """Run the test set through the model. Returns per-chunk arrays.

    Stratified sampling: collects up to max_per_class chunks per class,
    guaranteeing all classes are represented regardless of directory order.
    Deterministic via fixed seed.

    mu / logvar are pooled over the temporal axis (mean over T_lat) to give
    a single D-dim vector per chunk, suitable for PCA and KL-per-dim plots.
    """
    # Collect everything first, then stratified-subsample
    mu_list, logvar_list = [], []
    inp_list, rec_list   = [], []
    labels_out, fnames_out, cidx_out = [], [], []

    dataset = data_loader.dataset

    with torch.no_grad():
        for data_idx, label, data in data_loader:
            x = data.float().to(device)
            y_hat, mu, logvar, _ = model(x)

            mu_list.append(mu.mean(dim=2).cpu())
            logvar_list.append(logvar.mean(dim=2).cpu())
            inp_list.append(x.squeeze(1).cpu())
            rec_list.append(y_hat.squeeze(1).cpu())

            for b, (idx, lbl) in enumerate(zip(data_idx.tolist(), list(label))):
                _, fpath, chunk_start, *_ = dataset.items[idx]
                fnames_out.append(os.path.basename(fpath))
                cidx_out.append(chunk_start // dataset.chunk_size)
                labels_out.append(lbl)

    mu_all      = torch.cat(mu_list)
    logvar_all  = torch.cat(logvar_list)
    inp_all     = torch.cat(inp_list)
    rec_all     = torch.cat(rec_list)

    # Stratified subsample
    rng = random.Random(seed)
    by_class = {}
    for i, lbl in enumerate(labels_out):
        by_class.setdefault(lbl, []).append(i)
    keep = []
    for lbl, idxs in sorted(by_class.items()):
        keep += rng.sample(idxs, min(max_per_class, len(idxs)))
    keep.sort()

    print(f"  {len(inp_all)} chunks from {len(set(fnames_out))} files collected")
    print(f"  Stratified: {len(keep)} chunks kept ({len(by_class)} classes, "
          f"up to {max_per_class} each)")

    return (mu_all[keep], logvar_all[keep],
            inp_all[keep], rec_all[keep],
            [labels_out[i] for i in keep],
            [fnames_out[i] for i in keep],
            [cidx_out[i]   for i in keep])


def select_sample_indices(filenames, chunk_indices, n_samples, labels=None):
    """Pick one chunk per unique label/file for diverse visualisation."""
    key_to_file_to_indices = {}
    for i, (fname, cidx) in enumerate(zip(filenames, chunk_indices)):
        lbl = labels[i] if labels is not None else fname
        key_to_file_to_indices.setdefault(lbl, {}).setdefault(fname, []).append(i)

    all_file_idxs = [
        idxs[len(idxs) // 2]
        for fm in key_to_file_to_indices.values()
        for idxs in fm.values()
    ]
    label_first = [
        next(iter(fm.values()))[len(next(iter(fm.values()))) // 2]
        for fm in key_to_file_to_indices.values()
    ]
    seen, selected = set(), []
    for idx in label_first + all_file_idxs:
        if idx not in seen:
            seen.add(idx)
            selected.append(idx)
        if len(selected) >= n_samples:
            break
    return selected


def log_magnitude_stft(wave_np, n_fft=1024, hop=256):
    """Log-magnitude spectrogram of a 1-D float32 numpy waveform."""
    t   = torch.tensor(wave_np, dtype=torch.float32)
    win = torch.hann_window(n_fft)
    S   = torch.stft(t, n_fft, hop, n_fft, win, return_complex=True)
    return torch.log(S.abs() + 1e-5).numpy()   # (F, T_frames)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_waveforms(inputs, recons, labels, filenames, chunk_indices,
                   sample_indices, out_path, sr=22050):
    """Overlay original and reconstructed waveforms for selected chunks."""
    n      = len(sample_indices)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), squeeze=False)
    t_axis = np.arange(inputs.size(1)) / sr

    for row, i in enumerate(sample_indices):
        inp = inputs[i].numpy()
        rec = recons[i].numpy()
        ax  = axes[row, 0]
        ax.plot(t_axis, inp, alpha=0.7, linewidth=0.4, label='original', color='steelblue')
        ax.plot(t_axis, rec, alpha=0.7, linewidth=0.4, label='recon',    color='tomato')
        ax.set_xlim(0, t_axis[-1])
        fname = filenames[i] if i < len(filenames) else ''
        lbl   = labels[i]   if i < len(labels)    else ''
        cidx  = chunk_indices[i] if i < len(chunk_indices) else 0
        ax.set_title(f'[{lbl}]  {fname}  chunk {cidx}', fontsize=8)
        ax.set_xlabel('Time (s)', fontsize=7)
        ax.set_ylabel('Amplitude', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=7, loc='upper right')

    plt.suptitle('RawAudioVAE: input vs reconstruction waveforms', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out_path}')


def plot_spectrograms(inputs, recons, labels, filenames, chunk_indices,
                      sample_indices, out_path, n_fft=1024, hop=256):
    """Log-magnitude STFT of input vs reconstruction (and difference)."""
    n   = len(sample_indices)
    fig = plt.figure(figsize=(14, 3 * n))
    gs  = gridspec.GridSpec(n, 3, figure=fig, width_ratios=[1, 1, 1],
                            hspace=0.5, wspace=0.35)

    for row, i in enumerate(sample_indices):
        S_in  = log_magnitude_stft(inputs[i].numpy(), n_fft, hop)
        S_rec = log_magnitude_stft(recons[i].numpy(), n_fft, hop)
        diff  = S_in - S_rec

        vmin = min(S_in.min(), S_rec.min())
        vmax = max(S_in.max(), S_rec.max())

        ax_in  = fig.add_subplot(gs[row, 0])
        ax_rec = fig.add_subplot(gs[row, 1])
        ax_dif = fig.add_subplot(gs[row, 2])

        im0 = ax_in.imshow(S_in,  aspect='auto', origin='lower',
                           cmap='magma', vmin=vmin, vmax=vmax)
        im1 = ax_rec.imshow(S_rec, aspect='auto', origin='lower',
                            cmap='magma', vmin=vmin, vmax=vmax)
        im2 = ax_dif.imshow(diff,  aspect='auto', origin='lower',
                            cmap='RdBu_r',
                            vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())

        plt.colorbar(im0, ax=ax_in,  fraction=0.046, pad=0.04)
        plt.colorbar(im1, ax=ax_rec, fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=ax_dif, fraction=0.046, pad=0.04)

        fname = filenames[i] if i < len(filenames) else ''
        lbl   = labels[i]   if i < len(labels)    else ''
        cidx  = chunk_indices[i] if i < len(chunk_indices) else 0
        ax_in.set_title(f'Input [{lbl}]  {fname}  chunk {cidx}', fontsize=8)
        ax_rec.set_title('Reconstruction', fontsize=9)
        ax_dif.set_title('Difference (input − recon)', fontsize=9)
        for ax in (ax_in, ax_rec, ax_dif):
            ax.set_xlabel('Frame',    fontsize=7)
            ax.set_ylabel('Freq bin', fontsize=7)
            ax.tick_params(labelsize=6)

    plt.suptitle('RawAudioVAE: log-magnitude STFT — input vs reconstruction', fontsize=10)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out_path}')


def plot_latent_pca(mu, labels, out_path):
    """2-D PCA scatter of pooled latent means, coloured by class."""
    unique_labels = sorted(set(labels))
    colour_map    = plt.cm.get_cmap('tab10', len(unique_labels))
    label_to_idx  = {l: i for i, l in enumerate(unique_labels)}

    z2 = PCA(n_components=2).fit_transform(mu.numpy())

    fig, ax = plt.subplots(figsize=(6, 5))
    for lbl in unique_labels:
        mask = np.array([l == lbl for l in labels])
        ax.scatter(z2[mask, 0], z2[mask, 1],
                   c=[colour_map(label_to_idx[lbl])], label=lbl, alpha=0.6, s=15)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title('Latent space — 2-D PCA of q(z|x) means (pooled over time)')
    ax.legend(title='class', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'Saved  {out_path}')


def plot_kl_per_dim(mu, logvar, out_path, free_bits=None):
    """KL divergence per latent dimension (averaged over the test set).

    Bars are coloured by activity: green = active (KL > free_bits threshold),
    red = collapsed (KL <= free_bits threshold).  A threshold line is drawn
    if free_bits is provided.  Summary stats are printed to stdout.
    """
    kl      = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (N, D)
    kl_mean = kl.mean(0).numpy()                                # (D,)

    threshold = free_bits if free_bits is not None else 0.0
    colors    = ['#2ecc71' if v > threshold else '#e74c3c' for v in kl_mean]

    n_active   = int((kl_mean > threshold).sum())
    n_dims     = len(kl_mean)
    total_kl   = kl_mean.sum()
    active_kl  = kl_mean[kl_mean > threshold]
    mean_active = active_kl.mean() if len(active_kl) > 0 else 0.0

    # Sort indices by KL descending for the second panel
    order = np.argsort(kl_mean)[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3.5))

    # Panel 1: dims in original order
    ax1.bar(np.arange(n_dims), kl_mean, color=colors)
    if free_bits is not None:
        ax1.axhline(free_bits, color='k', linewidth=1.0, linestyle='--',
                    label=f'free_bits = {free_bits}')
        ax1.legend(fontsize=8)
    ax1.axhline(0, color='k', linewidth=0.5)
    ax1.set_xlabel('Latent dimension (original order)')
    ax1.set_ylabel('Mean KL')
    ax1.set_title(f'KL per dim  —  {n_active}/{n_dims} active  '
                  f'(total KL={total_kl:.2f}, mean active={mean_active:.2f})')

    # Panel 2: dims sorted by KL descending
    ax2.bar(np.arange(n_dims), kl_mean[order],
            color=[colors[i] for i in order])
    if free_bits is not None:
        ax2.axhline(free_bits, color='k', linewidth=1.0, linestyle='--')
    ax2.axhline(0, color='k', linewidth=0.5)
    ax2.set_xlabel('Rank (sorted by KL, descending)')
    ax2.set_ylabel('Mean KL')
    ax2.set_title('Sorted by activity')

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    print(f'Saved  {out_path}')
    print(f'  KL per dim: {n_active}/{n_dims} active dims  |  '
          f'total KL={total_kl:.3f}  |  mean active={mean_active:.3f}')
    top5 = order[:5]
    print(f'  Top-5 dims by KL: ' +
          ', '.join(f'dim{i}={kl_mean[i]:.3f}' for i in top5))


def reconstruct_full_file_raw(model, wav_path, device, sr=22050,
                              chunk_size=16384, crossfade=256):
    """Load a full wav, split into chunks, VAE encode/decode, concatenate.

    A short linear crossfade is applied at chunk boundaries to reduce clicks.
    Returns (orig_wave, recon_wave) as float32 numpy arrays, or (None, None)
    if the file is shorter than one chunk.
    """
    import torchaudio
    y, file_sr = torchaudio.load(wav_path)
    if file_sr != sr:
        y = torchaudio.functional.resample(y, file_sr, sr)
    if y.shape[0] > 1:
        y = y.mean(0, keepdim=True)
    y = y.squeeze(0).numpy()   # (T,)

    n_chunks = len(y) // chunk_size
    if n_chunks == 0:
        return None, None

    recon_chunks = []
    model.eval()
    with torch.no_grad():
        for i in range(n_chunks):
            chunk = torch.tensor(y[i * chunk_size:(i + 1) * chunk_size],
                                 dtype=torch.float32)
            x     = chunk.unsqueeze(0).unsqueeze(0).to(device)   # (1, 1, T)
            y_hat, _, _, _ = model(x)
            recon_chunks.append(y_hat.squeeze(0).squeeze(0).cpu().numpy())

    orig_full = y[:n_chunks * chunk_size]

    # Crossfade at chunk boundaries to suppress clicks
    if crossfade > 0 and len(recon_chunks) > 1:
        fade_out = np.linspace(1, 0, crossfade)
        fade_in  = np.linspace(0, 1, crossfade)
        result   = recon_chunks[0].copy()
        for chunk in recon_chunks[1:]:
            blend  = result[-crossfade:] * fade_out + chunk[:crossfade] * fade_in
            result = np.concatenate([result[:-crossfade], blend, chunk[crossfade:]])
        recon_full = result
    else:
        recon_full = np.concatenate(recon_chunks)

    return orig_full, recon_full


def save_audio_samples(model, dataset, labels, filenames, chunk_indices,
                       sample_indices, out_dir, device, sr=22050,
                       chunk_size=16384):
    """Reconstruct full source files through the VAE and save as .wav pairs."""
    os.makedirs(out_dir, exist_ok=True)

    # Map basename → full wav path (one entry per unique file)
    path_map = {}
    for _, fpath, *_ in dataset.items:
        path_map.setdefault(os.path.basename(fpath), fpath)

    seen, row = set(), 0
    for i in sample_indices:
        fname = filenames[i] if i < len(filenames) else None
        lbl   = labels[i]   if i < len(labels)    else 'unk'
        if fname is None or fname in seen:
            continue
        seen.add(fname)

        wav_path = path_map.get(fname)
        if wav_path is None:
            print(f'  Warning: could not find wav path for {fname}')
            continue

        orig, recon = reconstruct_full_file_raw(
            model, wav_path, device, sr=sr, chunk_size=chunk_size)
        if orig is None:
            print(f'  Warning: {fname} too short, skipping')
            continue

        stem = os.path.splitext(fname)[0]
        tag  = f'{row:03d}_{lbl}_{stem}'
        sf.write(os.path.join(out_dir, f'{tag}_orig.wav'),
                 orig  / (np.abs(orig).max()  + 1e-8), sr)
        sf.write(os.path.join(out_dir, f'{tag}_recon.wav'),
                 recon / (np.abs(recon).max() + 1e-8), sr)
        row += 1

    print(f'Saved  {row} full-file audio pairs → {out_dir}/')


def plot_full_file_spectrograms(model, dataset, labels, filenames, chunk_indices,
                                sample_indices, out_dir, device, sr=22050,
                                chunk_size=16384, n_fft=1024, hop=256):
    """For each selected file, plot the full-file log-STFT input vs reconstruction."""
    path_map = {}
    for _, fpath, *_ in dataset.items:
        path_map.setdefault(os.path.basename(fpath), fpath)

    seen, row = set(), 0
    for i in sample_indices:
        fname = filenames[i] if i < len(filenames) else None
        lbl   = labels[i]   if i < len(labels)    else 'unk'
        if fname is None or fname in seen:
            continue
        seen.add(fname)

        wav_path = path_map.get(fname)
        if wav_path is None:
            continue

        orig, recon = reconstruct_full_file_raw(
            model, wav_path, device, sr=sr, chunk_size=chunk_size)
        if orig is None:
            continue

        min_len = min(len(orig), len(recon))
        S_in  = log_magnitude_stft(orig[:min_len],  n_fft, hop)
        S_rec = log_magnitude_stft(recon[:min_len], n_fft, hop)
        diff  = S_in - S_rec
        vmin  = min(S_in.min(), S_rec.min())
        vmax  = max(S_in.max(), S_rec.max())

        fig, axes = plt.subplots(1, 3, figsize=(14, 3))
        for ax, data, title, cmap, v0, v1 in [
            (axes[0], S_in,  f'Input [{lbl}] {fname}', 'magma', vmin, vmax),
            (axes[1], S_rec, 'Reconstruction',          'magma', vmin, vmax),
            (axes[2], diff,  'Difference',     'RdBu_r',
             -np.abs(diff).max(), np.abs(diff).max()),
        ]:
            im = ax.imshow(data, aspect='auto', origin='lower',
                           cmap=cmap, vmin=v0, vmax=v1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=8)
            ax.set_xlabel('Frame'); ax.set_ylabel('Freq bin')
        fig.tight_layout()
        out_path = os.path.join(out_dir,
                                f'{row:03d}_{lbl}_{os.path.splitext(fname)[0]}_fullfile.png')
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved  {out_path}')
        row += 1


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


def get_chunk_latent(model, dataset, item_idx, device):
    """Encode one dataset item and return mu (D, T_lat) on CPU."""
    _, lbl, y = dataset[item_idx]
    x = y.float().unsqueeze(0).to(device)   # (1, 1, T)
    with torch.no_grad():
        mu, _, _ = model.encode(x)
    return mu.squeeze(0).cpu(), lbl          # (D, T_lat)


def decode_latent_seq(model, z_seq, device):
    """Decode a latent sequence (D, T_lat) → waveform (T,) numpy on CPU."""
    with torch.no_grad():
        y_hat = model.decode(z_seq.unsqueeze(0).to(device))
    return y_hat.squeeze(0).squeeze(0).cpu().numpy()   # (T,)


def plot_interpolations(model, dataset, filenames, chunk_indices, labels,
                        sample_indices, device, n_steps=8,
                        out_dir='.', sr=22050, n_fft=1024, hop=256):
    """Interpolate latent sequences between pairs of test files (lerp + slerp).

    Saves a spectrogram grid PNG and lerp .wav files for each pair.
    """
    # Build reverse map: (basename, chunk_idx) → dataset item index
    fname_cidx_to_item = {}
    for item_idx, (lbl, fpath, start, *_) in enumerate(dataset.items):
        key = (os.path.basename(fpath), start // dataset.chunk_size)
        fname_cidx_to_item.setdefault(key, item_idx)

    seen, file_latents, file_meta = set(), [], []
    for i in sample_indices:
        fname = filenames[i]
        if fname in seen:
            continue
        seen.add(fname)
        item_idx = fname_cidx_to_item.get((fname, chunk_indices[i]))
        if item_idx is None:
            print(f'  Warning: could not find dataset item for {fname}')
            continue
        z, _ = get_chunk_latent(model, dataset, item_idx, device)
        file_latents.append(z)
        file_meta.append((fname, labels[i]))

    if len(file_latents) < 2:
        print('  Not enough files for interpolation.')
        return

    ts        = np.linspace(0, 1, n_steps)
    audio_dir = os.path.join(out_dir, 'interp_audio')
    os.makedirs(audio_dir, exist_ok=True)

    for pair_idx in range(len(file_latents) - 1):
        z1, z2       = file_latents[pair_idx], file_latents[pair_idx + 1]
        fname1, lbl1 = file_meta[pair_idx]
        fname2, lbl2 = file_meta[pair_idx + 1]
        stem1 = os.path.splitext(fname1)[0]
        stem2 = os.path.splitext(fname2)[0]

        lerp_specs, slerp_specs, lerp_waves = [], [], []
        for t in ts:
            t_t     = torch.tensor(t, dtype=torch.float32)
            z_lerp  = (1 - t_t) * z1 + t_t * z2
            z_slerp = slerp(z1, z2, t_t)
            wave    = decode_latent_seq(model, z_lerp, device)
            lerp_waves.append(wave)
            lerp_specs.append(log_magnitude_stft(wave, n_fft, hop))
            slerp_specs.append(log_magnitude_stft(
                decode_latent_seq(model, z_slerp, device), n_fft, hop))

        vmin = min(s.min() for s in lerp_specs + slerp_specs)
        vmax = max(s.max() for s in lerp_specs + slerp_specs)

        fig, axes = plt.subplots(2, n_steps, figsize=(2.2 * n_steps, 5),
                                 sharex=True, sharey=True)
        for col, t in enumerate(ts):
            for row, (specs, method) in enumerate([(lerp_specs,  'lerp'),
                                                   (slerp_specs, 'slerp')]):
                ax = axes[row, col]
                ax.imshow(specs[col], aspect='auto', origin='lower',
                          cmap='magma', vmin=vmin, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(method, fontsize=9)
                if row == 0:
                    ax.set_title(f't={t:.2f}', fontsize=7)
        fig.suptitle(f'Interpolation: [{lbl1}] {stem1}  →  [{lbl2}] {stem2}',
                     fontsize=9)
        fig.tight_layout()
        grid_path = os.path.join(out_dir,
                                 f'interp_{pair_idx:02d}_{stem1}_to_{stem2}_grid.png')
        fig.savefig(grid_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved  {grid_path}')

        for t, wave in zip(ts, lerp_waves):
            wav_path = os.path.join(audio_dir,
                                    f'interp_{pair_idx:02d}_{stem1}_to_{stem2}_t{t:.2f}.wav')
            sf.write(wav_path, wave / (np.abs(wave).max() + 1e-8), sr)

    print(f'Saved  interpolation audio → {audio_dir}/')


# ---------------------------------------------------------------------------
# PCA traversal
# ---------------------------------------------------------------------------

def plot_pca_traversal(model, mu_all, T_lat, device, n_components=3, n_steps=7,
                       out_dir='.', n_fft=1024, hop=256):
    """Traverse top PCA directions of the pooled latent space.

    Starting from the dataset mean (replicated over T_lat), move ±3σ along
    each principal component and decode to log-magnitude spectrograms.
    """
    pca    = PCA(n_components=n_components)
    pca.fit(mu_all.numpy())
    mu_mean = torch.tensor(pca.mean_, dtype=torch.float32)   # (D,)

    for pc_idx in range(n_components):
        direction = torch.tensor(pca.components_[pc_idx], dtype=torch.float32)  # (D,)
        sigma     = float(np.sqrt(pca.explained_variance_[pc_idx]))
        alphas    = np.linspace(-3 * sigma, 3 * sigma, n_steps)

        specs = []
        for alpha in alphas:
            z_pool = mu_mean + alpha * direction           # (D,)
            z_seq  = z_pool.unsqueeze(-1).expand(-1, T_lat)  # (D, T_lat)
            wave   = decode_latent_seq(model, z_seq, device)
            specs.append(log_magnitude_stft(wave, n_fft, hop))

        var_ratio = pca.explained_variance_ratio_[pc_idx] * 100
        vmin = min(s.min() for s in specs)
        vmax = max(s.max() for s in specs)

        fig, axes = plt.subplots(1, n_steps, figsize=(2.2 * n_steps, 3),
                                 sharex=True, sharey=True)
        for col, (alpha, spec) in enumerate(zip(alphas, specs)):
            ax = axes[col]
            ax.imshow(spec, aspect='auto', origin='lower',
                      cmap='magma', vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{alpha:+.1f}', fontsize=7)
        axes[0].set_ylabel('Freq bin', fontsize=8)
        fig.suptitle(f'PC {pc_idx + 1}  ({var_ratio:.1f}% variance)'
                     f' — traversal from −3σ to +3σ', fontsize=9)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f'pca_traversal_pc{pc_idx + 1}.png')
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved  {out_path}')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(config, resume, n_samples, out_dir):
    device = get_device()
    print(f'Using device: {device}')
    os.makedirs(out_dir, exist_ok=True)

    model, data_loader = load_model_and_data(config, resume, device)
    sr         = config['data_loader']['args'].get('sr', 22050)
    chunk_size = config['data_loader']['args'].get('chunk_size', 16384)

    # Infer temporal latent length T_lat from a dummy forward pass
    with torch.no_grad():
        dummy      = torch.zeros(1, 1, chunk_size).to(device)
        mu_dummy, _, _ = model.encode(dummy)
        T_lat      = mu_dummy.shape[2]
    print(f'Chunk size: {chunk_size}  |  Latent frames T_lat: {T_lat}  |  sr: {sr}')

    print('Collecting latents and reconstructions…')
    mu, logvar, inputs, recons, labels, filenames, chunk_indices = \
        collect_latents_and_recons(model, data_loader, device)

    print(f'  {inputs.size(0)} chunks from {len(set(filenames))} files collected')
    print(f'  Input  range: [{inputs.min():.4f}, {inputs.max():.4f}]')
    print(f'  Recon  range: [{recons.min():.4f}, {recons.max():.4f}]')

    sample_indices = select_sample_indices(filenames, chunk_indices, n_samples, labels=labels)
    print('  Visualising: ' +
          ', '.join(f'{filenames[i]} chunk {chunk_indices[i]}' for i in sample_indices))

    plot_waveforms(inputs, recons, labels, filenames, chunk_indices,
                   sample_indices, os.path.join(out_dir, 'waveforms.png'), sr=sr)

    plot_spectrograms(inputs, recons, labels, filenames, chunk_indices,
                      sample_indices, os.path.join(out_dir, 'spectrograms.png'))

    plot_latent_pca(mu, labels, os.path.join(out_dir, 'latent_pca.png'))

    free_bits = config['trainer'].get('free_bits', None)
    plot_kl_per_dim(mu, logvar, os.path.join(out_dir, 'kl_per_dim.png'),
                    free_bits=free_bits)

    save_audio_samples(model, data_loader.dataset, labels, filenames, chunk_indices,
                       sample_indices, os.path.join(out_dir, 'audio'), device,
                       sr=sr, chunk_size=chunk_size)

    print('\nFull-file spectrogram plots…')
    plot_full_file_spectrograms(model, data_loader.dataset, labels, filenames,
                                chunk_indices, sample_indices, out_dir, device,
                                sr=sr, chunk_size=chunk_size)

    print('\nLatent space analysis…')
    plot_interpolations(model, data_loader.dataset, filenames, chunk_indices, labels,
                        sample_indices, device, n_steps=8, out_dir=out_dir, sr=sr)

    plot_pca_traversal(model, mu, T_lat, device, n_components=3, out_dir=out_dir)

    print('\nDone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RawAudioVAE evaluation')
    parser.add_argument('-r', '--resume',    required=True,
                        help='path to checkpoint (.pth)')
    parser.add_argument('-c', '--config',    default=None,
                        help='config file (inferred from checkpoint dir if omitted)')
    parser.add_argument('-d', '--device',    default=None)
    parser.add_argument('-n', '--n_samples', type=int, default=6,
                        help='number of chunks to visualise / save as audio')
    parser.add_argument('-o', '--out_dir',   default='eval/raw_run',
                        help='output directory (default: eval/raw_run/)')

    args   = parser.parse_args()
    config = ConfigParser(parser)
    main(config, args.resume, args.n_samples, args.out_dir)
