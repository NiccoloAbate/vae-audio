"""
Evaluate a trained SpecVAE checkpoint.

Produces:
  - spectrograms.png   : input vs reconstruction side-by-side for N test samples
  - latent_pca.png     : 2-D PCA of the latent space, coloured by class label
  - kl_per_dim.png     : KL contribution per latent dimension (detects posterior collapse)
  - audio/             : .wav files for original and reconstructed chunks

Usage:
    python evaluate.py -r saved/models/SpecVAE/<timestamp>/model_best.pth [-n 6] [-o eval/<run_name>]
"""

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import soundfile as sf
from sklearn.decomposition import PCA

import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_model_and_data(config, resume, device):
    dl_args = dict(config['data_loader']['args'])
    dl_args.update({'validation_split': 0.0, 'shuffle': False, 'subset': 'test'})
    data_loader = getattr(module_data, config['data_loader']['type'])(**dl_args)

    model = config.initialize('arch', module_arch)
    ckpt = torch.load(resume, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    return model, data_loader


def collect_latents_and_recons(model, data_loader, device, max_samples=500):
    """Run the full test set through the encoder/decoder, collect results.

    Returns per-chunk tensors plus per-chunk metadata (label, filename, chunk_index).
    Each original file contributes n_chunks consecutive entries.
    """
    all_mu, all_logvar = [], []
    all_labels, all_filenames, all_chunk_idx = [], [], []
    inputs, recons = [], []

    with torch.no_grad():
        for data_idx, label, data in data_loader:
            x = data.type('torch.FloatTensor').to(device)
            nF, nT = x.size(2), x.size(3)
            n_chunks = x.size(1)
            x_flat = x.view(-1, 1, nF, nT)          # (batch*n_chunks, 1, F, T)

            x_recon, mu, logvar, _ = model(x_flat)

            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())
            inputs.append(x_flat.squeeze(1).cpu())
            recons.append(x_recon.cpu())

            for b, (idx, lbl) in enumerate(zip(data_idx.tolist(), label)):
                fname = os.path.basename(
                    str(data_loader.dataset.path_to_data[idx]))
                for c in range(n_chunks):
                    all_labels.append(lbl)
                    all_filenames.append(fname)
                    all_chunk_idx.append(c)

            if sum(t.size(0) for t in inputs) >= max_samples:
                break

    return (torch.cat(all_mu), torch.cat(all_logvar),
            torch.cat(inputs), torch.cat(recons),
            all_labels, all_filenames, all_chunk_idx)


def select_sample_indices(filenames, chunk_indices, n_samples):
    """Pick one chunk per unique file (the middle chunk = more likely active).
    Returns a list of integer indices into the chunk arrays."""
    file_to_indices = {}
    for i, (fname, cidx) in enumerate(zip(filenames, chunk_indices)):
        file_to_indices.setdefault(fname, []).append(i)

    selected = []
    for fname, idxs in file_to_indices.items():
        selected.append(idxs[len(idxs) // 2])   # middle chunk of this file
        if len(selected) >= n_samples:
            break
    return selected


def mel_to_audio(spec_db, sr=22050, n_fft=2048, hop_length=735):
    """Convert a (64, T) dB mel spectrogram to a waveform via Griffin-Lim."""
    S_power = librosa.db_to_power(spec_db.astype(np.float32))
    return librosa.feature.inverse.mel_to_audio(
        S_power, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=64)


def reconstruct_full_file(model, npy_path, device,
                          chunk_size=15, sr=22050, n_fft=2048, hop_length=735):
    """
    Reconstruct a full audio file by:
      1. Loading the pre-computed mel spectrogram (.npy, shape (64, N_frames))
      2. Normalising to (-1, 1)
      3. Splitting into non-overlapping chunks of chunk_size frames
      4. Passing each chunk through the VAE
      5. Concatenating reconstructed chunks along the time axis
      6. Inverting both original and reconstruction to audio via Griffin-Lim

    Returns (y_orig, y_recon, spec_norm, recon_full) — all numpy arrays.
    """
    spec = np.load(npy_path)                       # (64, N_frames), dB
    spec_norm = spec / 40.0 + 1.0                  # normalise to (-1, 1)

    # Slice into non-overlapping chunks (same logic as SpecChunking)
    n_frames = spec_norm.shape[1]
    n_chunks = n_frames // chunk_size
    chunks = [spec_norm[:, i*chunk_size:(i+1)*chunk_size]
              for i in range(n_chunks)]             # list of (64, chunk_size)

    # Run each chunk through the VAE
    recon_chunks = []
    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            x = torch.tensor(chunk, dtype=torch.float32)
            x = x.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 64, chunk_size)
            x_recon, _, _, _ = model(x)
            recon_chunks.append(x_recon.squeeze(0).cpu().numpy())  # (64, chunk_size)

    # Concatenate along time axis → (64, n_chunks * chunk_size)
    recon_full = np.concatenate(recon_chunks, axis=1)
    orig_full  = spec_norm[:, :n_chunks * chunk_size]

    # Invert NormaliseSpecDb → dB, then Griffin-Lim
    y_orig  = mel_to_audio((orig_full  - 1.0) * 40.0, sr=sr, n_fft=n_fft, hop_length=hop_length)
    y_recon = mel_to_audio((recon_full - 1.0) * 40.0, sr=sr, n_fft=n_fft, hop_length=hop_length)

    return y_orig, y_recon, orig_full, recon_full


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_spectrograms(inputs, recons, labels, filenames, chunk_indices,
                      sample_indices, out_path):
    """Plot input vs reconstruction for selected chunks (one per file)."""
    n = len(sample_indices)
    fig = plt.figure(figsize=(14, 3 * n))
    gs = gridspec.GridSpec(n, 3, figure=fig,
                           width_ratios=[1, 1, 1], hspace=0.5, wspace=0.35)

    for row, i in enumerate(sample_indices):
        inp  = inputs[i].numpy()
        rec  = recons[i].numpy()
        diff = inp - rec

        ax_in  = fig.add_subplot(gs[row, 0])
        ax_rec = fig.add_subplot(gs[row, 1])
        ax_dif = fig.add_subplot(gs[row, 2])

        im0 = ax_in.imshow(inp,  aspect='auto', origin='lower', cmap='magma')
        im1 = ax_rec.imshow(rec, aspect='auto', origin='lower', cmap='magma')
        im2 = ax_dif.imshow(diff, aspect='auto', origin='lower',
                            cmap='RdBu_r',
                            vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())

        plt.colorbar(im0, ax=ax_in,  fraction=0.046, pad=0.04)
        plt.colorbar(im1, ax=ax_rec, fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=ax_dif, fraction=0.046, pad=0.04)

        fname = filenames[i] if i < len(filenames) else ''
        lbl   = labels[i]   if i < len(labels)    else ''
        cidx  = chunk_indices[i] if i < len(chunk_indices) else 0
        ax_in.set_title(f'Input  [{lbl}]  {fname}  chunk {cidx}', fontsize=8)
        ax_rec.set_title('Reconstruction', fontsize=9)
        ax_dif.set_title('Difference (input − recon)', fontsize=9)
        for ax in (ax_in, ax_rec, ax_dif):
            ax.set_xlabel('Time frame', fontsize=7)
            ax.set_ylabel('Mel bin', fontsize=7)
            ax.tick_params(labelsize=6)

    plt.suptitle('SpecVAE: input vs reconstruction  (both normalised to (−1, 1))',
                 fontsize=10)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out_path}')


def plot_latent_pca(mu, labels, out_path):
    """2-D PCA scatter of latent means, coloured by class."""
    unique_labels = sorted(set(labels))
    colour_map = plt.cm.get_cmap('tab10', len(unique_labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    z2 = PCA(n_components=2).fit_transform(mu.numpy())

    fig, ax = plt.subplots(figsize=(6, 5))
    for lbl in unique_labels:
        mask = np.array([l == lbl for l in labels])
        ax.scatter(z2[mask, 0], z2[mask, 1],
                   c=[colour_map(label_to_idx[lbl])],
                   label=lbl, alpha=0.6, s=15)
    ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')
    ax.set_title('Latent space — 2-D PCA of q(z|x) means')
    ax.legend(title='class', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'Saved  {out_path}')


def plot_kl_per_dim(mu, logvar, out_path):
    """KL divergence per latent dimension (averaged over the test set)."""
    # KL_i = -0.5 * (1 + logvar_i - mu_i^2 - exp(logvar_i))
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (N, D)
    kl_mean = kl.mean(0).numpy()                           # (D,)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(np.arange(len(kl_mean)), kl_mean, color='steelblue')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Latent dimension')
    ax.set_ylabel('Mean KL divergence')
    ax.set_title('KL per latent dimension  '
                 '(dims near 0 = "collapsed" / unused)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'Saved  {out_path}')


def save_audio_samples(model, data_loader, labels, filenames, chunk_indices,
                       sample_indices, out_dir, device,
                       sr=22050, n_fft=2048, hop_length=735):
    """
    For each selected sample, reconstruct the full source file (not just one chunk)
    by concatenating VAE reconstructions of all non-overlapping chunks.
    Saves original and reconstructed .wav files.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Map filename → full path via the dataset
    path_map = {os.path.basename(str(p)): str(p)
                for p in data_loader.dataset.path_to_data}

    seen = set()
    row = 0
    for i in sample_indices:
        fname = filenames[i] if i < len(filenames) else None
        lbl   = labels[i]   if i < len(labels)    else 'unk'
        if fname is None or fname in seen:
            continue
        seen.add(fname)

        npy_path = path_map.get(fname)
        if npy_path is None:
            print(f'  Warning: could not find path for {fname}')
            continue

        y_orig, y_recon, _, _ = reconstruct_full_file(
            model, npy_path, device, sr=sr, n_fft=n_fft, hop_length=hop_length)

        stem = os.path.splitext(fname)[0]
        tag  = f'{row:03d}_{lbl}_{stem}'
        sf.write(os.path.join(out_dir, f'{tag}_original.wav'),
                 y_orig  / (np.abs(y_orig).max()  + 1e-8), sr)
        sf.write(os.path.join(out_dir, f'{tag}_recon.wav'),
                 y_recon / (np.abs(y_recon).max() + 1e-8), sr)
        row += 1

    print(f'Saved  {row} full-file audio pairs → {out_dir}/')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(config, resume, n_samples, out_dir):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    os.makedirs(out_dir, exist_ok=True)

    model, data_loader = load_model_and_data(config, resume, device)
    print('Collecting latents and reconstructions…')
    mu, logvar, inputs, recons, labels, filenames, chunk_indices = \
        collect_latents_and_recons(model, data_loader, device)

    print(f'  {inputs.size(0)} chunks from '
          f'{len(set(filenames))} files collected')
    print(f'  Input   range: [{inputs.min():.2f}, {inputs.max():.2f}]')
    print(f'  Recon   range: [{recons.min():.2f}, {recons.max():.2f}]')

    sample_indices = select_sample_indices(filenames, chunk_indices, n_samples)
    print(f'  Visualising chunks from: '
          + ', '.join(filenames[i] for i in sample_indices))

    plot_spectrograms(inputs, recons, labels, filenames, chunk_indices,
                      sample_indices, os.path.join(out_dir, 'spectrograms.png'))

    plot_latent_pca(mu, labels,
                    os.path.join(out_dir, 'latent_pca.png'))

    plot_kl_per_dim(mu, logvar,
                    os.path.join(out_dir, 'kl_per_dim.png'))

    save_audio_samples(model, data_loader, labels, filenames, chunk_indices,
                       sample_indices, os.path.join(out_dir, 'audio'), device)

    # Full-file spectrogram plots
    path_map = {os.path.basename(str(p)): str(p)
                for p in data_loader.dataset.path_to_data}
    seen = set()
    for row, i in enumerate(sample_indices):
        fname = filenames[i]
        lbl   = labels[i]
        if fname in seen:
            continue
        seen.add(fname)
        npy_path = path_map.get(fname)
        if npy_path is None:
            continue
        _, _, orig_full, recon_full = reconstruct_full_file(
            model, npy_path, device)
        diff = orig_full - recon_full
        fig, axes = plt.subplots(1, 3, figsize=(14, 3))
        for ax, data, title, cmap, vmin, vmax in [
            (axes[0], orig_full,  f'Input [{lbl}] {fname}', 'magma', -1, 1),
            (axes[1], recon_full, 'Reconstruction',          'magma', -1, 1),
            (axes[2], diff,       'Difference',     'RdBu_r',
             -np.abs(diff).max(), np.abs(diff).max()),
        ]:
            im = ax.imshow(data, aspect='auto', origin='lower',
                           cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=8)
            ax.set_xlabel('Time frame'); ax.set_ylabel('Mel bin')
        fig.tight_layout()
        out_path = os.path.join(out_dir, f'{row:03d}_{lbl}_{os.path.splitext(fname)[0]}_fullfile.png')
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved  {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SpecVAE evaluation')
    parser.add_argument('-r', '--resume', required=True,
                        help='path to checkpoint')
    parser.add_argument('-c', '--config', default=None,
                        help='config file (inferred from checkpoint dir if omitted)')
    parser.add_argument('-d', '--device', default=None)
    parser.add_argument('-n', '--n_samples', type=int, default=6,
                        help='number of spectrogram chunks to visualise/save as audio')
    parser.add_argument('-o', '--out_dir', default='eval/run',
                        help='directory for output files (default: eval/run/)')

    args = parser.parse_args()
    config = ConfigParser(parser)
    main(config, args.resume, args.n_samples, args.out_dir)
