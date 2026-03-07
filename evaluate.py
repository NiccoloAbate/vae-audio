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


def select_sample_indices(filenames, chunk_indices, n_samples, labels=None):
    """Pick one chunk per unique label (or file if no labels), preferring the middle chunk.
    Ensures diversity across categories when labels are provided.
    Returns a list of integer indices into the chunk arrays."""
    # Group by label first, then by file within each label
    key_to_file_to_indices = {}
    for i, (fname, cidx) in enumerate(zip(filenames, chunk_indices)):
        lbl = labels[i] if labels is not None else fname
        key_to_file_to_indices.setdefault(lbl, {}).setdefault(fname, []).append(i)

    selected = []
    # First pass: one file per label for diversity
    # Second pass: additional files from each label until n_samples reached
    all_file_idxs = [
        idxs[len(idxs) // 2]
        for file_map in key_to_file_to_indices.values()
        for idxs in file_map.values()
    ]
    label_first = [
        next(iter(file_map.values()))[len(next(iter(file_map.values()))) // 2]
        for file_map in key_to_file_to_indices.values()
    ]
    # Add one per label first, then fill remaining slots from all files
    seen = set()
    for idx in label_first + all_file_idxs:
        if idx not in seen:
            seen.add(idx)
            selected.append(idx)
        if len(selected) >= n_samples:
            break
    return selected


_vocos_model = None

def _get_vocos():
    global _vocos_model
    if _vocos_model is None:
        from vocos import Vocos
        _vocos_model = Vocos.from_pretrained('charactr/vocos-mel-24khz')
        _vocos_model.eval()
    return _vocos_model


def mel_to_audio(spec_norm, sr=24000):
    """Convert a normalised safe_log mel spectrogram (-1, 1) to a waveform via Vocos.

    Denormalises with the inverse of SafeLogNorm: safe_log = spec_norm * 8 - 3.
    """
    vocos = _get_vocos()
    mel = torch.tensor(spec_norm * 8.0 - 3.0, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, T)
    with torch.no_grad():
        y = vocos.decode(mel)
    return y.squeeze(0).numpy()


def reconstruct_full_file(model, npy_path, device, chunk_size=47, sr=24000):
    """
    Reconstruct a full audio file by:
      1. Loading the pre-computed mel spectrogram (.npy, shape (n_mels, N_frames))
      2. Normalising to (-1, 1) via SafeLogNorm
      3. Splitting into non-overlapping chunks of chunk_size frames
      4. Passing each chunk through the VAE
      5. Concatenating reconstructed chunks along the time axis
      6. Decoding both original and reconstruction to audio via Vocos

    Returns (y_orig, y_recon, spec_norm, recon_full) — all numpy arrays.
    """
    spec = np.load(npy_path)                       # (n_mels, N_frames), safe_log
    spec_norm = (spec + 3.0) / 8.0                 # normalise to (-1, 1)

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

    # Decode normalised mel → audio via Vocos
    y_orig  = mel_to_audio(orig_full,  sr=sr)
    y_recon = mel_to_audio(recon_full, sr=sr)

    return y_orig, y_recon, orig_full, recon_full


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_spectrograms(inputs, recons, labels, filenames, chunk_indices,
                      sample_indices, out_path):
    """Plot input vs reconstruction for selected chunks (one per file)."""
    n = len(sample_indices)
    fig = plt.figure(figsize=(14, 3 * n))
    gs = gridspec.GridSpec(n, 3, figure=fig, width_ratios=[1, 1, 1], hspace=0.5, wspace=0.35)

    for row, i in enumerate(sample_indices):
        inp  = inputs[i].numpy()
        rec  = recons[i].numpy()
        diff = inp - rec

        ax_in  = fig.add_subplot(gs[row, 0])
        ax_rec = fig.add_subplot(gs[row, 1])
        ax_dif = fig.add_subplot(gs[row, 2])

        im0 = ax_in.imshow(inp,  aspect='auto', origin='lower', cmap='magma')
        im1 = ax_rec.imshow(rec, aspect='auto', origin='lower', cmap='magma')
        im2 = ax_dif.imshow(diff, aspect='auto', origin='lower', cmap='RdBu_r',
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

    plt.suptitle('SpecVAE: input vs reconstruction  (both normalised to (−1, 1))', fontsize=10)
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
        ax.scatter(z2[mask, 0], z2[mask, 1], c=[colour_map(label_to_idx[lbl])], label=lbl, alpha=0.6, s=15)
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
                       sr=24000):
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
            model, npy_path, device, sr=sr)

        stem = os.path.splitext(fname)[0]
        tag  = f'{row:03d}_{lbl}_{stem}'
        sf.write(os.path.join(out_dir, f'{tag}_mel_orig.wav'),
                 y_orig  / (np.abs(y_orig).max()  + 1e-8), sr)
        sf.write(os.path.join(out_dir, f'{tag}_recon.wav'),
                 y_recon / (np.abs(y_recon).max() + 1e-8), sr)

        # Try to save the actual source wav for a true quality comparison
        # npy path: .../medley_subset/{dataset_name}/{split}/{label}/{stem}.npy
        # wav path: .../medley_subset/audio/{split}/{label}/{stem}.wav
        try:
            import pathlib
            npy = pathlib.Path(npy_path)
            # npy: .../{dataset_root}/{dataset_name}/{split}/{label}/{stem}.npy
            # wav: .../{dataset_root}/{src_dir}/{split}/{label}/{stem}.wav
            # Try common source directory names (audio, organized)
            dataset_root = npy.parent.parent.parent.parent
            rel = pathlib.Path(npy.parent.parent.name) / npy.parent.name / (npy.stem + '.wav')
            wav_path = next(
                (dataset_root / src / rel for src in ('audio', 'organized')
                 if (dataset_root / src / rel).exists()),
                dataset_root / 'audio' / rel  # fallback (may not exist)
            )
            if wav_path.exists():
                import torchaudio as _ta
                y_src, src_sr = _ta.load(str(wav_path))
                if src_sr != sr:
                    y_src = _ta.functional.resample(y_src, src_sr, sr)
                if y_src.shape[0] > 1:
                    y_src = y_src.mean(0, keepdim=True)
                y_src = y_src.squeeze().numpy()
                sf.write(os.path.join(out_dir, f'{tag}_source.wav'),
                         y_src / (np.abs(y_src).max() + 1e-8), sr)
            else:
                print(f'  (source wav not found at {wav_path})')
        except Exception as e:
            print(f'  (could not save source wav: {e})')

        row += 1

    print(f'Saved  {row} full-file audio triplets → {out_dir}/'
          '  (source = actual wav, mel_orig = mel→Vocos, recon = VAE→Vocos)')


# ---------------------------------------------------------------------------
# latent space interpolation & traversal
# ---------------------------------------------------------------------------

def slerp(z1, z2, t):
    """
    Spherical linear interpolation between two latent vectors.

    Why slerp instead of lerp?
    In a d-dimensional isotropic Gaussian N(0,I), probability mass concentrates
    on a thin shell of radius √d as d grows.  A straight line (lerp) between two
    points on that shell cuts through the low-probability interior.  Slerp stays
    on the shell, keeping every interpolated point in a high-probability region.

    Formula:  z(t) = sin((1-t)ω)/sin(ω) · z1 + sin(t·ω)/sin(ω) · z2
    where ω = arccos(z1̂ · z2̂)  (angle between unit vectors).
    Falls back to lerp when z1 ≈ z2 (ω ≈ 0).
    """
    z1_n = z1 / (z1.norm() + 1e-8)
    z2_n = z2 / (z2.norm() + 1e-8)
    omega = torch.acos(torch.clamp((z1_n * z2_n).sum(), -1.0, 1.0))
    if omega.abs() < 1e-6:
        return (1 - t) * z1 + t * z2
    return (torch.sin((1 - t) * omega) * z1 +
            torch.sin(t * omega) * z2) / torch.sin(omega)


def get_file_latent(model, npy_path, device, chunk_size=47):
    """Encode the middle chunk of a file and return its latent mean (D,)."""
    spec = np.load(npy_path)
    spec_norm = (spec + 3.0) / 8.0
    n_chunks = spec_norm.shape[1] // chunk_size
    mid = n_chunks // 2
    chunk = spec_norm[:, mid*chunk_size:(mid+1)*chunk_size]
    x = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        mu, _, _ = model.encode(x)
    return mu.squeeze(0).cpu()


def decode_latent(model, z, device):
    """Decode a single latent vector (D,) → spectrogram (F, T) on CPU."""
    with torch.no_grad():
        x_recon = model.decode(z.unsqueeze(0).to(device))
    return x_recon.squeeze(0).cpu().numpy()


def plot_interpolations(model, data_loader, labels, filenames, chunk_indices,
                        sample_indices, device, n_steps=8, out_dir='.'):
    """
    For each consecutive pair in sample_indices (from different files), interpolate
    between their latent means using both lerp and slerp.  Saves:
      - interp_<i>_<j>_grid.png  : spectrogram grid (lerp top, slerp bottom)
      - interp_audio/             : wav files at each lerp step
    """
    path_map = {os.path.basename(str(p)): str(p)
                for p in data_loader.dataset.path_to_data}

    # Collect one latent per unique file from sample_indices
    seen, file_latents, file_meta = set(), [], []
    for i in sample_indices:
        fname = filenames[i]
        if fname in seen:
            continue
        seen.add(fname)
        npy_path = path_map.get(fname)
        if npy_path is None:
            continue
        z = get_file_latent(model, npy_path, device)
        file_latents.append(z)
        file_meta.append((fname, labels[i]))

    if len(file_latents) < 2:
        print('  Not enough files for interpolation.')
        return

    ts = np.linspace(0, 1, n_steps)
    audio_dir = os.path.join(out_dir, 'interp_audio')
    os.makedirs(audio_dir, exist_ok=True)

    # Pair up consecutive files
    for pair_idx in range(len(file_latents) - 1):
        z1, z2 = file_latents[pair_idx], file_latents[pair_idx + 1]
        fname1, lbl1 = file_meta[pair_idx]
        fname2, lbl2 = file_meta[pair_idx + 1]
        stem1, stem2 = os.path.splitext(fname1)[0], os.path.splitext(fname2)[0]

        lerp_specs, slerp_specs = [], []
        for t in ts:
            t_t = torch.tensor(t, dtype=torch.float32)
            z_lerp  = (1 - t_t) * z1 + t_t * z2
            z_slerp = slerp(z1, z2, t_t)
            lerp_specs.append(decode_latent(model, z_lerp,  device))
            slerp_specs.append(decode_latent(model, z_slerp, device))

        # Spectrogram grid: lerp (top row) and slerp (bottom row)
        fig, axes = plt.subplots(2, n_steps, figsize=(2.2 * n_steps, 5),
                                 sharex=True, sharey=True)
        vmin, vmax = -1, 1
        for col, t in enumerate(ts):
            for row, (specs, method) in enumerate([(lerp_specs, 'lerp'),
                                                   (slerp_specs, 'slerp')]):
                ax = axes[row, col]
                ax.imshow(specs[col], aspect='auto', origin='lower',
                          cmap='magma', vmin=vmin, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(method, fontsize=9)
                if row == 0:
                    ax.set_title(f't={t:.2f}', fontsize=7)
        fig.suptitle(f'Interpolation:  [{lbl1}] {stem1}  →  [{lbl2}] {stem2}',
                     fontsize=9)
        fig.tight_layout()
        grid_path = os.path.join(out_dir,
                                 f'interp_{pair_idx:02d}_{stem1}_to_{stem2}_grid.png')
        fig.savefig(grid_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved  {grid_path}')

        # Audio: lerp steps only
        for col, (t, spec) in enumerate(zip(ts, lerp_specs)):
            y = mel_to_audio(spec)
            wav_path = os.path.join(
                audio_dir,
                f'interp_{pair_idx:02d}_{stem1}_to_{stem2}_t{t:.2f}.wav')
            sf.write(wav_path, y / (np.abs(y).max() + 1e-8), 24000)
    print(f'Saved  interpolation audio → {audio_dir}/')


def compute_smoothness(model, mu_all, device, n_pairs=20, n_steps=16):
    """
    Quantitative smoothness score: average rate of change of decoded output
    per unit step in latent space along linear interpolation paths.

        S = mean_{pairs} mean_{steps} ‖decode(z_{t+1}) − decode(z_t)‖_F
                                      ─────────────────────────────────
                                          ‖z_{t+1} − z_t‖_2

    A lower score means the decoder changes more smoothly with latent position.
    Computed on random pairs drawn from the test set latent means.
    """
    N = mu_all.size(0)
    rng = np.random.default_rng(0)
    idx_a = rng.choice(N, n_pairs, replace=False)
    idx_b = rng.choice(N, n_pairs, replace=False)

    ts = torch.linspace(0, 1, n_steps + 1)
    scores = []
    with torch.no_grad():
        for a, b in zip(idx_a, idx_b):
            z1, z2 = mu_all[a], mu_all[b]
            step_latent = (z2 - z1).norm().item() / n_steps
            if step_latent < 1e-8:
                continue
            prev = decode_latent(model, (ts[0] * z2 + (1 - ts[0]) * z1), device)
            for t in ts[1:]:
                curr = decode_latent(model, (t * z2 + (1 - t) * z1), device)
                spec_change = np.linalg.norm(curr - prev, 'fro')
                scores.append(spec_change / step_latent)
                prev = curr

    score = float(np.mean(scores))
    print(f'  Smoothness score (lower = smoother): {score:.4f}')
    return score


def plot_pca_traversal(model, mu_all, device, n_components=3, n_steps=7, out_dir='.'):
    """
    Traverse the top PCA directions of the latent space, decoding at each step.

    Starting from the dataset mean, move ±3σ along each principal component.
    This reveals what each learned direction encodes perceptually.

    Produces one PNG per PC showing decoded spectrograms at each position.
    """
    pca = PCA(n_components=n_components)
    pca.fit(mu_all.numpy())
    mu_mean = torch.tensor(pca.mean_, dtype=torch.float32)

    for pc_idx in range(n_components):
        direction = torch.tensor(pca.components_[pc_idx], dtype=torch.float32)
        sigma = float(np.sqrt(pca.explained_variance_[pc_idx]))
        alphas = np.linspace(-3 * sigma, 3 * sigma, n_steps)

        specs = []
        for alpha in alphas:
            z = mu_mean + alpha * direction
            specs.append(decode_latent(model, z, device))

        var_ratio = pca.explained_variance_ratio_[pc_idx] * 100
        fig, axes = plt.subplots(1, n_steps, figsize=(2.2 * n_steps, 3),
                                 sharex=True, sharey=True)
        for col, (alpha, spec) in enumerate(zip(alphas, specs)):
            ax = axes[col]
            ax.imshow(spec, aspect='auto', origin='lower',
                      cmap='magma', vmin=-1, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{alpha:+.1f}', fontsize=7)
        axes[0].set_ylabel('Mel bin', fontsize=8)
        fig.suptitle(f'PC {pc_idx + 1}  ({var_ratio:.1f}% variance)  '
                     f'— traversal from −3σ to +3σ', fontsize=9)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f'pca_traversal_pc{pc_idx + 1}.png')
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved  {out_path}')


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

    sample_indices = select_sample_indices(filenames, chunk_indices, n_samples, labels=labels)
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

    print('\nLatent space analysis…')
    print('  Interpolations:')
    plot_interpolations(model, data_loader, labels, filenames, chunk_indices,
                        sample_indices, device, n_steps=8, out_dir=out_dir)

    print('  Smoothness score:')
    compute_smoothness(model, mu, device)

    print('  PCA traversal:')
    plot_pca_traversal(model, mu, device, n_components=3, out_dir=out_dir)

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
    parser.add_argument('-r', '--resume', required=True, help='path to checkpoint')
    parser.add_argument('-c', '--config', default=None, help='config file (inferred from checkpoint dir if omitted)')
    parser.add_argument('-d', '--device', default=None)
    parser.add_argument('-n', '--n_samples', type=int, default=6, help='number of spectrogram chunks to visualise/save as audio')
    parser.add_argument('-o', '--out_dir', default='eval/run', help='directory for output files (default: eval/run/)')

    args = parser.parse_args()
    config = ConfigParser(parser)
    main(config, args.resume, args.n_samples, args.out_dir)
