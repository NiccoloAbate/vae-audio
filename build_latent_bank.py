"""
Build a latent bank: a representative spread of drum latents encoded from the
dataset, for the synthesis engine (drum_engine.py) and the eventual plugin.

Per drum type it encodes up to --encode_per_type hits, then farthest-point-
samples --per_type of them (seeded at the medoid) so the bank is diverse rather
than redundant. Stores the FULL temporal latent (D, T_lat) per hit — averaging
would push the decoder off-distribution.

Output: a .pt dict { latents (N,D,T_lat) float32, types [str], files [str],
                      sr, chunk_size, latent_dim, checkpoint }.

Usage:
    python build_latent_bank.py -r saved/models/DrumOneShots/checkpoint-epoch1500.pth \
        --data_dir dataset/DrumOneShots --per_type 8 -o bank/drums_bank.pt
"""
import argparse
import os
import numpy as np
import torch

import model.model as module_arch
from dataset.datasets import OneShotAudioDataset
from drum_types import classify, TYPES


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_vae(checkpoint):
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt['config']['arch']
    vae = getattr(module_arch, cfg['type'])(**cfg['args'])
    vae.load_state_dict(ckpt['state_dict'], strict=False)
    vae.eval()
    dl = dict(ckpt['config']['data_loader']['args'])
    return vae, dl


def fps(X, k, seed_idx=None):
    """Farthest-point sampling on rows of X (n,d). Returns up to k indices."""
    n = len(X)
    if n <= k:
        return list(range(n))
    if seed_idx is None:                       # medoid: nearest to centroid
        seed_idx = int(((X - X.mean(0)) ** 2).sum(1).argmin())
    sel = [seed_idx]
    dist = ((X - X[seed_idx]) ** 2).sum(1)
    while len(sel) < k:
        i = int(dist.argmax())
        sel.append(i)
        dist = np.minimum(dist, ((X - X[i]) ** 2).sum(1))
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--data_dir', default='dataset/DrumOneShots')
    ap.add_argument('--per_type', type=int, default=8,
                    help='representatives kept per drum type')
    ap.add_argument('--encode_per_type', type=int, default=48,
                    help='hits encoded per type before FPS down to --per_type')
    ap.add_argument('--types', nargs='*', default=None,
                    help='restrict to these types (default: all found)')
    ap.add_argument('-o', '--output', default='bank/drums_bank.pt')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = get_device()
    print(f'Device: {device}')
    vae, dl = load_vae(args.resume)
    vae.to(device)
    sr, chunk = dl.get('sr', 22050), dl.get('chunk_size', 110080)

    dataset = OneShotAudioDataset(
        [args.data_dir], sr=sr, chunk_size=chunk,
        onset_thresh=dl.get('onset_thresh', 0.02),
        pre_pad=dl.get('pre_pad', 256), peak_norm=dl.get('peak_norm', True))

    # Bucket dataset indices by type
    by_type = {}
    for i, f in enumerate(dataset.files):
        by_type.setdefault(classify(f), []).append(i)
    wanted = args.types or [t for t in TYPES if t in by_type and t != 'exclude']

    rng = np.random.default_rng(args.seed)
    bank_lat, bank_type, bank_file = [], [], []

    for t in wanted:
        idxs = by_type.get(t, [])
        if not idxs:
            continue
        pick = rng.choice(idxs, min(args.encode_per_type, len(idxs)), replace=False)

        pooled, full, files = [], [], []
        with torch.no_grad():
            for j in pick:
                _, _, y = dataset[int(j)]
                mu, _, _ = vae.encode(y.float().unsqueeze(0).to(device))  # (1,D,T)
                mu = mu.squeeze(0).cpu()
                full.append(mu)
                pooled.append(mu.mean(dim=1).numpy())                     # (D,)
                files.append(dataset.files[int(j)])

        keep = fps(np.stack(pooled), args.per_type)
        for k in keep:
            bank_lat.append(full[k])
            bank_type.append(t)
            bank_file.append(files[k])
        print(f'  {t:>10}: encoded {len(pick):3d}, kept {len(keep)}')

    latents = torch.stack(bank_lat)  # (N, D, T)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save({
        'latents': latents, 'types': bank_type, 'files': bank_file,
        'sr': sr, 'chunk_size': chunk, 'latent_dim': vae.latent_dim,
        'checkpoint': os.path.abspath(args.resume),
    }, args.output)
    print(f'\nSaved bank: {latents.shape} ({len(set(bank_type))} types) -> {args.output}')


if __name__ == '__main__':
    main()
