"""
Encode N random chunks and save a latent bank for the CelloVAE JUCE plugin.

Stores the FULL temporal latent (latent_dim, T_lat) per chunk — not averaged.
This keeps the decoder in-distribution and preserves per-chunk timbre.

Binary format:
  bytes 0-3:   N          (uint32) number of latents
  bytes 4-7:   D          (uint32) latent_dim  (32)
  bytes 8-11:  T          (uint32) T_lat       (64)
  bytes 12+:   N*D*T      float32, row-major (each row is D*T floats)

Usage:
    python export_latents.py -r saved/models/.../checkpoint-epoch1340.pth -n 32
"""
import argparse, struct, sys
import torch
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import model.model as module_arch


def load_vae(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg  = ckpt['config']['arch']
    vae  = getattr(module_arch, cfg['type'])(**cfg['args'])
    vae.load_state_dict(ckpt['state_dict'], strict=False)
    vae.eval()
    return vae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume',   required=True)
    parser.add_argument('-n', '--n_chunks', type=int, default=32)
    parser.add_argument('--npy', default='/home/niccoloabate/Datasets/CCBachCello_chunks.npy')
    parser.add_argument('-o', '--output',   default=None)
    parser.add_argument('--seed',           type=int, default=42)
    args = parser.parse_args()

    ckpt_path = Path(args.resume)
    out_path  = Path(args.output) if args.output else ckpt_path.parent / 'latents.latents'

    np.random.seed(args.seed)
    data = np.load(args.npy, mmap_mode='r')
    idx  = np.random.choice(len(data), args.n_chunks, replace=False)

    vae = load_vae(str(ckpt_path))

    latents = []
    with torch.no_grad():
        for i in idx:
            chunk = torch.from_numpy(data[i].copy()).float().unsqueeze(0).unsqueeze(0)
            mu, _, _ = vae.encode(chunk)   # (1, D, T)
            latents.append(mu.squeeze(0))  # (D, T)

    bank = torch.stack(latents, dim=0).numpy().astype(np.float32)  # (N, D, T)
    N, D, T = bank.shape

    with open(out_path, 'wb') as f:
        f.write(struct.pack('III', N, D, T))
        f.write(bank.tobytes())

    print(f"Saved {N} latents  shape=({D},{T})  ->  {out_path}")
    print(f"Latent range: [{bank.min():.3f}, {bank.max():.3f}]")


if __name__ == '__main__':
    main()
