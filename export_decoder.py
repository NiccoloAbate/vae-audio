"""
Export a TorchScript decoder for use in the CelloVAE JUCE plugin.

The decoder accepts the full temporal latent (1, latent_dim, T_lat) directly.
T_lat = chunk_size / prod(strides) = 16384 / 256 = 64.

Usage:
    python export_decoder.py -r saved/models/RawAudioVAE_Adv_Cello/0520_122948/checkpoint-epoch1340.pth
"""
import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import model.model as module_arch


class TempDecoder(nn.Module):
    """
    Wraps RawAudioVAE.decode() with the native temporal latent interface.

    Input:  (1, latent_dim, T_lat)   — e.g. (1, 32, 64)
    Output: (1, 1, chunk_size)       — mono waveform in [-1, 1] at 22050 Hz
    """
    def __init__(self, vae: module_arch.RawAudioVAE):
        super().__init__()
        self.decoder = vae.decoder
        self.pqmf    = vae.pqmf

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder(z)
        if self.pqmf is not None:
            x = self.pqmf.synthesis(x)
        return x  # (1, 1, chunk_size)


def load_vae(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg  = ckpt['config']['arch']
    vae  = getattr(module_arch, cfg['type'])(**cfg['args'])
    vae.load_state_dict(ckpt['state_dict'], strict=False)
    vae.eval()
    return vae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', required=True)
    parser.add_argument('-o', '--output', default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.resume)
    out_path  = Path(args.output) if args.output else ckpt_path.parent / 'decoder.pt'

    print(f"Loading: {ckpt_path}")
    vae     = load_vae(str(ckpt_path))
    t_lat   = 16384 // (4 * 4 * 4 * 4)   # 64
    wrapper = TempDecoder(vae).eval()

    with torch.no_grad():
        dummy  = torch.zeros(1, vae.latent_dim, t_lat)
        output = wrapper(dummy)
    print(f"  input  : (1, {vae.latent_dim}, {t_lat})")
    print(f"  output : {tuple(output.shape)}  (expected (1, 1, 16384))")

    traced = torch.jit.trace(wrapper, dummy)
    traced.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
