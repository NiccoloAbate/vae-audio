"""
Export a drum kit for the plugin (drum_plugin/): one TorchScript file that renders
hits from a latent recipe, plus a JSON sidecar describing the bank and controls.

    kit.pt    KitRenderer, scripted: the traced decoder + the bank (mu, sigma),
              control directions, variation axes and fixed noise tables as buffers.
              render(...) runs the app's chain (drum_engine.py / app_drums.py):
                morph A -> B (blend / slerp / aligned / tail / crossover)
                -> controls + variation offset on the sounding frames
                -> punch -> decay stretch -> sampled decode (mu + sigma * eps) -> gate
              for a batch of L rows (e.g. velocity layers), returning (L, samples).
              encode(x) turns a prepared hit (mono, 44.1 kHz, onset-aligned,
              peak 0.95, `samples` long; see the preprocessing fields in kit.json)
              into (mu, sigma, timing) for render_z(), so dropped samples go
              through the same chain as bank hits.
    kit.json  sr, samples, preprocessing, bank entries (name, type, file), control
              names, seeds, and the checkpoint (latents are only valid for it).

Randomness comes from tables made with the app's seeds (torch.manual_seed(s)), so a
plugin seed s sounds like the app's seed s. Two differences from the app: the
sounding-frames mask comes from one decode of the morphed latent for all
controls together (as in the app), and nothing is trimmed or normalised (the
plugin does that). Rendering costs 1-4 decodes per row (mask, peak for punch,
peak for decay, final).

    python export_kit.py -r runs/.../checkpoint-epoch455.pth -b bank/v3_B1_ep455.pt \\
        --directions sweeps/v3_B1_ep455/directions.pt -o kits/v3_B1
"""
import argparse
import json
import os

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from drum_engine import DrumEngine, MORPH_MODES

MODES = MORPH_MODES                     # index = render(mode=...)


# ---- the engine's operations, TorchScript-compatible (see drum_engine.py) ----

def _frames(N: int, T: int, device: torch.device) -> torch.Tensor:
    return torch.arange(T, dtype=torch.float32, device=device).unsqueeze(0).expand(N, T)


def warp(z: torch.Tensor, src: torch.Tensor, pad: torch.Tensor, fade: float = 8.0) -> torch.Tensor:
    N, D, T = z.shape
    s = src.clamp(0, T - 1)
    lo = s.floor().long()
    hi = (lo + 1).clamp(max=T - 1)
    w = (s - lo.float()).unsqueeze(1)
    out = torch.gather(z, 2, lo.unsqueeze(1).expand(N, D, T)) * (1 - w) \
        + torch.gather(z, 2, hi.unsqueeze(1).expand(N, D, T)) * w
    a = ((src - (T - 1)) / fade).clamp(0, 1).unsqueeze(1)
    return out * (1 - a) + pad.view(1, -1, 1) * a


def frame_rms(y: torch.Tensor, T: int) -> torch.Tensor:
    N, L = y.shape
    hop = L // T
    return y[:, :hop * T].reshape(N, T, hop).pow(2).mean(2).sqrt()


def frame_activity(y: torch.Tensor, T: int, thresh_db: float = -50.0, dilate: int = 2) -> torch.Tensor:
    rms = frame_rms(y, T)
    m = (rms > rms.max(1, keepdim=True).values * 10 ** (thresh_db / 20)).float()
    return F.max_pool1d(m.unsqueeze(1), 2 * dilate + 1, 1, dilate).squeeze(1)


def align_src(peak: torch.Tensor, end: torch.Tensor, p: torch.Tensor, L: torch.Tensor, T: int) -> torch.Tensor:
    t = _frames(1, T, peak.device)
    pk, pp = peak.view(1, 1), p.view(1, 1)
    scale = ((end - peak) / L).view(1, 1)
    return torch.where(t <= pp, t - (pp - pk), pk + (t - pp) * scale)


def ramp(t: torch.Tensor, centre: torch.Tensor, width: float) -> torch.Tensor:
    return ((t - centre) / max(width, 1e-3) + 0.5).clamp(0, 1)


def noise_gate(y: torch.Tensor, gate_db: float, win: int = 1024, rmp: int = 441) -> torch.Tensor:
    x = y.unsqueeze(1)
    k = torch.hann_window(win, dtype=y.dtype, device=y.device)
    env = F.conv1d(x.pow(2), (k / k.sum()).view(1, 1, -1), padding=win // 2)
    env = env[..., :y.shape[-1]].sqrt()
    peak = y.abs().amax(-1, keepdim=True).unsqueeze(1)
    m = (env > peak * 10 ** (gate_db / 20)).to(y.dtype)
    m = F.avg_pool1d(m, rmp, 1, rmp // 2, count_include_pad=False)
    return (x * m[..., :y.shape[-1]]).squeeze(1)


class EncoderOnly(nn.Module):
    """RawAudioVAE.encode without the sampling: x (B, 1, L) -> mu, logvar (B, D, T)."""
    def __init__(self, vae):
        super().__init__()
        self.encoder, self.mu_proj, self.logvar_proj = vae.encoder, vae.mu_proj, vae.logvar_proj

    def forward(self, x):
        h = self.encoder(x)
        return self.mu_proj(h), self.logvar_proj(h).clamp(-30.0, 20.0)


class KitRenderer(nn.Module):
    def __init__(self, encoder, decoder, latents, sigmas, timing_peak, timing_end, z_silence,
                 directions, var_std, var_vt, var_normals, eps, sr: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.register_buffer('mu', latents)                # (N, D, T)
        self.register_buffer('sigma', sigmas)              # (N, D, T)
        self.register_buffer('peak', timing_peak.float())  # (N,)  sampled-decode timing
        self.register_buffer('end', timing_end.float())    # (N,)
        self.register_buffer('z_silence', z_silence)       # (2D,) [mu; sigma]
        self.register_buffer('directions', directions)     # (K, D) control vectors
        self.register_buffer('var_std', var_std)           # (P,)  bank spread along its PCs
        self.register_buffer('var_vt', var_vt)             # (P, D)
        self.register_buffer('var_normals', var_normals)   # (S, P) = randn(P) with seed s
        self.register_buffer('eps', eps)                   # (S, D, T) = randn(1, D, T) with seed s
        self.sr = sr
        self.D = latents.shape[1]
        self.T = latents.shape[2]

    def decode(self, z: torch.Tensor, seed: int) -> torch.Tensor:
        """Augmented (B, 2D, T) -> (B, samples), mu + sigma * eps[seed]."""
        D = self.D
        zz = z[:, :D] + z[:, D:].clamp_min(0) * self.eps[seed].unsqueeze(0)
        return self.decoder(zz).squeeze(1)

    def aug(self, i: int) -> torch.Tensor:
        return torch.cat([self.mu[i], self.sigma[i]], 0)

    @torch.jit.export
    def entry(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bank hit i: augmented latent (2D, T) and timing (peak, end) frames."""
        return self.aug(i), torch.stack([self.peak[i], self.end[i]])

    @torch.jit.export
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepared hit x (samples,) -> mu (D, T), sigma (D, T), timing (2,) = peak and
        end frame (last frame within 30 dB of the peak) of its sampled decode."""
        mu, logvar = self.encoder(x.to(self.mu.device).view(1, 1, -1))
        sigma = (0.5 * logvar).exp()
        y = self.decode(torch.cat([mu, sigma], 1), 0)
        rms = frame_rms(y, self.T)[0]
        peak = rms.argmax()
        loud = (rms >= rms.max() * 10 ** (-30.0 / 20)).long()
        end = (loud * torch.arange(self.T, device=rms.device)).argmax()
        end = torch.maximum(end, peak + 1)
        return mu[0], sigma[0], torch.stack([peak, end]).float()

    def morph(self, za: torch.Tensor, zb: torch.Tensor, ta: torch.Tensor, tb: torch.Tensor,
              t: float, mode: int, width: float) -> torch.Tensor:
        D, T = self.D, self.T
        if mode == 0 or (mode >= 2 and (t <= 0.0 or (t >= 1.0 and mode != 3))):
            if mode >= 2:
                return za.clone() if t <= 0.0 else zb.clone()
            return za * (1 - t) + zb * t                             # blend
        if mode == 1:                                                # slerp on mu
            out = za * (1 - t) + zb * t
            ua, ub = za[:D], zb[:D]
            na, nb = ua / (ua.norm() + 1e-8), ub / (ub.norm() + 1e-8)
            omega = torch.acos(torch.clamp((na * nb).sum(), -1.0, 1.0))
            if float(omega.abs()) >= 1e-6:
                s = torch.sin(omega)
                out[:D] = (torch.sin((1 - t) * omega) * ua + torch.sin(t * omega) * ub) / s
            return out
        pa, ea, pb, eb = ta[0], ta[1], tb[0], tb[1]
        La, Lb = ea - pa, eb - pb
        L = torch.exp((1 - t) * La.log() + t * Lb.log())
        p = pa if mode == 3 else (1 - t) * pa + t * pb
        Za = warp(za.unsqueeze(0), align_src(pa, ea, p, L, T), self.z_silence)[0]
        Zb = warp(zb.unsqueeze(0), align_src(pb, eb, p, L, T), self.z_silence)[0]
        tt = torch.arange(T, dtype=torch.float32, device=za.device)
        if mode == 2:
            c = torch.full((T,), t, device=za.device)
        elif mode == 3:
            c = t * ramp(tt, p + 2, width)
        else:
            tau = (1 - t) * (p + L + width / 2) - t * width / 2
            c = ramp(tt, tau, width)
        return Za * (1 - c) + Zb * c

    @torch.jit.export
    def render(self, a: int, b: int, mode: int, width: float, morph: torch.Tensor,
               controls: torch.Tensor, var_amount: torch.Tensor, var_seed: torch.Tensor,
               decay: torch.Tensor, punch: torch.Tensor, decode_seed: int,
               gate_db: float) -> torch.Tensor:
        """render_z() for bank hits a, b."""
        za, ta = self.entry(a)
        zb, tb = self.entry(b)
        return self.render_z(za, zb, ta, tb, mode, width, morph, controls, var_amount, var_seed,
                             decay, punch, decode_seed, gate_db)

    @torch.jit.export
    def render_z(self, za: torch.Tensor, zb: torch.Tensor, ta: torch.Tensor, tb: torch.Tensor,
                 mode: int, width: float, morph: torch.Tensor,
                 controls: torch.Tensor, var_amount: torch.Tensor, var_seed: torch.Tensor,
                 decay: torch.Tensor, punch: torch.Tensor, decode_seed: int,
                 gate_db: float) -> torch.Tensor:
        """Hits A, B as augmented latents [mu; sigma] (2D, T) with timing (2,) from
        entry() or encode(). L rows: morph (L,), controls (L, K) in direction units,
        var_amount (L,), var_seed (L,) long, decay (L,) as log2 tail factor, punch (L,)
        frames. gate_db <= -200 turns the gate off. Returns (L, samples)."""
        L = int(morph.shape[0])
        D, T = self.D, self.T
        dev = self.mu.device
        za, zb, ta, tb = za.to(dev), zb.to(dev), ta.to(dev), tb.to(dev)
        z = torch.stack([self.morph(za, zb, ta, tb, float(morph[i]), mode, width) for i in range(L)])
        # controls + variation: one constant offset of mu, on the sounding frames
        off = controls.to(dev) @ self.directions                                   # (L, D)
        seeds = var_seed.to(dev).long()
        off = off + var_amount.to(dev).unsqueeze(1) * \
            ((self.var_normals[seeds] * self.var_std.unsqueeze(0)) @ self.var_vt)
        if bool((off != 0).any()):
            mask = frame_activity(self.decode(z, decode_seed), T)                  # (L, T)
            z = torch.cat([z[:, :D] + off.unsqueeze(2) * mask.unsqueeze(1), z[:, D:]], 1)
        # punch, then decay stretch, around each row's peak frame
        pu, de = punch.to(dev), decay.to(dev)
        if bool((pu != 0).any()) or bool((de != 0).any()):
            pk = frame_rms(self.decode(z, decode_seed), T).argmax(1).float()     # (L,)
            t = _frames(L, T, dev)
            k = pu.clamp_min(-0.9).unsqueeze(1)
            p = pk.unsqueeze(1)
            src = torch.where(t <= p, t, torch.where(t <= p + 1 + k, p + (t - p) / (1 + k), t - k))
            src = torch.where((pu != 0).unsqueeze(1), src, t)
            z = warp(z, src, self.z_silence)
            if bool((pu != 0).any()) and bool((de != 0).any()):       # the app re-finds the peak
                p = frame_rms(self.decode(z, decode_seed), T).argmax(1).float().unsqueeze(1)
            r = torch.pow(2.0, de).unsqueeze(1)
            c = p + 2
            src = torch.where(t > c, c + (t - c) / r, t)
            src = torch.where((de != 0).unsqueeze(1), src, t)
            z = warp(z, src, self.z_silence)
        y = self.decode(z, decode_seed)
        if gate_db > -200.0:
            y = noise_gate(y, gate_db)
        return y

    def forward(self, morph: torch.Tensor) -> torch.Tensor:          # unused; scripting needs one
        return morph


def build(args):
    dev = torch.device('cpu')
    e = DrumEngine(checkpoint=args.resume, bank=args.bank, device=dev)
    assert e.has_sigma, 'bank has no sigmas: build_latent_bank.py --refresh'
    e.set_sample(True, 0)
    (peak, end), z_sil = e._bank_timing()[0:2], e.z_silence
    _, std, vt = e._bank_spread()

    keys, names = [], []
    if args.directions:
        keys = e.load_directions(args.directions, args.knobs)
        names = [e.knobs[k]['analogue'] for k in keys]
    dirs = [e.knobs[k]['vector'] for k in keys]
    for k in range(args.explore_pcs):                           # unlabelled bank PCs
        dirs.append(std[k] * vt[k]); keys.append(f'active/pc/{k + 1}'); names.append(f'PC{k + 1}')
    directions = torch.stack(dirs).float() if dirs else torch.zeros(0, e.latent_dim)

    S = args.n_seeds
    var_normals = torch.stack([torch.randn(len(std), generator=torch.Generator().manual_seed(s))
                               for s in range(S)])
    eps = torch.stack([torch.randn(1, e.latent_dim, e.T_lat, generator=torch.Generator().manual_seed(s))[0]
                       for s in range(S)])

    # mod_sigmoid's '+ 1e-7' traces as a float64 constant, which MPS cannot hold
    import model.model as M
    orig = M.mod_sigmoid
    M.mod_sigmoid = lambda x: 2 * torch.sigmoid(x) ** 2.3 + torch.tensor(1e-7, dtype=x.dtype)
    try:
        dec = torch.jit.trace(e._vae.decoder.eval(), torch.zeros(1, e.latent_dim, e.T_lat), check_trace=False)
    finally:
        M.mod_sigmoid = orig
    enc = torch.jit.trace(EncoderOnly(e._vae).eval(), torch.zeros(1, 1, e.chunk_size), check_trace=False)
    kit = KitRenderer(enc, dec, e.bank['latents'].float(), e.bank['sigmas'].float(), peak, end, z_sil.float(),
                      directions, std.float(), vt.float(), var_normals.float(), eps.float(), int(e.sr))
    return e, torch.jit.script(kit.eval()), keys, names


def check_parity(e, kit, keys, out_dir):
    """Render a few recipes through the app's chain and the kit; print the difference."""
    import numpy as np
    from drum_engine import frame_rms as fr
    T = e.T_lat
    cases = [(0, 8, 'blend', 0.0, {}, 0, 0, 0), (0, 8, 'blend', 0.5, {}, 0, 0, 0),
             (8, 20, 'slerp', 0.4, {}, 0, 0, 0), (16, 30, 'aligned', 0.5, {}, 0, 0, 0),
             (16, 30, 'tail', 0.6, {}, 0, 0, 0), (16, 30, 'crossover', 0.5, {}, 0, 0, 0),
             (24, 2, 'blend', 0.0, {0: 1.5}, 0.0, 0, 0), (24, 2, 'blend', 0.0, {}, 0.6, 0, 0),
             (24, 2, 'blend', 0.0, {}, 0, 0.0, 1.0), (40, 2, 'blend', 0.0, {}, 0, 2.0, 0)]
    print('parity, app chain vs kit (log-spectral L1; another bank hit is ~2.8):')
    for a, b, mode, t, ctl, var, punch, decay in cases:
        if any(k >= len(keys) for k in ctl):
            continue
        e.set_sample(True, 3)
        z = e.morph(a, b, t, mode, 4.0)
        vals = {keys[k]: v for k, v in ctl.items()}
        offset = e.variation(var, 5) if var > 0 else None
        z = e.steer(z, vals, offset)
        z = e.punch(z, punch)
        z = e.stretch(z, 2.0 ** decay)
        ya = torch.from_numpy(e.decode(z)).float()
        cv = torch.zeros(1, kit.directions.shape[0])
        for k, v in ctl.items():
            cv[0, k] = v
        yb = kit.render(a, b, MODES.index(mode), 4.0, torch.tensor([t]), cv, torch.tensor([float(var)]),
                        torch.tensor([5]), torch.tensor([float(decay)]), torch.tensor([float(punch)]), 3, -300.0)[0]
        S = lambda y: torch.log(torch.stft(y, 2048, 512, window=torch.hann_window(2048), return_complex=True).abs() + 1e-4)
        print(f'  {mode:9s} t={t:.1f} ctl={ctl} var={var} punch={punch} decay={decay}: '
              f'max|dy| {(ya - yb).abs().max():.2e}, spec {(S(ya) - S(yb)).abs().mean():.3f}')


def check_encode(e, kit):
    """kit.encode on dataset-prepared bank files == the bank's mu / sigma."""
    from dataset.datasets import OneShotAudioDataset
    worst = 0.0
    for i in range(0, len(e.bank['files']), max(1, len(e.bank['files']) // 6)):
        f = e.bank['files'][i]
        if not os.path.isfile(f):
            continue
        ds = OneShotAudioDataset(os.path.dirname(f), sr=e.sr, chunk_size=e.chunk_size, **e._preprocess)
        ds.files = [f]
        x = ds[0][2].reshape(-1).float()
        with torch.no_grad():
            mu, sigma, timing = kit.encode(x)
        worst = max(worst, (mu - e.bank['latents'][i]).abs().max().item(),
                    (sigma - e.bank['sigmas'][i]).abs().max().item())
    print(f'encode vs bank (mu, sigma): max |d| {worst:.2e}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-r', '--resume', required=True)
    p.add_argument('-b', '--bank', required=True, help='bank with sigmas (build_latent_bank.py)')
    p.add_argument('--directions', default=None, help='directions.pt from control_sweeps.py')
    p.add_argument('--knobs', nargs='+', default=None, help='direction keys (default: the app\'s pick)')
    p.add_argument('--explore_pcs', type=int, default=4, help='bank PCs added as unnamed controls')
    p.add_argument('--n_seeds', type=int, default=64)
    p.add_argument('-o', '--out_dir', required=True)
    p.add_argument('--no_check', action='store_true')
    args = p.parse_args()

    e, kit, keys, names = build(args)
    os.makedirs(args.out_dir, exist_ok=True)
    kit.save(os.path.join(args.out_dir, 'kit.pt'))
    meta = {
        'name': os.path.basename(os.path.normpath(args.out_dir)),
        'checkpoint': os.path.abspath(args.resume), 'bank': os.path.abspath(args.bank),
        'sr': int(e.sr), 'samples': int(e.chunk_size), 'latent_dim': int(e.latent_dim), 't_lat': int(e.T_lat),
        'modes': MODES, 'n_seeds': args.n_seeds,
        'preprocess': {k: (bool(v) if k == 'peak_norm' else v) for k, v in e._preprocess.items()},
        'controls': [{'name': n, 'key': k} for n, k in zip(names, keys)],
        'entries': [{'name': os.path.splitext(os.path.basename(f))[0], 'type': t, 'file': f}
                    for f, t in zip(e.bank['files'], e.bank['types'])],
    }
    json.dump(meta, open(os.path.join(args.out_dir, 'kit.json'), 'w'), indent=1)
    print(f'Saved {args.out_dir}/kit.pt + kit.json: {len(meta["entries"])} hits, '
          f'controls {[c["name"] for c in meta["controls"]]}')
    if not args.no_check:
        loaded = torch.jit.load(os.path.join(args.out_dir, 'kit.pt'))
        check_parity(e, loaded, keys, args.out_dir)
        check_encode(e, loaded)


if __name__ == '__main__':
    main()
