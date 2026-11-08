"""
Latent drum synthesis engine.

The core of the eventual latent drum machine: it holds a bank of drum latents
and turns latent-space operations into audio one-shots. Unlike the granular
cello synth (which granulates one sustained chunk), a drum hit is a single
transient event, so the interesting controls live in latent space:

    decode      z (D, T_lat)            -> waveform
    lerp/slerp  (z_a, z_b, t)           -> morph between two hits
    perturb     (z, sigma)              -> a variation of a hit
    render      z -> trimmed, normalised one-shot ready to trigger

`velocity`-style automation is just interpolation with t driven by a control
value (see generate_drums.py) — mapping velocity/MIDI to a latent direction is
the same operation, which is why the engine exposes the primitives rather than
baking in one behaviour.

Decodes via the checkpoint directly (no export needed) or a traced decoder.pt
(export_decoder.py) if given — the plugin will use the latter.
"""
import numpy as np
import torch

import model.model as module_arch


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def frame_activity(y, T, thresh_db=-50.0, dilate=2):
    """(N, L) decoded audio -> (N, T) float mask of latent frames where it sounds
    (frame RMS within thresh_db of the loudest frame, dilated by `dilate`)."""
    N, L = y.shape
    hop = L // T
    rms = y[:, :hop * T].reshape(N, T, hop).pow(2).mean(2).sqrt()
    m = (rms > rms.max(1, keepdim=True).values * 10 ** (thresh_db / 20)).float()
    if dilate:
        m = torch.nn.functional.max_pool1d(m.unsqueeze(1), 2 * dilate + 1, 1, dilate).squeeze(1)
    return m


class DrumEngine:
    def __init__(self, checkpoint=None, decoder_pt=None, bank=None, device=None):
        assert checkpoint or decoder_pt, 'need a checkpoint or a traced decoder'
        self.device = device or get_device()

        if decoder_pt:                                   # traced TorchScript path
            self._decoder = torch.jit.load(decoder_pt, map_location=self.device).eval()
            self._decode_raw = lambda z: self._decoder(z)
            ck = torch.load(checkpoint, map_location='cpu', weights_only=False) if checkpoint else None
        else:                                            # decode via the VAE
            ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
            cfg = ck['config']['arch']
            vae = getattr(module_arch, cfg['type'])(**cfg['args'])
            vae.load_state_dict(ck['state_dict'], strict=False)
            self._vae = vae.to(self.device).eval()
            self._decode_raw = lambda z: self._vae.decode(z)

        dl = dict(ck['config']['data_loader']['args']) if ck else {}
        self._preprocess = dict(onset_thresh=dl.get('onset_thresh', 0.02),
                                pre_pad=dl.get('pre_pad', 256), peak_norm=dl.get('peak_norm', True))
        self.sr = dl.get('sr', 22050)
        self.chunk_size = dl.get('chunk_size', 110080)

        # Latent bank (optional): dict from build_latent_bank.py, or a path
        self.bank = None
        if bank is not None:
            self.load_bank(bank)

    # ---- bank -------------------------------------------------------------
    def load_bank(self, bank):
        if isinstance(bank, str):
            bank = torch.load(bank, map_location='cpu', weights_only=False)
        self.bank = bank
        self.latent_dim = bank['latents'].shape[1]
        self.T_lat = bank['latents'].shape[2]
        return self

    def z(self, idx):
        """Latent (D, T_lat) for bank slot idx."""
        return self.bank['latents'][idx].clone()

    def original(self, idx, **trim_kw):
        """The bank slot's source file as the model saw it (mono, onset-aligned,
        peak-normalised, cut to the model's window), trimmed like render(): the
        reference to compare a reconstruction against. None if the file is gone."""
        import os
        from dataset.datasets import OneShotAudioDataset
        path = self.bank['files'][idx]
        if not os.path.isfile(path):
            return None
        ds = OneShotAudioDataset(os.path.dirname(path), sr=self.sr, chunk_size=self.chunk_size,
                                 **self._preprocess)
        ds.files = [path]
        return self.trim(ds[0][2].squeeze(0).numpy(), **trim_kw)

    def indices_of(self, drum_type):
        return [i for i, t in enumerate(self.bank['types']) if t == drum_type]

    # ---- latent ops -------------------------------------------------------
    @staticmethod
    def lerp(za, zb, t):
        return (1.0 - t) * za + t * zb

    @staticmethod
    def slerp(za, zb, t):
        """Spherical interpolation treating the whole (D, T_lat) as one vector."""
        a = za / (za.norm() + 1e-8)
        b = zb / (zb.norm() + 1e-8)
        omega = torch.acos(torch.clamp((a * b).sum(), -1.0, 1.0))
        if omega.abs() < 1e-6:
            return DrumEngine.lerp(za, zb, t)
        s = torch.sin(omega)
        return (torch.sin((1 - t) * omega) * za + torch.sin(t * omega) * zb) / s

    @staticmethod
    def perturb(z, sigma, generator=None):
        return z + sigma * torch.randn(z.shape, generator=generator)

    # ---- knobs (latent directions from control_sweeps.py) -----------------
    def load_directions(self, path, keys=None, min_monotone=0.75, min_slope=0.1):
        """Load knob directions. keys: e.g. ['active/brightness/pca']; by default,
        per knob the steepest 'active' direction that steered at least
        min_monotone of the swept hits the right way with slope >= min_slope.
        Returns the chosen keys (self.knobs[key] holds vector + calibration)."""
        d = torch.load(path, map_location='cpu', weights_only=False)
        dirs = d['directions']
        if keys is None:
            keys = []
            for k in d['knobs']:
                ok = [(v['slope'], key) for key, v in dirs.items()
                      if key.startswith(f'active/{k}/') and v['monotone'] >= min_monotone
                      and v['slope'] >= min_slope]
                if ok:
                    keys.append(max(ok)[1])
        self.knobs = {key: dirs[key] for key in keys}
        self.knob_source = d.get('checkpoint')
        self.knob_grid = (min(d['grid']), max(d['grid']))     # the range that was swept
        return keys

    def fit_explore(self, n_pcs=4, n_dims=4):
        """Unlabelled directions to explore by ear, from the bank itself: the
        top principal components of the bank's latents (each averaged over its
        sounding frames) and the n_dims raw latent dims that vary most across
        the bank. Each vector is one bank std long, so s = 1 moves a hit by
        about as much as hits differ along it. Keys: active/pc/<k>, active/dim/<d>."""
        Z = self.bank['latents']
        y = torch.from_numpy(self.decode(Z)).float()
        m = frame_activity(y, Z.shape[-1])                               # (N, T)
        X = (Z * m[:, None]).sum(2) / m.sum(1, keepdim=True).clamp_min(1)   # (N, D)
        Xc = X - X.mean(0)
        _, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
        std = S / max(len(X) - 1, 1) ** 0.5
        var = (S ** 2) / (S ** 2).sum()
        self.explore = {}
        for k in range(min(n_pcs, len(S))):
            self.explore[f'active/pc/{k + 1}'] = {
                'vector': std[k] * Vt[k], 'analogue': f'PC{k + 1}',
                'info': f'{var[k]:.0%} of the bank\'s latent variance'}
        dstd = X.std(0)
        for d in dstd.argsort(descending=True)[:n_dims].tolist():
            v = torch.zeros(X.shape[1]); v[d] = dstd[d]
            self.explore[f'active/dim/{d}'] = {
                'vector': v, 'analogue': f'dim {d}',
                'info': f'raw latent dim {d} (bank std {dstd[d]:.2f})'}
        return list(self.explore)

    def steer(self, z, values):
        """z (D, T_lat) + sum of s * knob vector, for values {key: s}, over the
        knobs and the explore directions. 'active' keys only offset the frames
        where the hit sounds, as they were fit."""
        values = {k: float(s) for k, s in values.items() if s}
        if not values:
            return z
        dirs = {**(getattr(self, 'knobs', None) or {}), **(getattr(self, 'explore', None) or {})}
        mask = torch.ones(z.shape[-1])
        if any(k.startswith('active/') for k in values):
            y = torch.from_numpy(self.decode(z)).float().unsqueeze(0)
            mask = frame_activity(y, z.shape[-1])[0]
        out = z.clone()
        for k, s in values.items():
            m = mask if k.startswith('active/') else torch.ones_like(mask)
            out = out + s * dirs[k]['vector'][:, None] * m[None, :]
        return out

    # ---- decode / render --------------------------------------------------
    def decode(self, z):
        """z: (D, T_lat) or (B, D, T_lat) -> waveform np array (T,) or (B, T)."""
        single = (z.dim() == 2)
        if single:
            z = z.unsqueeze(0)
        with torch.no_grad():
            y = self._decode_raw(z.to(self.device)).squeeze(1).cpu().numpy()  # (B, T)
        return y[0] if single else y

    def render(self, z, trim_db=-45.0, tail_ms=60.0, normalize=True):
        """Decode then trim the trailing silence -> a usable one-shot.

        trim_db: cut once the signal stays this many dB below its peak.
        tail_ms: keep this much release after the last loud sample.
        """
        return self.trim(self.decode(z), trim_db, tail_ms, normalize)

    def trim(self, w, trim_db=-45.0, tail_ms=60.0, normalize=True):
        """Cut trailing silence (see render) and optionally peak-normalise."""
        env = np.abs(w)
        peak = env.max()
        if peak > 0:
            thresh = peak * (10.0 ** (trim_db / 20.0))
            loud = np.nonzero(env > thresh)[0]
            if len(loud):
                end = min(len(w), loud[-1] + int(tail_ms * 1e-3 * self.sr))
                w = w[:end]
            if normalize:
                w = w / peak * 0.95
        return w
