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

    def steer(self, z, values):
        """z (D, T_lat) + sum of s * knob vector, for values {key: s}. 'active'
        knobs only offset the frames where the hit sounds, as they were fit."""
        values = {k: float(s) for k, s in values.items() if s}
        if not values:
            return z
        mask = torch.ones(z.shape[-1])
        if any(k.startswith('active/') for k in values):
            y = torch.from_numpy(self.decode(z)).float().unsqueeze(0)
            mask = frame_activity(y, z.shape[-1])[0]
        out = z.clone()
        for k, s in values.items():
            m = mask if k.startswith('active/') else torch.ones_like(mask)
            out = out + s * self.knobs[k]['vector'][:, None] * m[None, :]
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
        w = self.decode(z)
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
