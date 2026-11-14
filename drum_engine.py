"""
Latent drum synthesis engine.

The core of the eventual latent drum machine: it holds a bank of drum latents
and turns latent-space operations into audio one-shots. Unlike the granular
cello synth (which granulates one sustained chunk), a drum hit is a single
transient event, so the interesting controls live in latent space:

    decode      z (D, T_lat)            -> waveform
    lerp/slerp  (z_a, z_b, t)           -> morph between two hits
    perturb     (z, sigma)              -> a variation of a hit
    morph       (a, b, t, mode)         -> also shape-aware: re-timed, tail-only, crossover
    stretch     (z, factor)             -> same hit, tail resampled in time (decay)
    punch       (z, frames)             -> same hit, its peak held longer
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


def frame_rms(y, T):
    """(N, L) audio -> (N, T) RMS per latent frame."""
    N, L = y.shape
    hop = L // T
    return y[:, :hop * T].reshape(N, T, hop).pow(2).mean(2).sqrt()


# ---- latent surgery: the latent is a sequence of frames in time, so a hit can
# be re-timed by reading it at warped positions (src) instead of t.

def warp(z, src, pad=None, fade=8):
    """Read z (N, D, T) at fractional frame positions src (N, T), linearly
    interpolating between frames. Positions before 0 read frame 0 (pre-onset).
    Positions past the end repeat the last frame, or, with `pad` (D,) a silence
    latent, fade into it over `fade` source frames, as if the hit died away just
    past the window (a hit still ringing there would otherwise freeze into a drone)."""
    N, D, T = z.shape
    s = src.to(z.device).clamp(0, T - 1)
    lo = s.floor().long()
    hi = (lo + 1).clamp(max=T - 1)
    w = (s - lo)[:, None, :]
    g = lambda ix: torch.gather(z, 2, ix[:, None, :].expand(N, D, T))
    out = g(lo) * (1 - w) + g(hi) * w
    if pad is not None:
        a = ((src.to(z.device) - (T - 1)) / fade).clamp(0, 1)[:, None, :]
        out = out * (1 - a) + pad.to(z)[None, :, None] * a
    return out


def _frames(N, T, device=None):
    return torch.arange(T, dtype=torch.float32, device=device)[None, :].expand(N, T)


def stretch_tail(z, pivot, r, pad=None, fade=8):
    """Resample the frames after `pivot` (N,) by factor r (> 1 = longer tail):
    output frame t > pivot reads pivot + (t - pivot) / r; earlier frames are unchanged."""
    t, c = _frames(z.shape[0], z.shape[2]), pivot[:, None].float().cpu()
    return warp(z, torch.where(t > c, c + (t - c) / r, t), pad, fade)


def hold_peak(z, peak, k, pad=None, fade=8):
    """Punch: play the frame after the peak over 1 + k frames (k > 0 holds the loudest
    part longer, k < 0 shortens it); everything after shifts by k frames."""
    t, p = _frames(z.shape[0], z.shape[2]), peak[:, None].float().cpu()
    k = max(float(k), -0.9)
    src = torch.where(t <= p, t, torch.where(t <= p + 1 + k, p + (t - p) / (1 + k), t - k))
    return warp(z, src, pad, fade)


def hit_timing(y, T):
    """Peak frame and last sounding frame per hit, (N,) each, from decoded audio y (N, L)."""
    peak = frame_rms(y, T).argmax(1)
    act = frame_activity(y, T)
    end = (act * torch.arange(T)).argmax(1)               # last active frame
    return peak, torch.maximum(end, peak + 1)


def align_src(peak, end, p, L, T):
    """Source positions that move a hit's peak to frame p and resample its tail
    (peak -> end) to L frames: heads are shifted, tails stretched. All (N,)."""
    t = _frames(len(peak), T)
    peak, p = peak[:, None].float(), p[:, None].float()
    scale = (end - peak.squeeze(1)).float()[:, None] / L[:, None]
    return torch.where(t <= p, t - (p - peak), peak + (t - p) * scale)


def ramp(t, centre, width):
    """0 -> 1 linearly over `width` frames around `centre` (broadcast)."""
    return ((t - centre) / max(width, 1e-3) + 0.5).clamp(0, 1)


MORPH_MODES = ['blend', 'slerp', 'aligned', 'tail', 'crossover']


def morph_aligned(za, zb, ta, tb, alpha, mode, width=4.0, pad=None):
    """Shape-aware morphs of za, zb (D, T) with timings ta, tb = (peak, end) frames.
      aligned    both hits re-timed to the in-between peak frame and tail length
                 (tail lengths interpolate geometrically), then blended by alpha
      tail       A's head kept; the tail re-timed to the in-between length and
                 blended by alpha from just after A's peak (a ramp of `width` frames)
      crossover  A turns into B at a time set by alpha: 0 = all A, 1 = all B, between
                 = the hand-over (a `width`-frame ramp) moves linearly from the end of
                 the hit (alpha near 0) to its start (near 1); both re-timed as in 'aligned'"""
    D, T = za.shape
    a = float(alpha)
    if a <= 0 or (a >= 1 and mode != 'tail'):           # exact endpoints
        return (za if a <= 0 else zb).clone()
    (pa, ea), (pb, eb) = [tuple(torch.as_tensor(v).reshape(1) for v in tt) for tt in (ta, tb)]
    La, Lb = (ea - pa).float(), (eb - pb).float()
    L = torch.exp((1 - a) * La.log() + a * Lb.log())
    p = pa.float() if mode == 'tail' else (1 - a) * pa.float() + a * pb.float()
    Za = warp(za[None], align_src(pa, ea, p, L, T), pad)[0]
    Zb = warp(zb[None], align_src(pb, eb, p, L, T), pad)[0]
    t = torch.arange(T, dtype=torch.float32)
    if mode == 'aligned':
        c = torch.full((T,), a)
    elif mode == 'tail':
        c = a * ramp(t, p + 2, width)
    elif mode == 'crossover':
        # hand-over centre moves linearly from just past the end of the (re-timed)
        # hit at 0 to just before frame 0 at 1
        end = float(p + L)
        tau = (1 - a) * (end + width / 2) - a * width / 2
        c = ramp(t, tau, width) if 0 < a < 1 else torch.full((T,), a)
    else:
        raise ValueError(mode)
    return Za * (1 - c) + Zb * c


def silence_latent(z, y):
    """Mean latent frame where decoded hits are silent: z (N, D, T), y (N, L) -> (D,)."""
    quiet = frame_activity(y, z.shape[-1]) == 0                         # (N, T)
    return z.permute(0, 2, 1)[quiet].mean(0)


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

    # ---- time surgery (see warp / morph_aligned above) ---------------------
    def _bank_timing(self):
        """Once per bank: each hit's (peak, end) frames and the silence latent."""
        if getattr(self, '_timing', None) is None:
            Z = self.bank['latents']
            y = torch.from_numpy(self.decode(Z)).float()
            self._timing = hit_timing(y, Z.shape[-1])
            self.z_silence = silence_latent(Z, y)
        return self._timing

    def _peak(self, z):
        y = torch.from_numpy(self.decode(z)).float().unsqueeze(0)
        return frame_rms(y, z.shape[-1]).argmax(1)

    def morph(self, a_idx, b_idx, t, mode='blend', width=4.0):
        """Bank hits a -> b by amount t. blend / slerp: frame by frame as stored;
        aligned / tail / crossover: re-timed first (morph_aligned)."""
        za, zb = self.z(a_idx), self.z(b_idx)
        if mode in ('blend', 'slerp'):
            return (self.slerp if mode == 'slerp' else self.lerp)(za, zb, torch.tensor(float(t)))
        peak, end = self._bank_timing()
        return morph_aligned(za, zb, (peak[a_idx], end[a_idx]), (peak[b_idx], end[b_idx]),
                             t, mode, width, pad=self.z_silence)

    def stretch(self, z, factor, head=2):
        """Decay control with no learned direction: keep z (D, T_lat) up to `head`
        frames past the hit's loudest frame, and resample the frames after it by
        `factor` (2 = tail twice as long); a compressed tail fades into the bank's
        silence latent past the window end. Costs one extra decode to find the peak."""
        if factor == 1:
            return z
        self._bank_timing()
        return stretch_tail(z[None], self._peak(z) + head, factor, pad=self.z_silence)[0]

    def punch(self, z, frames):
        """Hold the loudest part of the hit `frames` latent frames longer (11.6 ms each
        at 44.1 kHz; < 0 shortens it); the rest of the hit follows later."""
        if frames == 0:
            return z
        self._bank_timing()
        return hold_peak(z[None], self._peak(z), frames, pad=self.z_silence)[0]

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
