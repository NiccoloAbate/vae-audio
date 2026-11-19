"""
Discriminators beyond the MelGAN-style waveform MultiScaleDiscriminator (model.py).

    MultiPeriodDiscriminator   HiFi-GAN: the waveform folded at prime periods, so each
                               sub-discriminator sees every p-th sample (periodic structure)
    ComplexSTFTDiscriminator   EnCodec: real and imaginary STFT parts at several
                               resolutions, so it sees phase, unlike a log-magnitude one
                               (and unlike the multi-resolution STFT loss)
    HybridDiscriminator        any of the above plus the waveform MultiScaleDiscriminator

All return a list of (feature_maps, logits), one entry per sub-discriminator, the
interface of MultiScaleDiscriminator, so the hinge and feature-matching losses
(model/loss.py) average over every sub-discriminator alike.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from model.model import MultiScaleDiscriminator


class PeriodDiscriminator(nn.Module):
    """One period of HiFi-GAN's MPD. capacity c gives channels c*(1, 4, 16, 32, 32)
    (HiFi-GAN: c = 32)."""
    def __init__(self, period, capacity=16):
        super().__init__()
        self.period = period
        ch = [1] + [capacity * m for m in (1, 4, 16, 32, 32)]
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(ch[i], ch[i + 1], (5, 1), (3, 1) if i < 4 else 1, padding=(2, 0)))
            for i in range(5)])
        self.post = weight_norm(nn.Conv2d(ch[-1], 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):                                  # (B, 1, T)
        B, C, T = x.shape
        if T % self.period:
            x = F.pad(x, (0, self.period - T % self.period), 'reflect')
        x = x.view(B, C, -1, self.period)                  # (B, 1, T/p, p)
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)
        logits = self.post(x)
        feats.append(logits)
        return feats, logits


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11), capacity=16):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p, capacity) for p in periods])

    def forward(self, x):
        return [d(x) for d in self.discriminators]


class STFTDiscriminator(nn.Module):
    """One resolution of EnCodec's MS-STFT discriminator: (real, imag) of a normalised
    STFT as 2 channels over (time, freq); convs stride 2 in frequency, dilate in time."""
    def __init__(self, n_fft, filters=32, dilations=(1, 2, 4), kernel=(3, 9)):
        super().__init__()
        self.n_fft, self.hop = n_fft, n_fft // 4
        self.register_buffer('window', torch.hann_window(n_fft), persistent=False)
        pad = lambda d: ((kernel[0] - 1) * d // 2, (kernel[1] - 1) // 2)
        convs = [weight_norm(nn.Conv2d(2, filters, kernel, padding=pad(1)))]
        for d in dilations:
            convs.append(weight_norm(nn.Conv2d(filters, filters, kernel, (1, 2), dilation=(d, 1),
                                               padding=pad(d))))
        convs.append(weight_norm(nn.Conv2d(filters, filters, (3, 3), padding=(1, 1))))
        self.convs = nn.ModuleList(convs)
        self.post = weight_norm(nn.Conv2d(filters, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):                                  # (B, 1, T)
        S = torch.stft(x.squeeze(1), self.n_fft, self.hop, self.n_fft, self.window,
                       normalized=True, return_complex=True)          # (B, F, frames)
        z = torch.stack([S.real, S.imag], 1).transpose(2, 3)          # (B, 2, frames, F)
        feats = []
        for conv in self.convs:
            z = F.leaky_relu(conv(z), 0.2)
            feats.append(z)
        logits = self.post(z)
        feats.append(logits)
        return feats, logits


class ComplexSTFTDiscriminator(nn.Module):
    def __init__(self, n_ffts=(2048, 1024, 512, 256, 128), filters=32):
        super().__init__()
        self.discriminators = nn.ModuleList([STFTDiscriminator(n, filters) for n in n_ffts])

    def forward(self, x):
        return [d(x) for d in self.discriminators]


class HybridDiscriminator(nn.Module):
    """parts: any of 'msd' (waveform, MultiScaleDiscriminator), 'mpd', 'stft'.
    groups lists (name, number of sub-discriminators) in output order, so the
    trainer can log each family's hinge loss."""
    def __init__(self, parts=('msd', 'mpd', 'stft'), msd_capacity=16, mpd_capacity=16,
                 mpd_periods=(2, 3, 5, 7, 11), stft_ffts=(2048, 1024, 512, 256, 128),
                 stft_filters=32):
        super().__init__()
        make = {'msd': lambda: MultiScaleDiscriminator(capacity=msd_capacity),
                'mpd': lambda: MultiPeriodDiscriminator(mpd_periods, mpd_capacity),
                'stft': lambda: ComplexSTFTDiscriminator(stft_ffts, stft_filters)}
        self.parts = nn.ModuleDict({p: make[p]() for p in parts})
        self.groups = [(p, len(self.parts[p].discriminators)) for p in parts]

    def forward(self, x):
        return [out for p in self.parts.values() for out in p(x)]
