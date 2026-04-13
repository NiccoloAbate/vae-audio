import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal.windows import kaiser
from base import BaseModel, BaseVAE, BaseGMVAE


# ---------------------------------------------------------------------------
# Raw Audio VAE
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Dilated residual conv block: x -> LeakyReLU -> dilated conv -> LeakyReLU -> 1x1 conv -> + x"""
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 3, dilation=dilation, padding=dilation),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class ResidualStack(nn.Module):
    """Stack of residual blocks with exponentially growing dilations [1, 3, 9]."""
    def __init__(self, channels, n_layers=3):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock(channels, 3 ** i) for i in range(n_layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class EncoderBlock(nn.Module):
    """Strided Conv1d downsampling + residual stack."""
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=2 * stride, stride=stride, padding=stride // 2)
        self.res = ResidualStack(out_ch)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.res(self.conv(self.act(x)))


class DecoderBlock(nn.Module):
    """Strided ConvTranspose1d upsampling + residual stack."""
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2 * stride, stride=stride, padding=stride // 2)
        self.res = ResidualStack(out_ch)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.res(self.conv(self.act(x)))


def _pqmf_prototype_filter(taps=62, cutoff_ratio=0.142, beta=9.0):
    """Kaiser-windowed sinc prototype lowpass filter for PQMF.

    Based on: 'A Kaiser window approach for the design of prototype filters
    of cosine modulated filterbanks' (IEEE 1998).
    """
    assert taps % 2 == 0
    assert 0.0 < cutoff_ratio < 1.0
    omega_c = np.pi * cutoff_ratio
    idx = np.arange(taps + 1)
    with np.errstate(invalid="ignore"):
        h = np.sin(omega_c * (idx - 0.5 * taps)) / (np.pi * (idx - 0.5 * taps))
    h[taps // 2] = np.cos(0) * cutoff_ratio   # fix NaN at centre tap
    h *= kaiser(taps + 1, beta)
    return h


class PQMF(nn.Module):
    """Pseudo-Quadrature Mirror Filterbank (analysis + synthesis).

    Based on ParallelWaveGAN / multiband-HiFiGAN implementation
    (kan-bayashi, MIT License).

    Analysis:  (B, 1, T) -> (B, n_bands, T // n_bands)
    Synthesis: (B, n_bands, T // n_bands) -> (B, 1, T)

    Args:
        n_bands:      number of subbands (default 4; use 16 for RAVE-style)
        taps:         prototype filter length in taps (default 62)
        cutoff_ratio: prototype lowpass cutoff as fraction of Nyquist.
                      Rule of thumb: slightly above 1 / (2 * n_bands).
                      Default 0.142 is tuned for n_bands=4; use ~0.04 for n_bands=16.
        beta:         kaiser window beta (default 9.0)
    """
    def __init__(self, n_bands=4, taps=62, cutoff_ratio=0.142, beta=9.0):
        super().__init__()
        self.n_bands = n_bands

        h_proto = _pqmf_prototype_filter(taps, cutoff_ratio, beta)

        h_analysis  = np.zeros((n_bands, len(h_proto)))
        h_synthesis = np.zeros((n_bands, len(h_proto)))
        for k in range(n_bands):
            cf = (2 * k + 1) * (np.pi / (2 * n_bands)) * (np.arange(taps + 1) - taps / 2)
            phase = (-1) ** k * np.pi / 4
            h_analysis[k]  = 2 * h_proto * np.cos(cf + phase)
            h_synthesis[k] = 2 * h_proto * np.cos(cf - phase)

        # analysis_filter:  (n_bands, 1, taps+1)   — used as grouped conv weight
        # synthesis_filter: (1, n_bands, taps+1)   — single-output conv weight
        self.register_buffer("analysis_filter",
                             torch.from_numpy(h_analysis).float().unsqueeze(1))
        self.register_buffer("synthesis_filter",
                             torch.from_numpy(h_synthesis).float().unsqueeze(0))

        # Identity-stride filter for polyphase up/downsampling
        updown = torch.zeros(n_bands, n_bands, n_bands).float()
        for k in range(n_bands):
            updown[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown)

        self.pad_fn = nn.ConstantPad1d(taps // 2, 0.0)

    def analysis(self, x):
        """(B, 1, T) -> (B, n_bands, T // n_bands)"""
        x = F.conv1d(self.pad_fn(x), self.analysis_filter)
        return F.conv1d(x, self.updown_filter, stride=self.n_bands)

    def synthesis(self, x):
        """(B, n_bands, T // n_bands) -> (B, 1, T)"""
        x = F.conv_transpose1d(x, self.updown_filter * self.n_bands, stride=self.n_bands)
        return F.conv1d(self.pad_fn(x), self.synthesis_filter)

    def forward(self, x):
        return self.analysis(x)


class RawAudioVAE(nn.Module):
    """
    Raw-waveform VAE with a temporal latent space.

    Without PQMF (n_bands=None):
        Encoder: (B, 1, T) -> (B, latent_dim, T//product(strides))
        Decoder: reverse
        Default strides=[4,4,4,4], chunk=16384 → T_lat=64

    With PQMF (n_bands=16):
        PQMF first decomposes (B,1,T) -> (B,16,T//16), then encoder/decoder
        operate on subbands, and inverse PQMF reconstructs audio.
        Use strides=[4,4,4], channels=[64,128,256,512] → T_lat=16
        Total compression: 16 × 64 = 1024x
    """
    def __init__(self, latent_dim=64, channels=(32, 64, 128, 256, 512),
                 strides=(4, 4, 4, 4), n_bands=None):
        super().__init__()
        channels = list(channels)
        strides  = list(strides)
        assert len(channels) == len(strides) + 1

        self.latent_dim = latent_dim
        self.n_bands    = n_bands

        # Optional PQMF filterbank
        # (taps, cutoff_ratio) pairs optimised per band count by SNR sweep:
        #   N=4:  taps=62,  cutoff=0.142 → ~40 dB SNR
        #   N=16: taps=128, cutoff=0.039 → ~32 dB SNR (was 17 dB with taps=62)
        if n_bands is not None:
            pqmf_params = {4: (62, 0.142), 8: (62, 0.072), 16: (128, 0.039)}
            taps, cutoff_ratio = pqmf_params.get(n_bands, (128, 0.039))
            self.pqmf    = PQMF(n_bands=n_bands, taps=taps, cutoff_ratio=cutoff_ratio)
            in_ch        = n_bands
            out_ch       = n_bands
        else:
            self.pqmf    = None
            in_ch        = 1
            out_ch       = 1

        # Encoder
        enc = [nn.Conv1d(in_ch, channels[0], 7, padding=3)]
        for i, s in enumerate(strides):
            enc.append(EncoderBlock(channels[i], channels[i + 1], s))
        self.encoder     = nn.Sequential(*enc)
        self.mu_proj     = nn.Conv1d(channels[-1], latent_dim, 1)
        self.logvar_proj = nn.Conv1d(channels[-1], latent_dim, 1)

        # Decoder
        dec_ch = list(reversed(channels))
        dec    = [nn.Conv1d(latent_dim, dec_ch[0], 1)]
        for i, s in enumerate(reversed(strides)):
            dec.append(DecoderBlock(dec_ch[i], dec_ch[i + 1], s))
        dec += [nn.LeakyReLU(0.2), nn.Conv1d(dec_ch[-1], out_ch, 7, padding=3)]
        if n_bands is None:
            dec.append(nn.Tanh())   # raw waveform output clipped to [-1, 1]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        if self.pqmf is not None:
            x = self.pqmf.analysis(x)               # (B, N, T//N)
        h      = self.encoder(x)                     # (B, C, T_lat)
        mu     = self.mu_proj(h)                     # (B, D, T_lat)
        logvar = self.logvar_proj(h)
        z      = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return mu, logvar, z

    def decode(self, z):
        x = self.decoder(z)                          # (B, N or 1, T//N or T)
        if self.pqmf is not None:
            x = self.pqmf.synthesis(x)               # (B, 1, T)
        return x

    def forward(self, x):
        mu, logvar, z = self.encode(x)
        y_hat = self.decode(z)
        return y_hat, mu, logvar, z


# ---------------------------------------------------------------------------
# Specral Audio VAE
# ---------------------------------------------------------------------------

def spec_conv1d(n_layer=3, n_channel=[64, 32, 16, 8], filter_size=[1, 3, 3], stride=[1, 2, 2]):
    """
    Construction of conv. layers. Note the current implementation always effectively turn to 1-D conv,
    inspired by https://arxiv.org/pdf/1704.04222.pdf.
    :param n_layer: number of conv. layers
    :param n_channel: in/output number of channels for each layer ( len(n_channel) = n_layer + 1 ).
            The first channel is the number of freqeuncy bands of input spectrograms
    :param filter_size: the filter size (x-axis) for each layer ( len(filter_size) = n_layer )
    :param stride: filter stride size (x-axis) for each layer ( len(stride) = n_layer )
    :return: an object (nn.Sequential) constructed of specified conv. layers
    TODO:
        [x] directly use nn.Conv1d for implementation
        [] allow different activations and batch normalization functions
    """

    assert len(n_channel) == n_layer + 1, "This must fulfill: len(n_channel) = n_layer + 1"
    ast_msg = "The following must fulfill: len(filter_size) == len(stride) == n_layer"
    assert len(filter_size) == len(stride) == n_layer, ast_msg

    # construct layers
    conv_layers = []
    for i in range(n_layer):
        in_channel, out_channel = n_channel[i:i + 2]
        conv_layers += [
            nn.Conv1d(in_channel, out_channel, filter_size[i], stride[i]),
            nn.BatchNorm1d(out_channel),
            nn.Tanh()
        ]

    return nn.Sequential(*conv_layers)


def spec_deconv1d(n_layer=3, n_channel=[64, 32, 16, 8], filter_size=[1, 3, 3], stride=[1, 2, 2]):
    """
    Construction of deconv. layers. Input the arguments in normal conv. order.
    E.g., n_channel = [1, 32, 16, 8] gives deconv. layers of [8, 16, 32, 1].
    :param n_layer: number of deconv. layers
    :param n_channel: in/output number of channels for each layer ( len(n_channel) = n_layer + 1 )
            The first channel is the number of freqeuncy bands of input spectrograms
    :param filter_size: the filter size (x-axis) for each layer ( len(filter_size) = n_layer )
    :param stride: filter stride size (x-axis) for each layer ( len(stride) = n_layer )
    :return: an object (nn.Sequential) constructed of specified deconv. layers.
    TODO:
        [x] directly use nn.Conv1d for implementation
        [] allow different activations and batch normalization functions
    """

    assert len(n_channel) == n_layer + 1, "This must fulfill: len(n_channel) = n_layer + 1"
    ast_msg = "The following must fulfill: len(filter_size) == len(stride) == n_layer"
    assert len(filter_size) == len(stride) == n_layer, ast_msg

    n_channel, filter_size, stride = n_channel[::-1], filter_size[::-1], stride[::-1]

    deconv_layers = []
    for i in range(n_layer - 1):
        in_channel, out_channel = n_channel[i:i + 2]
        deconv_layers += [
            nn.ConvTranspose1d(in_channel, out_channel, filter_size[i], stride[i]),
            nn.BatchNorm1d(out_channel),
            nn.Tanh()
        ]

    # Construct the output layer
    deconv_layers += [
        nn.ConvTranspose1d(n_channel[-2], n_channel[-1], filter_size[-1], stride[-1]),
        nn.Tanh()  # check the effect of with or without BatchNorm in this layer
    ]

    return nn.Sequential(*deconv_layers)


def fc(n_layer, n_channel, activation='tanh', batchNorm=True):
    """
    Construction of fc. layers.
    :param n_layer: number of fc. layers
    :param n_channel: in/output number of neurons for each layer ( len(n_channel) = n_layer + 1 )
    :param activation: allow either 'tanh' or None for now
    :param batchNorm: True|False, indicate apply batch normalization or not
    TODO:
        [] allow different activations and batch normalization functions
    """

    assert len(n_channel) == n_layer + 1, "This must fulfill: len(n_channel) = n_layer + 1"
    assert activation in [None, 'tanh'], "Only implement 'tanh' for now"

    fc_layers = []
    for i in range(n_layer):
        layer = [nn.Linear(n_channel[i], n_channel[i + 1])]
        if batchNorm:
            layer.append(nn.BatchNorm1d(n_channel[i + 1]))
        if activation:
            layer.append(nn.Tanh())
        fc_layers += layer

    return nn.Sequential(*fc_layers)


class SpecVAE(BaseVAE):
    def __init__(self, input_size=(64, 15), latent_dim=32, is_featExtract=False,
                 n_convLayer=3, n_convChannel=[32, 16, 8], filter_size=[1, 3, 3], stride=[1, 2, 2],
                 n_fcLayer=1, n_fcChannel=[256]):
        """
        Construction of VAE
        :param input_size: (n_channel, n_freqBand, n_contextWin);
                           assume a spectrogram input of size (n_freqBand, n_contextWin)
        :param latent_dim: the dimension of the latent vector
        :param is_featExtract: if True, output z as mu; otherwise, output z derived from reparameterization trick
        """
        super(SpecVAE, self).__init__(input_size, latent_dim, is_featExtract)
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.is_featExtract = is_featExtract

        self.n_freqBand, self.n_contextWin = input_size

        # Construct encoder and Gaussian layers
        self.encoder = spec_conv1d(n_convLayer, [self.n_freqBand] + n_convChannel, filter_size, stride)
        self.flat_size, self.encoder_outputSize = self._infer_flat_size()
        self.encoder_fc = fc(n_fcLayer, [self.flat_size, *n_fcChannel], activation='tanh', batchNorm=True)
        self.mu_fc = fc(1, [n_fcChannel[-1], latent_dim], activation=None, batchNorm=False)
        self.logvar_fc = fc(1, [n_fcChannel[-1], latent_dim], activation=None, batchNorm=False)

        # Construct decoder
        self.decoder_fc = fc(n_fcLayer + 1, [self.latent_dim, *n_fcChannel[::-1], self.flat_size],
                             activation='tanh', batchNorm=True)
        self.decoder = spec_deconv1d(n_convLayer, [self.n_freqBand] + n_convChannel, filter_size, stride)

    def _infer_flat_size(self):
        encoder_output = self.encoder(torch.ones(1, *self.input_size))
        return int(np.prod(encoder_output.size()[1:])), encoder_output.size()[1:]

    def encode(self, x):
        if len(x.shape) == 4:
            assert x.shape[1] == 1
            x = x.squeeze(1)

        h = self.encoder(x)
        h2 = self.encoder_fc(h.view(-1, self.flat_size))
        mu = self.mu_fc(h2)
        logvar = self.logvar_fc(h2)
        mu, logvar, z = self._infer_latent(mu, logvar)

        return mu, logvar, z

    def decode(self, z):
        h = self.decoder_fc(z)
        x_recon = self.decoder(h.view(-1, *self.encoder_outputSize))
        return x_recon

    def forward(self, x):
        mu, logvar, z = self.encode(x)
        x_recon = self.decode(z)
        # print(x_recon.size(), mu.size(), var.size(), z.size())
        return x_recon, mu, logvar, z


# ---------------------------------------------------------------------------
# Multi-Scale Waveform Discriminator (MelGAN-style)
# ---------------------------------------------------------------------------

class WaveformDiscriminatorBlock(nn.Module):
    """Single-scale waveform discriminator.

    Returns (feature_maps, logits) where feature_maps is a list of
    intermediate activations used for feature-matching loss.

    Args:
        capacity: channel width multiplier. Default 8 gives ~1.5M params/scale
                  (3 scales ≈ 4.5M total). Use 16 to restore the original
                  ~5.6M/scale (16.9M total) if needed.
    """
    def __init__(self, capacity=8):
        super().__init__()
        c = capacity
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(1,    c,    15, 1,  padding=7)),
            nn.utils.weight_norm(nn.Conv1d(c,    4*c,  41, 4,  groups=max(1, c//4),   padding=20)),
            nn.utils.weight_norm(nn.Conv1d(4*c,  16*c, 41, 4,  groups=c,              padding=20)),
            nn.utils.weight_norm(nn.Conv1d(16*c, 64*c, 41, 4,  groups=4*c,            padding=20)),
            nn.utils.weight_norm(nn.Conv1d(64*c, 64*c, 41, 4,  groups=16*c,           padding=20)),
            nn.utils.weight_norm(nn.Conv1d(64*c, 64*c, 5,  1,  padding=2)),
        ])
        self.output_conv = nn.utils.weight_norm(nn.Conv1d(64*c, 1, 3, 1, padding=1))

    def forward(self, x):
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
            features.append(x)
        logits = self.output_conv(x)
        features.append(logits)
        return features, logits


class MultiScaleDiscriminator(nn.Module):
    """Three-scale waveform discriminator.

    Runs three WaveformDiscriminatorBlocks on the waveform at
    progressively lower resolutions (original, 2x, 4x downsampled).

    forward() returns a list of (feature_maps, logits) — one per scale.

    Args:
        capacity: passed to each WaveformDiscriminatorBlock.
    """
    def __init__(self, capacity=8):
        super().__init__()
        self.discriminators = nn.ModuleList([
            WaveformDiscriminatorBlock(capacity),
            WaveformDiscriminatorBlock(capacity),
            WaveformDiscriminatorBlock(capacity),
        ])

    def forward(self, x):
        results = []
        x_i = x
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x_i = F.avg_pool1d(x_i, kernel_size=4, stride=2, padding=2)
            feats, logits = disc(x_i)
            results.append((feats, logits))
        return results


class SpectrogramDiscriminatorBlock(nn.Module):
    """Single-scale spectrogram discriminator.

    Operates on a log-magnitude STFT (B, 1, F, T) using 2-D convolutions.
    Returns (feature_maps, logits) — same interface as WaveformDiscriminatorBlock.
    """
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            # stride=(freq, time) — downsample freq quickly to avoid huge feature maps
            nn.utils.weight_norm(nn.Conv2d(1,   32,  (3, 9), stride=(2, 1), padding=(1, 4))),
            nn.utils.weight_norm(nn.Conv2d(32,  64,  (3, 9), stride=(2, 2), padding=(1, 4))),
            nn.utils.weight_norm(nn.Conv2d(64,  128, (3, 9), stride=(2, 2), padding=(1, 4))),
            nn.utils.weight_norm(nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=(1, 1))),
            nn.utils.weight_norm(nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1))),
        ])
        self.output_conv = nn.utils.weight_norm(
            nn.Conv2d(256, 1, (3, 3), stride=(1, 1), padding=(1, 1))
        )

    def forward(self, spec):
        # spec: (B, 1, F, T_frames)
        features, x = [], spec
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
            features.append(x)
        logits = self.output_conv(x)   # (B, 1, F', T')
        features.append(logits)
        return features, logits


class MultiScaleSpecDiscriminator(nn.Module):
    """Multi-scale spectrogram discriminator (RAVE-style).

    Computes log-magnitude STFTs at multiple resolutions and runs a
    SpectrogramDiscriminatorBlock on each. Because it operates on magnitude
    spectrograms rather than raw waveforms, it is phase-agnostic and compatible
    with STFT-based reconstruction losses.

    forward() returns a list of (feature_maps, logits) — one per scale,
    matching the MultiScaleDiscriminator interface exactly.
    """
    def __init__(self,
                 fft_sizes=(2048, 1024, 512),
                 hop_sizes=(512,  256,  128)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.discriminators = nn.ModuleList([
            SpectrogramDiscriminatorBlock() for _ in fft_sizes
        ])

    def _log_mag_spec(self, x, n_fft, hop):
        """Compute log-magnitude STFT from waveform (B, 1, T) → (B, 1, F, T_frames)."""
        x_sq = x.squeeze(1)                                         # (B, T)
        win  = torch.hann_window(n_fft, device=x.device)
        S    = torch.stft(x_sq, n_fft, hop, n_fft, win, return_complex=True)
        mag  = torch.log(S.abs() + 1e-5)                            # (B, F, T_frames)
        return mag.unsqueeze(1)                                      # (B, 1, F, T_frames)

    def forward(self, x):
        results = []
        for disc, n_fft, hop in zip(self.discriminators, self.fft_sizes, self.hop_sizes):
            spec = self._log_mag_spec(x, n_fft, hop)
            feats, logits = disc(spec)
            results.append((feats, logits))
        return results


class Conv1dGMVAE(BaseGMVAE):
    def __init__(self, input_size=(128, 20), latent_dim=16, n_component=12,
                 pow_exp=0, logvar_trainable=False, is_featExtract=False):
        super(Conv1dGMVAE, self).__init__(input_size, latent_dim, n_component, is_featExtract)
        self.n_channel = input_size[0]
        self.pow_exp, self.logvar_trainable = pow_exp, logvar_trainable
        self._build_logvar_lookup(pow_exp=pow_exp, logvar_trainable=logvar_trainable)

        self.encoder = nn.Sequential(
            nn.Conv1d(self.n_channel, 512, 3, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.flat_size, self.encoder_outputSize = self._infer_flat_size()

        self.encoder_fc = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.lin_mu = nn.Linear(512, latent_dim)
        self.lin_logvar = nn.Linear(512, latent_dim)
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.flat_size),
            nn.BatchNorm1d(self.flat_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 512, 3, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(512, self.n_channel, 3, 1),
            nn.Tanh()
        )

    def _infer_flat_size(self):
        print(torch.ones(1, *self.input_size).shape)
        encoder_output = self.encoder(torch.ones(1, *self.input_size))
        return int(np.prod(encoder_output.size()[1:])), encoder_output.size()[1:]

    def encode(self, x):
        h = self.encoder(x)
        h2 = self.encoder_fc(h.view(-1, self.flat_size))
        mu = self.lin_mu(h2)
        logvar = self.lin_logvar(h2)
        mu, logvar, z = self._infer_latent(mu, logvar)
        logLogit_qy_x, qy_x, y = self._infer_class(z)

        return mu, logvar, z, logLogit_qy_x, qy_x, y

    def decode(self, z):
        h = self.decoder_fc(z)
        x_recon = self.decoder(h.view(-1, *self.encoder_outputSize))
        return x_recon

    def forward(self, x):
        mu, logvar, z, logLogit_qy_x, qy_x, y = self.encode(x)
        x_recon = self.decode(z)

        return x_recon, mu, logvar, z, logLogit_qy_x, qy_x, y
