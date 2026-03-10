import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel, BaseVAE, BaseGMVAE


# ---------------------------------------------------------------------------
# Raw Audio VAE
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Dilated residual conv block: x → LeakyReLU → dilated conv → LeakyReLU → 1x1 conv → + x"""
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
        self.blocks = nn.ModuleList(
            [ResidualBlock(channels, 3 ** i) for i in range(n_layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class EncoderBlock(nn.Module):
    """Strided Conv1d downsampling + residual stack."""
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=2 * stride,
                              stride=stride, padding=stride // 2)
        self.res = ResidualStack(out_ch)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.res(self.conv(self.act(x)))


class DecoderBlock(nn.Module):
    """Strided ConvTranspose1d upsampling + residual stack."""
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2 * stride,
                                       stride=stride, padding=stride // 2)
        self.res = ResidualStack(out_ch)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.res(self.conv(self.act(x)))


class RawAudioVAE(nn.Module):
    """
    Raw-waveform VAE with a temporal latent space.

    Encoder downsamples (B, 1, T) → (B, latent_dim, T//total_stride).
    Decoder upsamples (B, latent_dim, T//total_stride) → (B, 1, T).
    Total downsampling = product(strides).

    With default strides=[4,4,4,4] and chunk_size=16384:
      latent frames = 16384 / 256 = 64  (at 22050 Hz → ~86 Hz latent rate)
    """
    def __init__(self, latent_dim=64,
                 channels=(32, 64, 128, 256, 512),
                 strides=(4, 4, 4, 4)):
        super().__init__()
        channels = list(channels)
        strides  = list(strides)
        assert len(channels) == len(strides) + 1

        self.latent_dim = latent_dim

        # Encoder
        enc = [nn.Conv1d(1, channels[0], 7, padding=3)]
        for i, s in enumerate(strides):
            enc.append(EncoderBlock(channels[i], channels[i + 1], s))
        self.encoder      = nn.Sequential(*enc)
        self.mu_proj      = nn.Conv1d(channels[-1], latent_dim, 1)
        self.logvar_proj  = nn.Conv1d(channels[-1], latent_dim, 1)

        # Decoder
        dec_ch = list(reversed(channels))
        dec = [nn.Conv1d(latent_dim, dec_ch[0], 1)]
        for i, s in enumerate(reversed(strides)):
            dec.append(DecoderBlock(dec_ch[i], dec_ch[i + 1], s))
        dec += [nn.LeakyReLU(0.2),
                nn.Conv1d(dec_ch[-1], 1, 7, padding=3),
                nn.Tanh()]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        h      = self.encoder(x)                             # (B, C, T_lat)
        mu     = self.mu_proj(h)                             # (B, D, T_lat)
        logvar = self.logvar_proj(h)
        z      = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return mu, logvar, z

    def decode(self, z):
        return self.decoder(z)                               # (B, 1, T)

    def forward(self, x):
        mu, logvar, z = self.encode(x)
        y_hat = self.decode(z)
        return y_hat, mu, logvar, z


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
