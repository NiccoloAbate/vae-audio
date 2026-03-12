import numpy as np
import torch
import torch.nn.functional as F
from base import approx_qy_x


def multi_res_stft_loss(y, y_hat,
                        fft_sizes=(2048, 1024, 512, 256),
                        hop_sizes=(512,  256,  128,  64)):
    """
    Multi-resolution STFT loss (spectral convergence + log magnitude L1).
    y, y_hat : (B, 1, T) waveforms — kept on the original device for gradient flow.
    """
    device = y.device
    y     = y.squeeze(1).float()
    y_hat = y_hat.squeeze(1).float()

    total = torch.zeros(1, device=device)
    for n_fft, hop in zip(fft_sizes, hop_sizes):
        win = torch.hann_window(n_fft, device=device)
        Y    = torch.stft(y,     n_fft, hop, n_fft, win, return_complex=True)
        Yh   = torch.stft(y_hat, n_fft, hop, n_fft, win, return_complex=True)
        Y_m  = Y.abs()
        Yh_m = Yh.abs()
        # spectral convergence
        sc   = (Y_m - Yh_m).norm() / (Y_m.norm() + 1e-8)
        # log magnitude L1
        log  = F.l1_loss(torch.log(Y_m + 1e-5), torch.log(Yh_m + 1e-5))
        total = total + sc + log

    return total / len(fft_sizes)


def kld_temporal(mu, logvar, free_bits=1.0):
    """KL divergence for temporal latents (B, D, T): sum over D,T; mean over B.

    free_bits: minimum KL per latent dimension (averaged over time).
    Dimensions below this threshold are not penalized, preventing posterior collapse.
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, D, T)
    if free_bits > 0:
        kl_per_dim = kl.mean(dim=2).clamp(min=free_bits)  # (B, D)
        return kl_per_dim.sum(dim=1).mean()
    return kl.sum(dim=(1, 2)).mean()


def raw_audio_vae_loss(y, y_hat, mu, logvar, free_bits=0.25):
    """Combined loss for RawAudioVAE: returns (recon_loss, kl_loss)."""
    return multi_res_stft_loss(y, y_hat), kld_temporal(mu, logvar, free_bits)


def discriminator_loss(real_outputs, fake_outputs):
    """Hinge loss for the multi-scale discriminator.

    real_outputs / fake_outputs: list of (feature_maps, logits) — one per scale.
    Loss = mean over scales of [ relu(1-real) + relu(1+fake) ].
    """
    loss = torch.zeros(1, device=real_outputs[0][1].device)
    for (_, real_logits), (_, fake_logits) in zip(real_outputs, fake_outputs):
        loss = loss + F.relu(1.0 - real_logits).mean()
        loss = loss + F.relu(1.0 + fake_logits).mean()
    return loss / len(real_outputs)


def generator_adversarial_loss(fake_outputs):
    """Hinge generator loss: fool the discriminator by pushing logits positive.

    fake_outputs: list of (feature_maps, logits) from disc(y_hat).
    """
    loss = torch.zeros(1, device=fake_outputs[0][1].device)
    for _, fake_logits in fake_outputs:
        loss = loss - fake_logits.mean()
    return loss / len(fake_outputs)


def feature_matching_loss(real_outputs, fake_outputs):
    """L1 between discriminator intermediate feature maps (real vs fake).

    Real features are detached so no gradient flows back through them.
    fake_outputs must NOT be detached — gradients flow to the generator.
    """
    loss = torch.zeros(1, device=fake_outputs[0][1].device)
    n = 0
    for (real_feats, _), (fake_feats, _) in zip(real_outputs, fake_outputs):
        for rf, ff in zip(real_feats, fake_feats):
            loss = loss + F.l1_loss(ff, rf.detach())
            n += 1
    return loss / n


def vae_loss(q_mu, q_logvar, output, target):
    return mse_loss(output, target), kld_gauss(q_mu, q_logvar)


def gmvae_loss(output, target, logLogit_qy_x, qy_x, q_mu, q_logvar, mu_lookup, logvar_lookup, n_component):
    """
    Basic GMVAE loss (https://arxiv.org/abs/1611.05148)
    """
    return mse_loss(output, target),\
        kld_latent(qy_x, q_mu, q_logvar, mu_lookup, logvar_lookup),\
        kld_class(logLogit_qy_x, qy_x, n_component)


def mse_loss(output, target, avg_batch=True):
    """
    Reconstruction loss
    To prevent posterior collapse of q(y|x) in GMVAE, there is no normalization performed w.r.t.
    number of frequency bins and time frames; which makes scale of reconstruction loss relatively large
    compared to KL terms.
    TODO:
        [] Allow optional normalization w.r.t. frequency and time axis
        [] Find a good normalization scheme w.r.t frequency and time axis
    """
    output = F.mse_loss(output, target, reduction='none')
    output = torch.sum(output)  # sum over all TF units
    if avg_batch:
        output = torch.mean(output, dim=0)
    # return F.mse_loss(output, target, reduction=reduce)  # careful about the scaling
    return output


def kld_gauss(q_mu, q_logvar, mu=None, logvar=None, avg_batch=True):
    """
    KL divergence between two diagonal Gaussians
    in standard VAEs, the prior p(z) is a standard Gaussian.
    :param q_mu: posterior mean
    :param q_logvar: posterior log-variance
    :param mu: prior mean
    :param logvar: prior log-variance
    """
    # set prior to a standard Gaussian
    if mu is None:
        mu = torch.zeros_like(q_mu)
    if logvar is None:
        logvar = torch.zeros_like(q_logvar)

    output = torch.sum(1 + q_logvar - logvar - (torch.pow(q_mu - mu, 2) + torch.exp(q_logvar)) / torch.exp(logvar),
                        dim=1)
    output *= -0.5
    if avg_batch:
        output = torch.mean(output, dim=0)
    return output


def kld_class(logLogit_qy_x, qy_x, n_component, avg_batch=True):
    h_qy_x = torch.sum(qy_x * torch.nn.functional.log_softmax(logLogit_qy_x, dim=1), dim=1)
    output = h_qy_x - np.log(1 / n_component)
    if avg_batch:
        output = torch.mean(output, dim=0)
    # return h_qy_x - np.log(1 / n_component)  # , h_qy_x
    return output


def kld_latent(qy_x, q_mu, q_logvar, mu_lookup, logvar_lookup, avg_batch=True):
    """
    Calculate the term of KLD in the ELBO of GMVAEs:
    sum_{y}{ q(y|x) * KLD[ q(z|x) | p(z|y) ] }
    :param qy_x: q(y|x)
    :param q_mu: approximated posterior mean
    :param q_logvar: approximated posterior log-variance
    :param mu_lookup: conditional prior mean
    :param logvar_lookup: conditional prior log-variance
    """
    batch_size, n_component = list(qy_x.size())
    kl_sumOver = torch.zeros(batch_size, n_component)
    for k_i in torch.arange(0, n_component):
        # KLD
        kl_sumOver[:, k_i] = kld_gauss(q_mu, q_logvar, mu_lookup(k_i), logvar_lookup(k_i), avg_batch=False)
        # weighted sum by q(y|x)
        kl_sumOver[:, k_i] *= qy_x[:, k_i]
    # sum over components
    output = torch.sum(kl_sumOver, dim=1)
    if avg_batch:
        output = torch.mean(output, dim=0)
    return output
