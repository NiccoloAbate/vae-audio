import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from model.model import MultiScaleSpecDiscriminator
from model.loss import discriminator_loss, generator_adversarial_loss, feature_matching_loss


class SpecVaeTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super(SpecVaeTrainer, self).__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _reshape(self, x):
        n_freqBand, n_contextWin = x.size(2), x.size(3)
        return x.view(-1, 1, n_freqBand, n_contextWin)

    def _forward_and_computeLoss(self, x, target):
        x_recon, mu, logvar, z = self.model(x)
        loss_recon, loss_kl = self.loss(mu, logvar, x_recon, target.squeeze(1))
        loss = loss_recon + 0.1 * loss_kl
        return loss, loss_recon, loss_kl

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_recon = 0
        total_kl = 0
        # total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data_idx, label, data) in enumerate(self.data_loader):
            x = data.type('torch.FloatTensor').to(self.device)
            x = self._reshape(x)

            self.optimizer.zero_grad()
            loss, loss_recon, loss_kl = self._forward_and_computeLoss(x, x)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_kl += loss_kl.item()
            # total_metrics += self._eval_metrics(output, target)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                # TODO: visualize input/reconstructed spectrograms in TensorBoard
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
            'loss_recon': total_recon / len(self.data_loader),
            'loss_kl': total_kl / len(self.data_loader)
            # 'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_recon = 0
        total_val_kl = 0
        # total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data_idx, label, data) in enumerate(self.valid_data_loader):
                x = data.type('torch.FloatTensor').to(self.device)
                x = self._reshape(x)

                loss, loss_recon, loss_kl = self._forward_and_computeLoss(x, x)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_recon += loss_recon.item()
                total_val_kl += loss_kl.item()
                # total_val_metrics += self._eval_metrics(output, target)
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_loss_recon': total_val_recon / len(self.valid_data_loader),
            'val_loss_kl': total_val_kl / len(self.valid_data_loader)
            # 'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }


class RawAudioVaeTrainer(BaseTrainer):
    """Trainer for RawAudioVAE. Input data is (B, 1, T) waveform chunks."""

    def __init__(self, model, loss, metrics, optimizer, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, loss, metrics, optimizer, config)
        self.data_loader       = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation     = valid_data_loader is not None
        self.lr_scheduler      = lr_scheduler
        self.log_step          = int(np.sqrt(data_loader.batch_size))
        cfg = config['trainer']
        self.beta_max    = cfg.get('beta_max',    0.05)
        self.beta_warmup = cfg.get('beta_warmup', 100)
        self.free_bits   = cfg.get('free_bits',   0.25)

    def _beta(self, epoch):
        """Linear KL warmup: 0 → beta_max over beta_warmup epochs."""
        return min(epoch / self.beta_warmup, 1.0) * self.beta_max

    def _forward_and_computeLoss(self, x, epoch):
        y_hat, mu, logvar, z = self.model(x)
        loss_recon, loss_kl  = self.loss(x, y_hat, mu, logvar, free_bits=self.free_bits)
        loss = loss_recon + self._beta(epoch) * loss_kl
        return loss, loss_recon, loss_kl

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss, total_recon, total_kl = 0, 0, 0

        for batch_idx, (data_idx, label, data) in enumerate(self.data_loader):
            x = data.float().to(self.device)
            self.optimizer.zero_grad()
            loss, loss_recon, loss_kl = self._forward_and_computeLoss(x, epoch)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss  += loss.item()
            total_recon += loss_recon.item()
            total_kl    += loss_kl.item()

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        loss.item()))

        log = {
            'loss':       total_loss  / len(self.data_loader),
            'loss_recon': total_recon / len(self.data_loader),
            'loss_kl':    total_kl    / len(self.data_loader),
            'beta':       self._beta(epoch),
        }
        if self.do_validation:
            log = {**log, **self._valid_epoch(epoch)}
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        total_val_loss, total_val_recon, total_val_kl = 0, 0, 0

        with torch.no_grad():
            for batch_idx, (data_idx, label, data) in enumerate(self.valid_data_loader):
                x    = data.float().to(self.device)
                _, loss_recon, loss_kl = self._forward_and_computeLoss(x, epoch)
                # Use fixed beta_max for validation so val_loss is a consistent
                # epoch-independent metric for checkpoint selection.
                loss = loss_recon + self.beta_max * loss_kl
                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss  += loss.item()
                total_val_recon += loss_recon.item()
                total_val_kl    += loss_kl.item()

        if epoch % 5 == 0:
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss':       total_val_loss  / len(self.valid_data_loader),
            'val_loss_recon': total_val_recon / len(self.valid_data_loader),
            'val_loss_kl':    total_val_kl    / len(self.valid_data_loader),
        }


class RawAudioVaeAdversarialTrainer(BaseTrainer):
    """VAE-GAN trainer for RawAudioVAE + MultiScaleDiscriminator.

    The discriminator and its optimizer are created internally.
    Adversarial loss is phased in after `adv_start_epoch` so the VAE
    can first learn a reasonable reconstruction before the disc is introduced.

    Per-batch update order:
        1. Forward VAE → y_hat
        2. Update disc:  hinge loss on disc(x) vs disc(y_hat.detach())
        3. Update VAE:   STFT recon + β·KL + λ_adv·gen_hinge + λ_fm·feat_match
    """

    def __init__(self, model, loss, metrics, optimizer, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, loss, metrics, optimizer, config)
        self.data_loader       = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation     = valid_data_loader is not None
        self.lr_scheduler      = lr_scheduler
        self.log_step          = int(np.sqrt(data_loader.batch_size))

        # KL annealing + free bits (configurable via config['trainer'])
        cfg_adv = config['trainer']
        self.beta_max    = cfg_adv.get('beta_max',    0.05)
        self.beta_warmup = cfg_adv.get('beta_warmup', 100)
        self.free_bits   = cfg_adv.get('free_bits',   0.25)

        # Adversarial hyper-parameters
        self.adv_start_epoch = cfg_adv.get('adv_start_epoch', 50)
        self.lambda_adv      = cfg_adv.get('lambda_adv', 1.0)
        self.lambda_fm       = cfg_adv.get('lambda_fm', 2.0)

        # Discriminator (created here so it isn't exposed to train.py)
        self.disc = MultiScaleSpecDiscriminator().to(self.device)
        self.disc_optimizer = torch.optim.Adam(
            self.disc.parameters(), lr=3e-4, betas=(0.5, 0.9)
        )

    def _beta(self, epoch):
        return min(epoch / self.beta_warmup, 1.0) * self.beta_max

    def _train_epoch(self, epoch):
        self.model.train()
        self.disc.train()
        use_adv = epoch >= self.adv_start_epoch

        total_loss = total_recon = total_kl = 0
        total_disc = total_gen_adv = total_fm = 0

        for batch_idx, (data_idx, label, data) in enumerate(self.data_loader):
            x = data.float().to(self.device)

            # ── Forward VAE ──────────────────────────────────────────────
            y_hat, mu, logvar, z = self.model(x)
            loss_recon, loss_kl  = self.loss(x, y_hat, mu, logvar, free_bits=self.free_bits)

            if use_adv:
                # ── Update Discriminator ──────────────────────────────────
                real_out   = self.disc(x)
                fake_out_d = self.disc(y_hat.detach())
                loss_disc  = discriminator_loss(real_out, fake_out_d)

                self.disc_optimizer.zero_grad()
                loss_disc.backward()
                self.disc_optimizer.step()

                # ── Adversarial + Feature-Matching losses for VAE ─────────
                # real_out tensor values survive backward(); reuse for FM.
                fake_out_gen = self.disc(y_hat)          # gradients flow to VAE
                loss_gen_adv = generator_adversarial_loss(fake_out_gen)
                loss_fm      = feature_matching_loss(real_out, fake_out_gen)

            # ── Update VAE ───────────────────────────────────────────────
            beta = self._beta(epoch)
            if use_adv:
                vae_loss = (loss_recon
                            + beta * loss_kl
                            + self.lambda_adv * loss_gen_adv
                            + self.lambda_fm  * loss_fm)
            else:
                vae_loss = loss_recon + beta * loss_kl

            self.optimizer.zero_grad()
            vae_loss.backward()
            self.optimizer.step()

            # ── Logging ──────────────────────────────────────────────────
            step = (epoch - 1) * len(self.data_loader) + batch_idx
            self.writer.set_step(step)
            self.writer.add_scalar('loss',       vae_loss.item())
            self.writer.add_scalar('loss_recon', loss_recon.item())
            self.writer.add_scalar('loss_kl',    loss_kl.item())
            if use_adv:
                self.writer.add_scalar('loss_disc',    loss_disc.item())
                self.writer.add_scalar('loss_gen_adv', loss_gen_adv.item())
                self.writer.add_scalar('loss_fm',      loss_fm.item())

            total_loss  += vae_loss.item()
            total_recon += loss_recon.item()
            total_kl    += loss_kl.item()
            if use_adv:
                total_disc    += loss_disc.item()
                total_gen_adv += loss_gen_adv.item()
                total_fm      += loss_fm.item()

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}{}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        vae_loss.item(),
                        ' [adv ON]' if use_adv else ''))

        n = len(self.data_loader)
        log = {
            'loss':       total_loss  / n,
            'loss_recon': total_recon / n,
            'loss_kl':    total_kl    / n,
            'beta':       self._beta(epoch),
        }
        if use_adv:
            log['loss_disc']    = total_disc    / n
            log['loss_gen_adv'] = total_gen_adv / n
            log['loss_fm']      = total_fm      / n

        if self.do_validation:
            log = {**log, **self._valid_epoch(epoch)}
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        total_val_loss = total_val_recon = total_val_kl = 0

        with torch.no_grad():
            for batch_idx, (data_idx, label, data) in enumerate(self.valid_data_loader):
                x = data.float().to(self.device)
                y_hat, mu, logvar, z = self.model(x)
                loss_recon, loss_kl  = self.loss(x, y_hat, mu, logvar, free_bits=self.free_bits)
                loss = loss_recon + self.beta_max * loss_kl

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss  += loss.item()
                total_val_recon += loss_recon.item()
                total_val_kl    += loss_kl.item()

        if epoch % 10 == 0:
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')

        n = len(self.valid_data_loader)
        return {
            'val_loss':       total_val_loss  / n,
            'val_loss_recon': total_val_recon / n,
            'val_loss_kl':    total_val_kl    / n,
        }


class GMVAETrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super(GMVAETrainer, self).__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _reshape(self, x):
        # assume dimensions to be [batch_size, n_freqBand, n_contextWin]
        return x.unsqueeze(2)

    def _forward_and_computeLoss(self, x, target):
        x_recon, q_mu, q_logvar, z, logLogit_qy_x, qy_x, y = self.model(x)
        neg_logpx_z, kld_latent, kld_class = self.loss(x_recon, target, logLogit_qy_x, qy_x, q_mu, q_logvar,
                                                        self.model.mu_lookup, self.model.logvar_lookup,
                                                        self.model.n_component)
        loss = neg_logpx_z + kld_latent + kld_class
        return loss, neg_logpx_z, kld_latent, kld_class

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_recon = 0
        total_kld_latent = 0
        total_kld_class = 0
        # total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data_idx, label, data) in enumerate(self.data_loader):
            x = data.type('torch.FloatTensor').to(self.device)
            # x = self._reshape(x)

            self.optimizer.zero_grad()
            loss, neg_logpx_z, kld_latent, kld_class = self._forward_and_computeLoss(x, x)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_recon += neg_logpx_z.item()
            total_kld_latent += kld_latent.item()
            total_kld_class += kld_class.item()
            # total_metrics += self._eval_metrics(output, target)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                # TODO: visualize input/reconstructed spectrograms in TensorBoard
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
            'loss_recon': total_recon / len(self.data_loader),
            'loss_kld_latent': total_kld_latent / len(self.data_loader),
            'loss_kld_class': total_kld_class / len(self.data_loader)
            # 'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_kld_latent = 0
        total_kld_class = 0
        # total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data_idx, label, data) in enumerate(self.valid_data_loader):
                x = data.type('torch.FloatTensor').to(self.device)
                # x = self._reshape(x)

                loss, neg_logpx_z, kld_latent, kld_class = self._forward_and_computeLoss(x, x)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_loss += loss.item()
                total_recon += neg_logpx_z.item()
                total_kld_latent += kld_latent.item()
                total_kld_class += kld_class.item()
                # total_val_metrics += self._eval_metrics(output, target)
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_loss / len(self.valid_data_loader),
            'val_loss_recon': total_recon / len(self.valid_data_loader),
            'val_loss_kld_latent': total_kld_latent / len(self.valid_data_loader),
            'val_loss_kld_class': total_kld_class / len(self.valid_data_loader)
            # 'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
