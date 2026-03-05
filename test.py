import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.model as module_arch
from parse_config import ConfigParser


def main(config, resume):
    logger = config.get_logger('test')

    # Load test set: use config's data_dir but override subset and validation settings
    dl_args = dict(config['data_loader']['args'])
    dl_args.update({'validation_split': 0.0, 'shuffle': False, 'subset': 'test'})
    data_loader = getattr(module_data, config['data_loader']['type'])(**dl_args)

    # Build model and load checkpoint
    model = config.initialize('arch', module_arch)
    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])

    # Device selection: MPS > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info('Using device: {}'.format(device))

    model = model.to(device)
    model.eval()

    loss_fn = getattr(module_loss, config['loss'])

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    with torch.no_grad():
        for data_idx, label, data in tqdm(data_loader):
            x = data.type('torch.FloatTensor').to(device)
            # Reshape (batch, chunks, freq, time) -> (batch*chunks, 1, freq, time)
            n_freqBand, n_contextWin = x.size(2), x.size(3)
            x = x.view(-1, 1, n_freqBand, n_contextWin)

            x_recon, mu, logvar, z = model(x)
            loss_recon, loss_kl = loss_fn(mu, logvar, x_recon, x.squeeze(1))
            loss = loss_recon + loss_kl

            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_kl += loss_kl.item()
            n_batches += 1

    log = {
        'loss': total_loss / n_batches,
        'loss_recon': total_recon / n_batches,
        'loss_kl': total_kl / n_batches,
    }
    for key, value in log.items():
        logger.info('    {:15s}: {}'.format(key, value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE-Audio Test')
    parser.add_argument('-r', '--resume', required=True, type=str,
                        help='path to checkpoint to evaluate')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()
    config = ConfigParser(parser)
    main(config, args.resume)
