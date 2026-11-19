"""
Write the v2 run configs (new data): A = the b005 recipe, B = A + frozen encoder,
augmentation and mild oversampling of the rarest types, two seeds each. Both skip
the training copies of val/test audio (splits/exclude_train.txt, dedupe_splits.py).

The trainer schedules in epochs, so with more data the same epoch counts mean more
steps. This counts the training files and converts b005's schedule, in optimizer
steps, to epochs: beta warm-up 78k, LR halved every 116k, adversarial start 195k,
end 388k (b005's FAD had plateaued by then: 0.070 at 388k, 0.069 at 582k), and a
checkpoint every ~19k (pass the printed save_period to eval_trajectory.py --every).

Run it once the dataset is final, where the data is:
    python make_v2_configs.py --data_dir /scratch/users/nlabate/datasets/drums_oneshots

v3 = the same runs with the RAVE-style amplitude output (arch amp_out), which lets
the decoder output true silence instead of a frame-rate buzz:
    python make_v2_configs.py --data_dir ... --tag v3 --amp_out

v4 = single changes on v3 B1, aimed at cymbal/open-hat timbre (outputs smear the
partials into smooth noise): a RAVE noise branch, or 4096/8192 STFT loss scales:
    python make_v2_configs.py --data_dir ... --tag v4noise --amp_out --noise_hop 128 --arms B --seeds 1
    python make_v2_configs.py --data_dir ... --tag v4fft --amp_out --loss_fft 8192 4096 2048 1024 512 256 --arms B --seeds 1

v5 = v3 B1 on the data with the two Freesound cymbal packs (import_freesound.py),
then + a GAN stage that actually pulls: on v2/v3 the adversarial and feature-matching
gradients were only 0.1-0.8% of the recon gradient at the output. v5gan sets each to
10% (balancer in target mode, ramped in over 10k steps) and adds a multi-period and
a complex-STFT discriminator to v3's waveform one; v5bal is the balancer alone:
    python make_v2_configs.py --data_dir ... --tag v5data --amp_out --arms B --seeds 1
    python make_v2_configs.py --data_dir ... --tag v5gan --amp_out --arms B --seeds 1 --trainer_json \
        '{"discriminator": "hybrid", "disc_capacity": 8, "disc_update_every": 1,
          "disc_args": {"parts": ["msd", "mpd", "stft"], "mpd_capacity": 16, "stft_filters": 32},
          "balance": {"adv": 0.1, "fm": 0.1}, "balance_mode": "target", "balance_warmup": 10000}'
    python make_v2_configs.py --data_dir ... --tag v5bal --amp_out --arms B --seeds 1 --trainer_json \
        '{"balance": {"adv": 0.1, "fm": 0.1}, "balance_mode": "target", "balance_warmup": 10000}'
"""

import argparse
import copy
import json
import math

from dataset.datasets import OneShotAudioDataset

STEPS = dict(beta_warmup=77_600, lr_step=116_400, adv_start=194_000, total=388_000, save=19_400)

ARMS = {
    'A': {},
    'B': {'trainer': {'freeze_encoder': True},
          'data_loader': {'augment': {'p_pitch': 0.5, 'semitones': 2.0,
                                      'p_decay': 0.25, 'decay_start': [0.05, 0.4],
                                      'decay_tau': [0.02, 0.15]},
                          'type_weights': {'cymbal': 2.0, 'tom': 2.0, 'clap': 1.5}}},
}


def main(args):
    base = json.load(open(args.base))
    dl = base['data_loader']['args']
    ds = OneShotAudioDataset(args.data_dir, sr=dl['sr'], chunk_size=dl['chunk_size'],
                             split='train', types=dl['types'], exclude=args.exclude)
    spe = math.ceil(len(ds) / dl['batch_size'])
    ep = {k: max(1, round(v / spe)) for k, v in STEPS.items()}
    print(f'{len(ds)} training files, {spe} steps/epoch -> {ep}')
    for arm, over in ARMS.items():
        if args.arms and arm not in args.arms:
            continue
        for seed in args.seeds:
            c = copy.deepcopy(base)
            c['name'] = f'Drums_{args.tag}_{arm}{seed}'
            c['seed'] = seed
            t = c['trainer']
            t.update(beta_warmup=ep['beta_warmup'], adv_start_epoch=ep['adv_start'],
                     epochs=ep['total'], early_stop=ep['total'], save_period=ep['save'])
            c['lr_scheduler']['args']['step_size'] = ep['lr_step']
            c['data_loader']['args']['exclude'] = args.exclude
            if args.amp_out:
                c['arch']['args']['amp_out'] = True
            if args.noise_hop:
                c['arch']['args']['noise_hop'] = args.noise_hop
            if args.loss_fft:
                c['loss_args'] = {'fft_sizes': args.loss_fft, 'hop_sizes': [n // 4 for n in args.loss_fft]}
            t.update(over.get('trainer', {}))
            t.update(json.loads(args.trainer_json) if args.trainer_json else {})
            c['data_loader']['args'].update(over.get('data_loader', {}))
            path = f'config_raw_drums_{args.tag}_{arm}{seed}.json'
            json.dump(c, open(path, 'w'), indent=4)
            print('wrote', path)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='v2 configs with b005\'s schedule in steps')
    p.add_argument('--data_dir', required=True)
    p.add_argument('--base', default='config_raw_drums_44k_b005.json')
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    p.add_argument('--tag', default='v2', help='run names Drums_<tag>_<arm><seed>')
    p.add_argument('--amp_out', action='store_true', help='RAVE-style amplitude output (v3)')
    p.add_argument('--noise_hop', type=int, default=0, help='RAVE noise branch frame hop (0 = off)')
    p.add_argument('--loss_fft', type=int, nargs='+', help='STFT loss window sizes (hop = n/4)')
    p.add_argument('--arms', nargs='+', help='subset of arms (default all)')
    p.add_argument('--trainer_json', help='JSON dict merged into config["trainer"] last')
    p.add_argument('--exclude', default='splits/exclude_train.txt', help='from dedupe_splits.py')
    main(p.parse_args())
