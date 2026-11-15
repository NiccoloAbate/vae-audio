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
        for seed in args.seeds:
            c = copy.deepcopy(base)
            c['name'] = f'Drums_v2_{arm}{seed}'
            c['seed'] = seed
            t = c['trainer']
            t.update(beta_warmup=ep['beta_warmup'], adv_start_epoch=ep['adv_start'],
                     epochs=ep['total'], early_stop=ep['total'], save_period=ep['save'])
            c['lr_scheduler']['args']['step_size'] = ep['lr_step']
            c['data_loader']['args']['exclude'] = args.exclude
            t.update(over.get('trainer', {}))
            c['data_loader']['args'].update(over.get('data_loader', {}))
            path = f'config_raw_drums_v2_{arm}{seed}.json'
            json.dump(c, open(path, 'w'), indent=4)
            print('wrote', path)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='v2 configs with b005\'s schedule in steps')
    p.add_argument('--data_dir', required=True)
    p.add_argument('--base', default='config_raw_drums_44k_b005.json')
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    p.add_argument('--exclude', default='splits/exclude_train.txt', help='from dedupe_splits.py')
    main(p.parse_args())
