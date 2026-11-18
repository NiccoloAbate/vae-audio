"""
Interactive latent drum synthesis — Gradio app.

Pick two hits from the latent bank, drag the morph slider (the velocity /
automation axis) to interpolate between them, and add latent perturbation for
variations. Everything decodes live through the trained VAE decoder.

This is the tangible prototype of the drum-machine engine: morph = "interpolate
toward another drum", perturb = "a new variation of this hit".

Run:
    python app_drums.py                 # uses bank/drums_bank.pt + the drum checkpoint
    python app_drums.py -b <bank.pt> -r <checkpoint.pth> [--device cpu|mps|cuda] [--share]
    python app_drums.py --directions sweeps/prototype/directions.pt   # + knob sliders

Time surgery (no directions needed; the latent is a sequence of frames in time):
Decay resamples the latent after the hit's peak (log2 of the tail-length factor),
Punch holds the peak frame longer, and the morph modes re-time the two hits before
blending them: shape-aligned, tail only, or a crossover from A into B (drum_engine).

Knob sliders (Tone, Noise, ...) appear only with --directions: latent
directions fit and calibrated by control_sweeps.py, applied after morph and
perturbation. Slider units are the sweep's s (one = the change the direction
promises, a std of the descriptor); the range is the sweep's safe range.

Explore sliders (bottom, --explore_pcs / --explore_dims, default 4 + 4): unlabelled
directions from the bank itself (DrumEngine.fit_explore), to hear what the
latent does beyond the named knobs. Units: one = the bank's spread (std) along it.
"""
import argparse
import base64
import io

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr
import soundfile as sf

from drum_engine import DrumEngine

# morph-mode dropdown label -> DrumEngine.morph mode
MODES = {'Blend': 'blend', 'Blend (slerp)': 'slerp', 'Shape-aligned': 'aligned',
         'Tail only': 'tail', 'Crossover in time': 'crossover'}


def _logspec(w, n_fft=1024, hop=256):
    t = torch.tensor(w, dtype=torch.float32)
    S = torch.stft(t, n_fft, hop, n_fft, torch.hann_window(n_fft), return_complex=True)
    return torch.log(S.abs() + 1e-5).numpy()


def build_app(engine):
    bank = engine.bank
    types = bank['types']
    files = [f.split('/')[-1] for f in bank['files']]
    # Dropdown choices as (label, index) so the callback receives an int
    choices = [(f'{i:02d} · {types[i]} · {files[i][:38]}', i) for i in range(len(types))]

    def default_idx(t, fallback):
        hits = [i for i, ty in enumerate(types) if ty == t]
        return hits[0] if hits else fallback

    a0, b0 = default_idx('kick', 0), default_idx('snare', 1)

    knob_keys = list(getattr(engine, 'knobs', {}) or {})
    explore_keys = list(getattr(engine, 'explore', {}) or {})
    dir_keys = knob_keys + explore_keys
    dirs = {**(getattr(engine, 'knobs', {}) or {}), **(getattr(engine, 'explore', {}) or {})}

    def render(a_idx, b_idx, morph, mode, width, sigma, seed, sampled, gated, decay, punch,
               *knob_vals):
        if engine.has_sigma:
            engine.set_sample(sampled, seed)
        z = engine.morph(int(a_idx), int(b_idx), float(morph), MODES[mode], float(width))
        offset = engine.variation(sigma, seed) if sigma > 0 else None
        z = engine.steer(z, dict(zip(dir_keys, knob_vals)), offset)
        z = engine.punch(z, float(punch))                   # last: time operations on the result
        z = engine.stretch(z, 2.0 ** float(decay))
        w = engine.render(z, gate_db=engine.gate_db_default if gated else None).astype(np.float32)
        turned = '  ' + ' '.join(f'{dirs[k]["analogue"]}{v:+.1f}'
                                 for k, v in zip(dir_keys, knob_vals) if v) if dir_keys else ''
        turned += f'  decay x{2.0 ** float(decay):.2f}' if decay else ''
        turned += f'  punch {punch:+.2f}' if punch else ''

        fig, ax = plt.subplots(2, 1, figsize=(7, 4), gridspec_kw={'height_ratios': [1, 2]})
        ax[0].plot(np.arange(len(w)) / engine.sr, w, lw=0.5, color='seagreen')
        ax[0].set_xlim(0, len(w) / engine.sr); ax[0].set_title(
            f'[{types[int(a_idx)]}] →({morph:.2f} {MODES[mode]})→ [{types[int(b_idx)]}]'
            f'{f"  variation {sigma:.2f} #{int(seed)}" if sigma > 0 else ""}{turned}'
            f'  [{"z~q(z|x)" if engine.sample else "z=mu"}, gate {"on" if gated else "off"}]',
            fontsize=9)
        ax[0].set_ylim(-1.05, 1.05); ax[0].tick_params(labelsize=6)
        ax[1].imshow(_logspec(w), aspect='auto', origin='lower', cmap='magma')
        ax[1].set_xlabel('frame'); ax[1].set_ylabel('freq bin'); ax[1].tick_params(labelsize=6)
        fig.tight_layout()
        return (engine.sr, w), fig

    with gr.Blocks(title='Latent Drum Synth') as demo:
        gr.Markdown('## 🥁 Latent Drum Synth\n'
                    'Morph between two bank hits (the **velocity / automation** axis) '
                    'and perturb for variations — decoded live through the VAE.')
        with gr.Row():
            with gr.Column():
                a = gr.Dropdown(choices, value=a0, label='Slot A (morph = 0)')
                orig_a = gr.Button('▷ original', size='sm', variant='secondary')
            with gr.Column():
                b = gr.Dropdown(choices, value=b0, label='Slot B (morph = 1)')
                orig_b = gr.Button('▷ original', size='sm', variant='secondary')
        # hidden player for the originals: each click swaps in a fresh <audio autoplay>
        orig_player = gr.HTML('', padding=False)
        morph = gr.Slider(0, 1, value=0.0, step=0.01, label='Morph  A → B   (velocity axis)')
        with gr.Row():
            mode = gr.Dropdown(list(MODES), value='Blend', label='Morph mode',
                               info='Blend: frame by frame as stored. Shape-aligned: peaks and '
                                    'tail lengths matched first. Tail only: A\'s attack, tail '
                                    'morphed. Crossover: A turns into B, the morph sets when.')
            width = gr.Slider(1, 16, value=4, step=1, label='Crossfade width (frames)',
                              info='Tail only / crossover: how gradual the hand-over is '
                                   '(1 frame = 11.6 ms)')
        with gr.Row():
            sigma = gr.Slider(0, 2, value=0.0, step=0.01, label='Variation amount',
                              info='a random offset of the whole hit (where it sounds), drawn '
                                   'from how the bank\'s hits differ; 1 = about as far as a '
                                   'typical hit is from the bank average')
            seed = gr.Number(value=0, precision=0, label='Variation seed',
                             info='also seeds the sampled decode')
        with gr.Accordion('Diagnostics', open=False), gr.Row():
            sampled = gr.Checkbox(value=engine.has_sigma and engine.sample, interactive=engine.has_sigma,
                                  label='Sampled decode: mu + sigma * eps (as in training)',
                                  info='off = decode the posterior mean mu, which leaves a tonal '
                                       'floor in silence' + ('' if engine.has_sigma else
                                       ' (bank has no sigmas: rebuild it to enable)'))
            gated = gr.Checkbox(value=engine.gate_db is not None,
                                label=f'Noise gate ({engine.gate_db_default:g} dB below the peak)')
        with gr.Row():
            decay = gr.Slider(-2, 2, value=0.0, step=0.1, label='Decay (stretch): tail length x 2^s',
                              info='the latent after the peak is resampled in time; -2 = x0.25, '
                                   '+2 = x4 (long hits stop at the 1.5 s window)')
            punch = gr.Slider(-0.75, 4, value=0.0, step=0.25, label='Punch: hold the peak (frames)',
                              info='the loudest frame is held this many frames longer '
                                   '(11.6 ms each); < 0 shortens it')
        knob_sliders = []
        if knob_keys:
            gr.Markdown('**Knobs** — latent directions from the control sweeps '
                        '(0 = unchanged; range = where steering stayed monotone)')
            with gr.Row():
                for k in knob_keys:
                    d = engine.knobs[k]
                    lo, hi = d['safe_range']
                    info = f'measured slope {d["slope"]:.2f}, {d["monotone"]:.0%} monotone'
                    if lo == hi:          # never steered reliably: offer the swept range
                        lo, hi = engine.knob_grid
                        info += ' (no reliable range: full sweep range shown)'
                    _, desc, kind = k.split('/')
                    knob_sliders.append(gr.Slider(
                        lo, hi, value=0.0, step=0.01, label=f'{d["analogue"]} ({desc}, {kind})',
                        info=info))
        with gr.Row():
            gen = gr.Button('▶ Generate', variant='primary')
            rand_pair = gr.Button('🔀 Random pair')
            new_var = gr.Button('🎲 New variation')
        audio = gr.Audio(label='Output', autoplay=True)
        plot = gr.Plot(label='Waveform + spectrogram')

        explore_sliders = []
        if explore_keys:
            with gr.Accordion('Explore: unlabelled latent directions', open=True):
                gr.Markdown('Principal components of the bank\'s latents and the raw latent '
                            'dims that vary most across it. No promises: 1 = the bank\'s spread '
                            'along that direction; past ±2 the decoder is off its training data.')
                for group in ('pc', 'dim'):
                    keys = [k for k in explore_keys if k.split('/')[1] == group]
                    if keys:
                        with gr.Row():
                            for k in keys:
                                explore_sliders.append(gr.Slider(
                                    -3, 3, value=0.0, step=0.01, label=engine.explore[k]['analogue'],
                                    info=engine.explore[k]['info']))
                reset = gr.Button('Reset explore sliders', size='sm', variant='secondary')

        inputs = [a, b, morph, mode, width, sigma, seed, sampled, gated, decay, punch] \
            + knob_sliders + explore_sliders
        gen.click(render, inputs, [audio, plot])
        # live re-render when the sliders settle
        for sl in [morph, width, sigma, decay, punch] + knob_sliders + explore_sliders:
            sl.release(render, inputs, [audio, plot])
        mode.change(render, inputs, [audio, plot])
        sampled.change(render, inputs, [audio, plot])
        gated.change(render, inputs, [audio, plot])
        if explore_sliders:
            reset.click(lambda: [0.0] * len(explore_sliders), None, explore_sliders).then(
                render, inputs, [audio, plot])

        def _original(idx):
            """The slot's source sample as the model sees it (aligned, normalised,
            cut to its window), played once through an invisible audio element."""
            w = engine.original(int(idx))
            if w is None:
                raise gr.Error(f'Source file not found: {engine.bank["files"][int(idx)]}')
            buf = io.BytesIO()
            sf.write(buf, w.astype(np.float32), engine.sr, format='WAV', subtype='PCM_16')
            b64 = base64.b64encode(buf.getvalue()).decode()
            nonce = np.random.randint(1 << 30)        # new element each click, so replays work
            return (f'<audio id="orig{nonce}" autoplay style="display:none" '
                    f'src="data:audio/wav;base64,{b64}"></audio>')
        orig_a.click(_original, a, orig_player)
        orig_b.click(_original, b, orig_player)

        def _random_pair():
            i, j = np.random.randint(0, len(types), 2)
            return int(i), int(j)
        rand_pair.click(_random_pair, None, [a, b]).then(render, inputs, [audio, plot])

        def _new_seed():
            return int(np.random.randint(0, 1_000_000))
        new_var.click(_new_seed, None, seed).then(render, inputs, [audio, plot])

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-b', '--bank', default='bank/drums_bank.pt')
    ap.add_argument('-r', '--resume', default='saved/models/DrumOneShots/checkpoint-epoch1500.pth')
    ap.add_argument('--decoder_pt', default=None)
    ap.add_argument('--device', default='cpu', help='cpu (default, most stable in a server), mps, or cuda')
    ap.add_argument('--directions', default=None,
                    help='directions.pt from control_sweeps.py: adds knob sliders')
    ap.add_argument('--knobs', nargs='+', default=None,
                    help="direction keys to expose (default: each knob's best usable one)")
    ap.add_argument('--explore_pcs', type=int, default=4,
                    help='bank principal components exposed as explore sliders (0: none)')
    ap.add_argument('--explore_dims', type=int, default=4,
                    help='most-varying raw latent dims exposed as explore sliders (0: none)')
    ap.add_argument('--gate_db', type=float, default=-50.0,
                    help='noise gate below each hit\'s peak, removes the decoder\'s buzz in silence')
    ap.add_argument('--no_gate', action='store_true', help='start with the noise gate off')
    ap.add_argument('--decode', choices=['sample', 'mu'], default='sample',
                    help='start with sampled decoding (mu + sigma * eps, needs a bank with sigmas) '
                         'or mu; the app has a checkbox to switch')
    ap.add_argument('--share', action='store_true')
    ap.add_argument('--port', type=int, default=7860)
    args = ap.parse_args()

    device = torch.device(args.device)
    engine = DrumEngine(checkpoint=args.resume, decoder_pt=args.decoder_pt,
                        bank=args.bank, device=device)
    engine.gate_db_default = args.gate_db
    engine.gate_db = None if args.no_gate else args.gate_db
    if args.decode == 'sample' and not engine.has_sigma:
        print('WARNING: bank has no sigmas (rebuild it with build_latent_bank.py); decoding mu')
    engine.set_sample(args.decode == 'sample' and engine.has_sigma)
    print(f'Loaded bank {engine.bank["latents"].shape} on {device}, '
          f'decode {"mu + sigma * eps" if engine.sample else "mu"}, '
          f'gate {"off" if args.no_gate else f"{args.gate_db:g} dB"}')
    if args.directions:
        keys = engine.load_directions(args.directions, args.knobs)
        if engine.knob_source and engine.knob_source != args.resume:
            print(f'WARNING: directions were fit on {engine.knob_source}, not {args.resume}')
        print(f'Knobs: {keys or "none passed the usability bar"}')
    if args.explore_pcs or args.explore_dims:
        print(f'Explore: {engine.fit_explore(args.explore_pcs, args.explore_dims)}')
    build_app(engine).launch(server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
