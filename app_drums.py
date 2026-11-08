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

Knob sliders (Tone, Snappy, ...) appear only with --directions: latent
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

    def render(a_idx, b_idx, morph, use_slerp, sigma, seed, *knob_vals):
        za, zb = engine.z(int(a_idx)), engine.z(int(b_idx))
        fn = engine.slerp if use_slerp else engine.lerp
        z = fn(za, zb, torch.tensor(float(morph)))
        if sigma > 0:
            g = torch.Generator().manual_seed(int(seed))
            z = engine.perturb(z, float(sigma), g)
        z = engine.steer(z, dict(zip(dir_keys, knob_vals))) if dir_keys else z
        w = engine.render(z).astype(np.float32)
        turned = '  ' + ' '.join(f'{dirs[k]["analogue"]}{v:+.1f}'
                                 for k, v in zip(dir_keys, knob_vals) if v) if dir_keys else ''

        fig, ax = plt.subplots(2, 1, figsize=(7, 4), gridspec_kw={'height_ratios': [1, 2]})
        ax[0].plot(np.arange(len(w)) / engine.sr, w, lw=0.5, color='seagreen')
        ax[0].set_xlim(0, len(w) / engine.sr); ax[0].set_title(
            f'[{types[int(a_idx)]}] →({morph:.2f})→ [{types[int(b_idx)]}]'
            f'{"  +noise" if sigma > 0 else ""}{turned}', fontsize=9)
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
            sigma = gr.Slider(0, 1.2, value=0.0, step=0.01, label='Perturbation σ (variation)')
            use_slerp = gr.Checkbox(value=False, label='Spherical (slerp)')
            seed = gr.Number(value=0, precision=0, label='Variation seed')
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

        inputs = [a, b, morph, use_slerp, sigma, seed] + knob_sliders + explore_sliders
        gen.click(render, inputs, [audio, plot])
        # live re-render when the sliders settle
        for sl in [morph, sigma] + knob_sliders + explore_sliders:
            sl.release(render, inputs, [audio, plot])
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
    ap.add_argument('--share', action='store_true')
    ap.add_argument('--port', type=int, default=7860)
    args = ap.parse_args()

    device = torch.device(args.device)
    engine = DrumEngine(checkpoint=args.resume, decoder_pt=args.decoder_pt,
                        bank=args.bank, device=device)
    print(f'Loaded bank {engine.bank["latents"].shape} on {device}')
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
