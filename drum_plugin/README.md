# Latent Drum Machine (plugin, v0)

A JUCE instrument (VST3 / AU / Standalone) that plays drum hits rendered by the
drum VAE. It is a **neural sampler**: each pad's hit is rendered in the background
from a latent recipe (the same chain as `app_drums.py`) into velocity layers and
variations, then played back like samples. Triggering is instant and costs no
inference. Changing a control re-renders that pad in about 0.1–0.5 s, and the old sound
keeps playing until the new one is ready.

## Build

Needs CMake, Xcode, and the Python environment with `torch` (libtorch is taken
from it, as in CelloVAESynth). JUCE is fetched on the first configure.

```bash
cd drum_plugin
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This installs `Latent Drum Machine` into `~/Library/Audio/Plug-Ins/{VST3,Components}`
and builds the standalone app under `build/LatentDrumMachine_artefacts/Release/Standalone/`.
The plugin links libtorch inside the Python install, so it runs on this machine
but is not a portable build yet.

## Kits

A kit is a model + bank exported by `../export_kit.py`:

```bash
python export_kit.py -r <checkpoint.pth> -b bank/<bank>.pt \
    --directions sweeps/<run>/directions.pt -o kits/<name>
```

`kits/<name>/kit.pt` (TorchScript renderer with the decoder and bank inside) and
`kit.json` (bank entries, control names). Kits made so far from v3 B1: `v3_B1`
(56 hits, 8 per type), `v3_B1_setB` and `v3_B1_setC` (112 hits each, 16 per type,
no hit shared with the other kits; `build_latent_bank.py --avoid`). The exporter checks that the kit renders
exactly what the app renders. The bank must store sigmas (`build_latent_bank.py
--refresh`). New instances load `kits/v3_B1/kit.pt` (set with
`-DLDM_DEFAULT_KIT=...`); **Load kit...** picks another. The kit path is saved
with the session.

## Playing

8 pads on General MIDI notes (kick 36, snare 38, clap 39, closed hat 42, open hat
46, toms 45/48, crash 49; each pad's note can be changed). Click a pad to select
it and hear it: clicking higher on the pad plays louder.

Per pad:
- **Slot A / B, Morph, Morph mode, Xfade width**: the bank hits and the morph between them (as in the app).
- **Controls**: the kit's latent directions (Tone, Noise, PC1–4 on v3_B1), applied on the sounding frames.
- **Decay, Punch**: time-based operations on the latent sequence.
- **Variation, Var. seed, Variations, Var. order**: a random constant offset. With Variations > 1, that many seeds are rendered (seed, seed + 1, ...), and each hit picks one: at random (Random), in turn (Cycle), or at random but never the same twice in a row (Random, no repeat; with 2 variations that alternates).
- **Velocity -> / Vel. depth / Vel. -> gain**: see *Velocity* below.
- **Stereo width, Width var.**: width > 0 also renders a partner for every hit (a fresh decode seed plus its own random offset of size Width var.), and plays mid = the hit, side = high-passed (partner - hit) / 2 with whatever correlates with the hit removed frame by frame: L = mid + width * side, R = mid - width * side. Left + right is exactly the mono hit, and left and right carry equal energy (without the decorrelation, noisy hits leaned right by up to ~9 dB). The partners differ mostly in the tails, so attacks stay centred, and kicks stay nearly mono. Turning width on the first time renders the partners (about twice the render time); after that the width plays live.
- **Gain, Pan, Pan spread, Tune, Tune spread, Choke, Self-choke**: playback only, with no re-render. Pan spread moves each hit's pan at random by up to that amount around Pan; Tune (semitones) is a varispeed read, so the hit gets shorter as it goes up, as on a sampler; Tune spread detunes each hit at random by up to that many semitones. The hats share choke group 1.

**Your own samples:** drag audio files (wav, aiff, flac, mp3, ...) onto the window:
- onto a pad: sets its Slot A (with two files: A and B);
- onto the Slot A or Slot B box: sets that slot of the selected pad;
- anywhere else on the right panel: Slot A.

The file is prepared like the training data (mono, 44.1 kHz, onset placed at 512
samples, the first 1.49 s, peak 0.95), then encoded (about 0.3 s). It appears under
"your hits" in the slot menus (slots 400+) and works with every control and morph.
Your hits are stored as latents in the session, so a project reopens without the
files. The file path is kept too: loading a kit built on a different model re-encodes
them. Kits exported before the encoder existed can't encode (re-export them).

Global: master level, the decode seed (the σ·ε draw of the sampled decode), and
an optional −50 dB noise gate (not needed with sampled decoding).

### Velocity

16 layers are rendered at v = 0, 1/15, ..., 1, with shift = depth × (1 − v), so full
velocity is always the pad as set, and lower velocities move the target linearly:

| Target | Layer value |
|---|---|
| Morph A-B | morph × (1 − shift): towards A, reaching A at v = 0 with depth 1 |
| A control (Tone, Noise, PC1–4) | control − 3 × shift (up to 3 units lower at v = 0, depth 1) |
| Decay | decay − 2 × shift (log2 tail factor: up to 4× shorter) |
| Gain only | one layer, no re-rendering per velocity |

A hit with MIDI velocity m uses v = m / 127. It plays layers floor(15v) and the next one up,
crossfaded by the fractional part, at gain × (1 − velToGain × (1 − v²)). The whole
set is normalised together, so softer layers keep their own level.

Offline bounces wait for pending renders, so exports always use the current settings.

## Layout

- `src/KitModel.*`: loads `kit.pt` + `kit.json` and calls `render()` (MPS if available, else CPU).
- `src/Layers.*`: pad recipe → render job (velocity layers × variations) → normalised, trimmed, resampled buffers.
- `src/RenderWorker.*`: background thread with debounce, atomic swaps, and a graveyard so the audio thread never frees memory.
- `src/Sampler.*`: voices, two-layer velocity crossfade, choke groups.
- `src/PluginProcessor.*`, `src/PluginEditor.*`: parameters (per pad), MIDI, state, interface.
- `tools/RenderTest.cpp`: `kit_render_test kit.pt out.wav [slot] [velTarget]` renders one pad from the command line.
