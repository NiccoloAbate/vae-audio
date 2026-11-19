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
- **Variation, Var. seed, Variations, Var. order**: a random constant offset. With Variations > 1, that many seeds are rendered (seed, seed + 1, ...), and each hit picks one, either at random (never the same twice in a row) or in turn (Cycle).
- **Velocity -> / Vel. depth / Vel. -> gain**: see *Velocity* below.
- **Gain, Pan, Choke, Self-choke**: playback only, with no re-render. The hats share choke group 1.

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
