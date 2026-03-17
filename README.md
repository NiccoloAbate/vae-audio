# Exploring VAEs for Audio Synthesis and Timbre Transfer

EE 269 — Signal Processing of Music and Audio
CCRMA, Stanford University — Niccolo Abate, March 2026

This project explores Variational Autoencoders (VAEs) as a framework for audio synthesis and timbre transfer, progressing from a spectrogram-domain baseline (SpecVAE) to a raw-waveform model (RawAudioVAE) trained on increasingly large datasets, with an optional adversarial extension.

---

## Models

**SpecVAE** — spectrogram-domain baseline adapted from [yjlolo/vae-audio](https://github.com/yjlolo/vae-audio). Operates on log-mel spectrograms, reconstructs via Griffin-Lim.

**RawAudioVAE** — raw waveform VAE with strided dilated-residual Conv1d encoder/decoder (inspired by RAVE). Learns a temporal latent representation at ~86 Hz from 22 kHz audio. Trained with multi-resolution STFT loss, β warmup, and free bits.

**RawAudioVAE + Adversarial** — extends RawAudioVAE with a multi-scale spectrogram discriminator and feature-matching loss for improved perceptual quality.

---

## Demo

Out-of-distribution sounds (percussion and other tonal instruments) encoded into the Medley-Solos-DB VAE latent space and interpolated between using slerp.

[![Demo](https://img.youtube.com/vi/pBTpLXmS16Y/0.jpg)](https://www.youtube.com/watch?v=pBTpLXmS16Y)

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python train.py -c config_raw.json            # RawAudioVAE
python train.py -c config_raw_adv.json        # RawAudioVAE + adversarial
python train.py -c config_raw_medley.json     # RawAudioVAE on Medley-Solos-DB
```

Checkpoints and logs are saved to `saved/`. Training can be monitored with TensorBoard:

```bash
tensorboard --logdir saved/log
```

---

## Evaluation

```bash
python evaluate_raw.py -r <checkpoint>.pth -o eval/my_run
```

Produces spectrograms, waveform comparisons, KL-per-dimension plot, PCA latent scatter, interpolation grids, and audio files.

---

## Interactive Demo

Upload 2–3 audio files and morph between them in the learned latent space using slerp interpolation.

```bash
python demo_raw.py -r <checkpoint>.pth
```

Two interpolation modes are available: **per-chunk slerp** (morphs full latent sequence) and **mean + offset** (morphs timbral identity while preserving temporal structure).

---

## Datasets

- **Small dataset** — 29 recordings, 2 classes (included in repo via [yjlolo/vae-audio](https://github.com/yjlolo/vae-audio))
- **ESC-50** — 2,000 environmental sound clips, 50 classes
- **Medley-Solos-DB** — 21,571 clips of 8 tonal instruments

Place audio files under `dataset/` and point the config `data_dir` field accordingly.
