"""
Milestone evaluation: score one system on the fixed test split.

Two ways in:
    -r CKPT          a RawAudioVAE checkpoint; every test hit is encoded and
                     decoded (reconstruction), and the latent is analysed.
    --audio_dir DIR  any other system (DSP baseline, DrumBlender, diffusion)
                     that has written one wav per test file, named
                     flat_name(relpath), i.e. the path with '/' -> '__'.
    --gen_dir DIR    an unpaired *generated* set (generate_set.py,
                     gen_stable_audio.py): preprocessed like the reference,
                     then scored with FAD only, since there is no paired
                     target. Timing comes from DIR/timing.json if present.

Every system is scored at ONE evaluation rate and length (default 44.1 kHz,
65536 samples ~ 1.49 s) against the same reference audio, whatever its native
rate: output is cropped to the eval duration and resampled. A 22 kHz model
therefore pays for its missing top octave, which is the point of comparing it
with the 44.1 kHz model. `stft_native` repeats the STFT error at the model's
own rate, to separate "band-limited" from "inaccurate".

Metrics (paired, per hit, then averaged):
    stft      multi-resolution STFT distance: the training loss with magnitudes
              floored at -80 dBFS so inaudible tails don't dominate (stft_err)
    lat_med   median |LAT(sys) - LAT(ref)| in decades, where LAT is the
              log10 attack time (20% -> 90% of the envelope peak); 0.30 = a
              factor of 2. Median, since envelope-peak misses (flams, double
              hits) make the mean outlier-driven. lat_within2 = share of hits
              within x2; lat_bias = median signed error (> 0 = slower attack)
    fad       Frechet Audio Distance via fadtk (separate env, see below)
    rtf       decode time / audio duration, batch 1 on CPU (plugin setting)
  latent usage (checkpoint mode only):
    pca_hit_95    PCs of the time-averaged latent (one vector per hit) for 95%
                  of its variance: roughly, the dimensionality of hit identity
    pca_frame_95  same over every latent frame: also counts how the latent
                  moves within a hit
    kl_dims       dims whose mean KL exceeds --kl_thresh (0.2 nats, i.e. above
                  the 0.15 free-bits floor where unused dims are parked)
    trunc_k90     fewest per-frame PCs that recover 90% of the reconstruction
                  gain (error with all PCs vs with none); full curve in
                  truncation.png. Asks what the decoder uses, not just where
                  the variance is.
    kl_total      KL in nats per frame, summed over dims

FAD runs through the fadtk CLI, which needs Python <= 3.13.0 and so lives in
its own env (plus sox):
    conda create -n fad python=3.12 && conda run -n fad pip install torch fadtk
    brew install sox
then pass --fadtk /opt/anaconda3/envs/fad/bin/fadtk. Without --fadtk, FAD is skipped (the
audio is still written, so it can be added later with --fad_only). Run
`--fad_sanity` once per embedding: FAD(test real, val real) is the floor any
system can reach on this set size.

Outputs:
    eval/reference/<split>_<sr>_<len>/   reference wavs (shared by all systems)
    <out_dir>/audio/                     system wavs, same file names
    <out_dir>/per_hit.csv, per_type.txt, recon_spectrograms.png,
    <out_dir>/latent_pca.png, kl_per_dim.png, truncation.png/.json  (checkpoint mode)
    eval/results.csv                     one appended row per run

Usage:
    python eval_pipeline.py -r saved/models/DrumOneShots/checkpoint-epoch1500.pth \
        --data_dir dataset/DrumOneShots --name prototype_22k
    python eval_pipeline.py --audio_dir eval/dsp/audio --data_dir ... --name dsp
    python eval_pipeline.py --fad_sanity --data_dir ... --fadtk ~/envs/fad/bin/fadtk
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import shutil
import subprocess
import time
from collections import defaultdict

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import torchaudio

import model.model as module_arch
from dataset.datasets import OneShotAudioDataset
from drum_types import classify
from evaluate_raw import get_device, plot_latent_pca, plot_kl_per_dim
from evaluate_drums import plot_recon_spectrograms

RESULT_FIELDS = ['name', 'mode', 'checkpoint', 'epoch', 'native_sr', 'eval_sr', 'eval_len',
                 'split', 'n', 'test_clean', 'stft', 'stft_native', 'lat_med', 'lat_within2',
                 'lat_bias', 'fad_model', 'fad', 'fad_inf_r2', 'rtf_cpu', 'decode_ms_cpu',
                 'pca_hit_95', 'pca_frame_95', 'kl_dims', 'trunc_k90', 'kl_total',
                 'gen_ms_per_hit', 'gen_device', 'timestamp']


# ---------------------------------------------------------------------------
# test set + audio I/O
# ---------------------------------------------------------------------------

def flat_name(rel):
    """Test-file relpath -> flat wav name (fadtk wants a flat dir of audio)."""
    return os.path.splitext(rel)[0].replace('/', '__') + '.wav'


def read_split(split_file):
    with open(split_file) as f:
        return [line.rstrip('\n') for line in f if line.strip()]


def make_dataset(data_dir, rel_files, sr, chunk_size, onset_thresh=0.02, pre_pad=256,
                 peak_norm=True):
    """OneShotAudioDataset over exactly `rel_files`, so the system input and the
    reference go through the training-time preprocessing (attack alignment,
    peak norm, pad/crop). Labels are drum types."""
    ds = OneShotAudioDataset(data_dir, sr=sr, chunk_size=chunk_size,
                             onset_thresh=onset_thresh, pre_pad=pre_pad, peak_norm=peak_norm)
    ds.files = [os.path.join(data_dir, r) for r in rel_files]
    missing = [f for f in ds.files if not os.path.isfile(f)]
    if missing:
        raise FileNotFoundError(f'{len(missing)} test files missing under {data_dir}, '
                                f'e.g. {missing[0]}')
    ds.labels = [classify(f, root=data_dir) for f in ds.files]
    return ds


def to_eval(w, sr, eval_sr, eval_len):
    """(T,) tensor at sr -> (eval_len,) at eval_sr: crop to the eval duration
    first (cheaper resampling), resample, then pad/crop exactly."""
    need = int(math.ceil(eval_len * sr / eval_sr))
    w = w[:need]
    if sr != eval_sr:
        # librosa "kaiser_best" settings: the default filter leaks images above
        # the old Nyquist that would be scored against lower-rate systems
        w = torchaudio.functional.resample(
            w.unsqueeze(0), sr, eval_sr, lowpass_filter_width=64, rolloff=0.9475,
            resampling_method='sinc_interp_kaiser', beta=14.769656459379492).squeeze(0)
    if w.shape[0] < eval_len:
        w = torch.nn.functional.pad(w, (0, eval_len - w.shape[0]))
    return w[:eval_len]


def write_wav(path, w, sr):
    sf.write(path, np.clip(w, -1.0, 1.0), sr, subtype='PCM_16')


def prepare_gen_dir(src, dst, eval_sr, eval_len):
    """Every audio file under src -> dst/<basename>.wav through the reference's
    preprocessing (mono, onset alignment, peak norm, pad/crop at eval_sr)."""
    ds = OneShotAudioDataset(src, sr=eval_sr, chunk_size=eval_len,
                             pre_pad=int(round(256 / 22050 * eval_sr)))
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for i, f in enumerate(ds.files):
        name = os.path.splitext(os.path.basename(f))[0] + '.wav'
        write_wav(os.path.join(dst, name), ds[i][2].squeeze(0).numpy(), eval_sr)
    return len(ds.files)


def build_reference(data_dir, rel_files, ref_dir, eval_sr, eval_len):
    """Reference audio at the eval rate, written once and shared by all systems."""
    ds = make_dataset(data_dir, rel_files, eval_sr, eval_len,
                      pre_pad=int(round(256 / 22050 * eval_sr)))   # ~11.6 ms, as in training
    names = [flat_name(r) for r in rel_files]
    if not (os.path.isdir(ref_dir) and
            all(os.path.isfile(os.path.join(ref_dir, n)) for n in names)):
        os.makedirs(ref_dir, exist_ok=True)
        print(f'Writing {len(names)} reference wavs -> {ref_dir}/')
        for i, n in enumerate(names):
            write_wav(os.path.join(ref_dir, n), ds[i][2].squeeze(0).numpy(), eval_sr)
    refs = torch.stack([torch.from_numpy(sf.read(os.path.join(ref_dir, n), dtype='float32')[0])
                        for n in names])
    return refs, ds.labels


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def log_attack_time(w, sr, lo=0.2, hi=0.9, smooth_ms=1.0):
    """log10 of the time the amplitude envelope takes to rise from lo to hi of
    its peak (Peeters et al. 2011 style; floored at one sample)."""
    env = np.abs(scipy.signal.hilbert(w))
    k = max(1, int(sr * smooth_ms / 1000))
    env = np.convolve(env, np.ones(k) / k, mode='same')
    pk_i = int(env.argmax())
    pk = env[pk_i]
    if pk <= 1e-8:
        return float('nan')
    seg = env[:pk_i + 1]
    t0 = int(np.argmax(seg >= lo * pk))
    t1 = int(np.argmax(seg >= hi * pk))
    return float(np.log10(max(t1 - t0, 1) / sr))


def stft_err(x, y, floor_db=-80.0, fft_sizes=(2048, 1024, 512, 256)):
    """Multi-resolution STFT distance: the training loss (spectral convergence +
    log-magnitude L1, same resolutions as model/loss.py) with both magnitudes
    floored at floor_db re a full-scale sine. Unfloored, the log term is
    dominated by comparing a model's ~-90 dB noise tail with the exact zeros
    of the padded reference, and a system whose output was quantised to 16 bit
    would score 'perfect' there; neither is audible."""
    x, y = x.float().view(1, -1), y.float().view(1, -1)
    total = 0.0
    for n_fft in fft_sizes:
        win = torch.hann_window(n_fft)
        floor = 10 ** (floor_db / 20) * win.sum() / 2
        X = torch.stft(x, n_fft, n_fft // 4, n_fft, win, return_complex=True).abs().clamp_min(floor)
        Y = torch.stft(y, n_fft, n_fft // 4, n_fft, win, return_complex=True).abs().clamp_min(floor)
        sc = (X - Y).norm() / X.norm()
        total += float(sc + (X.log() - Y.log()).abs().mean())
    return total / len(fft_sizes)


def paired_metrics(refs, outs, eval_sr):
    rows = []
    for i in range(refs.shape[0]):
        lat_r = log_attack_time(refs[i].numpy(), eval_sr)
        lat_s = log_attack_time(outs[i].numpy(), eval_sr)
        rows.append({'stft': stft_err(refs[i], outs[i]),
                     'lat_ref': lat_r, 'lat_sys': lat_s, 'lat_signed': lat_s - lat_r})
    return rows


def pca_dims(m, var_frac=0.95):
    """Principal components of the rows of m needed for var_frac of the variance."""
    s = torch.linalg.svdvals(m - m.mean(0))
    ratio = (s ** 2).cumsum(0) / (s ** 2).sum()
    return int((ratio < var_frac).sum()) + 1


def latent_stats(mus, logvars, kl_thresh):
    """Latent-usage summary from posteriors (lists of (B, D, T) batches).

    Free bits leave unused dims parked just under the floor rather than at 0,
    so a KL count is only meaningful with the threshold clearly above it; the
    PCA counts need no KL threshold (but 95% is still a choice)."""
    mu = torch.cat(mus); lv = torch.cat(logvars)
    D = mu.shape[1]
    kl_dim = (0.5 * (mu.pow(2) + lv.exp() - 1 - lv)).mean((0, 2))     # (D,)
    mu_f = mu.transpose(1, 2).reshape(-1, D)
    return {'pca_hit_95': pca_dims(mu.mean(2)), 'pca_frame_95': pca_dims(mu_f),
            'kl_dims': int((kl_dim > kl_thresh).sum()), 'kl_total': float(kl_dim.sum())}


def truncation_curve(model, mus, ins_native, n_native, device, err_full,
                     plot_ks=(1, 2, 4, 8, 16, 24), gain=0.9):
    """Reconstruction error when the latent is projected onto its top-k
    per-frame principal components (the rest held at the mean), as in RAVE's
    'fidelity' cropping but measured through the decoder.

    Returns ({k: err}, k90): k=0 is no information (every frame at the mean),
    k=D is the full latent; k90 is the fewest PCs recovering `gain` of the
    error reduction from k=0 to k=D, found by bisection (the curve is
    monotone in practice), so it costs ~log2(D) extra passes."""
    D = mus[0].shape[1]
    mu_f = torch.cat([m.transpose(1, 2).reshape(-1, D) for m in mus])
    mean = mu_f.mean(0)
    V = torch.linalg.svd(mu_f - mean, full_matrices=False)[2].T          # (D, D)
    curve = {D: err_full}

    def err(k):
        if k in curve:
            return curve[k]
        P = V[:, :k] @ V[:, :k].T
        e, i = [], 0
        with torch.no_grad():
            for m in mus:
                z = ((m.transpose(1, 2) - mean) @ P + mean).transpose(1, 2)
                y = model.decode(z.contiguous().to(device)).cpu()
                for b in range(y.shape[0]):
                    e.append(stft_err(ins_native[i], y[b, 0, :n_native])); i += 1
        curve[k] = float(np.mean(e))
        return curve[k]

    for k in plot_ks:
        if k < D:
            err(k)
    target = err(0) - gain * (err(0) - err_full)
    lo, hi = 0, D                       # err(lo) > target >= err(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if err(mid) <= target:
            hi = mid
        else:
            lo = mid
    return dict(sorted(curve.items())), hi


def plot_truncation(curve, k90, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ks = list(curve); es = [curve[k] for k in ks]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(ks, es, 'o-', color='steelblue')
    ax.axvline(k90, color='tomato', ls='--', lw=1, label=f'90% of gain at k={k90}')
    ax.set_xlabel('principal components kept (k)'); ax.set_ylabel('STFT error (native rate)')
    ax.set_title('Decoding from the top-k latent PCs'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
    print(f'Saved  {out_path}')


def time_decode(model, z, n_warmup=2, n_runs=10):
    """Median seconds for one batch-1 decode on CPU."""
    m = copy.deepcopy(model).cpu().eval()
    z = z.cpu()
    with torch.no_grad():
        for _ in range(n_warmup):
            m.decode(z)
        ts = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            m.decode(z)
            ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# ---------------------------------------------------------------------------
# systems
# ---------------------------------------------------------------------------

def load_checkpoint(resume, config_path=None):
    config_path = config_path or os.path.join(os.path.dirname(resume), 'config.json')
    ckpt = torch.load(resume, map_location='cpu', weights_only=False)
    if os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
    else:
        cfg = ckpt['config'].config            # ConfigParser pickled in the checkpoint
    model = getattr(module_arch, cfg['arch']['type'])(**cfg['arch']['args'])
    model.load_state_dict(ckpt['state_dict'])
    return model.eval(), cfg, ckpt.get('epoch')


def run_checkpoint(model, cfg, data_dir, rel_files, eval_sr, eval_len, device, batch_size,
                   kl_thresh=0.2, truncation=True):
    dl = cfg['data_loader']['args']
    sr, chunk = dl.get('sr', 22050), dl.get('chunk_size', 110080)
    ds = make_dataset(data_dir, rel_files, sr, chunk, onset_thresh=dl.get('onset_thresh', 0.02),
                      pre_pad=dl.get('pre_pad', 256), peak_norm=dl.get('peak_norm', True))
    model = model.to(device)
    outs, outs_native, ins_native, mus, logvars = [], [], [], [], []
    n_native = int(math.ceil(eval_len * sr / eval_sr))
    with torch.no_grad():
        for s in range(0, len(ds), batch_size):
            x = torch.stack([ds[i][2] for i in range(s, min(s + batch_size, len(ds)))])
            mu, logvar, _ = model.encode(x.to(device))
            y = model.decode(mu).cpu()             # posterior mean: deterministic recon
            mus.append(mu.cpu()); logvars.append(logvar.cpu())
            for b in range(y.shape[0]):
                outs.append(to_eval(y[b, 0], sr, eval_sr, eval_len))
                outs_native.append(y[b, 0, :n_native])
                ins_native.append(x[b, 0, :n_native])
            print(f'  {min(s + batch_size, len(ds))}/{len(ds)}', end='\r')
    print()
    stft_native = [stft_err(a, b) for a, b in zip(ins_native, outs_native)]
    extra = {'native_sr': sr, 'stft_native': float(np.mean(stft_native)),
             'test_clean': bool(dl.get('fixed_split', False)),
             **latent_stats(mus, logvars, kl_thresh)}
    curve = None
    if truncation:
        print('Truncation curve…')
        curve, extra['trunc_k90'] = truncation_curve(
            model, mus, ins_native, n_native, device, extra['stft_native'])
    dec_s = time_decode(model, mus[0][:1])
    extra.update(rtf_cpu=dec_s / (chunk / sr), decode_ms_cpu=dec_s * 1000)
    # per-frame posteriors (N*T, D) for the KL plot: pooling over time first
    # averages away most of a temporal latent's variation
    mu_f = torch.cat([m.transpose(1, 2).reshape(-1, m.shape[1]) for m in mus])
    lv_f = torch.cat([lv.transpose(1, 2).reshape(-1, lv.shape[1]) for lv in logvars])
    pooled_mu = torch.cat([m.mean(2) for m in mus])
    return torch.stack(outs), stft_native, extra, (pooled_mu, mu_f, lv_f, curve)


def run_audio_dir(audio_dir, rel_files, eval_sr, eval_len):
    outs, missing = [], []
    for r in rel_files:
        p = os.path.join(audio_dir, flat_name(r))
        if not os.path.isfile(p):
            missing.append(p); continue
        w, sr = sf.read(p, dtype='float32', always_2d=True)
        outs.append(to_eval(torch.from_numpy(w.mean(1)), sr, eval_sr, eval_len))
    if missing:
        raise FileNotFoundError(f'{len(missing)} of {len(rel_files)} system files missing, '
                                f'e.g. {missing[0]}')
    return torch.stack(outs)


# ---------------------------------------------------------------------------
# FAD (fadtk CLI)
# ---------------------------------------------------------------------------

def run_fad(fadtk, fad_model, ref_dir, sys_dir, csv_path, inf=False, sox_path=None):
    """FAD(ref_dir, sys_dir) with fadtk; returns (score, inf_r2) from its csv."""
    # fadtk caches embeddings in subfolders keyed by file name; a re-rendered
    # system dir must not reuse embeddings of the previous audio.
    for d in os.listdir(sys_dir):
        if os.path.isdir(os.path.join(sys_dir, d)):
            shutil.rmtree(os.path.join(sys_dir, d))
    cmd = [fadtk, fad_model, ref_dir, sys_dir, csv_path]
    if inf:
        cmd.append('--inf')
    if sox_path:
        cmd += ['-s', sox_path]
    print('$ ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)
    with open(csv_path) as f:
        last = list(csv.DictReader(f))[-1]
    r2 = last.get('inf_r2')
    return float(last['score']), (None if r2 in (None, '', 'None') else float(r2))


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def lat_summary(rows):
    d = np.array([r['lat_signed'] for r in rows]); d = d[~np.isnan(d)]
    return {'lat_med': float(np.median(np.abs(d))),
            'lat_within2': float((np.abs(d) < np.log10(2)).mean()),
            'lat_bias': float(np.median(d))}


def per_type_table(labels, rows, out_path):
    by = defaultdict(list)
    for l, r in zip(labels, rows):
        by[l].append(r)
    lines = [f'{"type":<10}{"n":>5}{"stft":>10}{"lat_med":>10}{"within2":>9}{"lat_bias":>10}']
    for t in sorted(by) + ['OVERALL']:
        rs = rows if t == 'OVERALL' else by[t]
        if t == 'OVERALL':
            lines.append('-' * 54)
        L = lat_summary(rs)
        lines.append(f'{t:<10}{len(rs):>5}{np.mean([r["stft"] for r in rs]):>10.4f}'
                     f'{L["lat_med"]:>10.3f}{L["lat_within2"]:>9.0%}{L["lat_bias"]:>10.3f}')
    text = '\n'.join(lines)
    with open(out_path, 'w') as f:
        f.write(text + '\n')
    print('\n' + text + '\n')


def append_result(path, row):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if os.path.isfile(path):                    # columns added since: rewrite the header
        with open(path) as f:
            old = list(csv.DictReader(f))
        with open(path) as f:
            header = f.readline().strip().split(',')
        if header != RESULT_FIELDS:
            with open(path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
                w.writeheader()
                w.writerows({k: r.get(k, '') for k in RESULT_FIELDS} for r in old)
    new = not os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in RESULT_FIELDS})
    print(f'Appended row "{row["name"]}" -> {path}')


def update_result(path, name, fields):
    """Fill `fields` into the latest row called `name` (used by --fad_only)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    hits = [r for r in rows if r['name'] == name]
    if not hits:
        raise KeyError(f'no row "{name}" in {path}; score the system before --fad_only')
    hits[-1].update({k: v for k, v in fields.items() if v is not None})
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k, '') for k in RESULT_FIELDS} for r in rows)
    print(f'Updated row "{name}" in {path}: ' + ', '.join(f'{k}={fmt(v)}' for k, v in fields.items()))


def fmt(v):
    return f'{v:.4g}' if isinstance(v, float) else v


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args):
    rel_files = read_split(args.split_file)
    if args.max_items:
        rel_files = rel_files[:args.max_items]
    split = os.path.splitext(os.path.basename(args.split_file))[0]
    ref_dir = args.ref_dir or os.path.join(
        'eval', 'reference', f'{split}_{args.eval_sr}_{args.eval_len}'
        + (f'_first{args.max_items}' if args.max_items else ''))
    refs, labels = build_reference(args.data_dir, rel_files, ref_dir, args.eval_sr, args.eval_len)
    print(f'{len(rel_files)} test hits, reference at {args.eval_sr} Hz x {args.eval_len}')

    if args.fad_sanity:
        val_rel = read_split(os.path.join(os.path.dirname(args.split_file), 'val.txt'))
        val_dir = ref_dir.replace(f'{split}_', 'val_', 1)
        build_reference(args.data_dir, val_rel, val_dir, args.eval_sr, args.eval_len)
        fad, r2 = run_fad(args.fadtk, args.fad_model, ref_dir, val_dir,
                          os.path.join(os.path.dirname(args.results), 'fad_raw.csv'),
                          args.fad_inf, args.sox_path)
        append_result(args.results, {'name': 'real_val_vs_test', 'mode': 'sanity',
                                     'eval_sr': args.eval_sr, 'eval_len': args.eval_len,
                                     'split': split, 'n': len(rel_files), 'fad_model': args.fad_model,
                                     'fad': fad, 'fad_inf_r2': r2,
                                     'timestamp': time.strftime('%Y-%m-%d %H:%M')})
        return

    if args.gen_dir:
        out_dir = args.out_dir or os.path.join('eval', 'gen', args.name)
        audio_dir = os.path.join(out_dir, 'audio')
        n = prepare_gen_dir(args.gen_dir, audio_dir, args.eval_sr, args.eval_len)
        print(f'{n} generated hits preprocessed -> {audio_dir}')
        fad, r2 = run_fad(args.fadtk, args.fad_model, ref_dir, audio_dir,
                          os.path.join(os.path.dirname(args.results), 'fad_raw.csv'),
                          args.fad_inf, args.sox_path)
        row = {'name': args.name, 'mode': 'generation', 'eval_sr': args.eval_sr,
               'eval_len': args.eval_len, 'split': split, 'n': n, 'fad_model': args.fad_model,
               'fad': fad, 'fad_inf_r2': r2, 'timestamp': time.strftime('%Y-%m-%d %H:%M')}
        tj = os.path.join(args.gen_dir, 'timing.json')
        if os.path.isfile(tj):
            with open(tj) as f:
                t = json.load(f)
            sec = t.get('decode_sec_per_hit_median', t.get('sec_per_hit_batch1_median'))
            row.update(gen_ms_per_hit=sec * 1000 if sec else '', gen_device=t.get('device', ''),
                       checkpoint=t.get('checkpoint', t.get('model', '')), epoch=t.get('epoch', ''))
        append_result(args.results, row)
        print(f'FAD {fad:.4f}  (n={n})')
        return

    out_dir = args.out_dir or os.path.join('eval', 'milestone', args.name)
    audio_dir = os.path.join(out_dir, 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    row = {'name': args.name, 'eval_sr': args.eval_sr, 'eval_len': args.eval_len,
           'split': split, 'n': len(rel_files), 'timestamp': time.strftime('%Y-%m-%d %H:%M')}

    if args.fad_only:
        fad, r2 = run_fad(args.fadtk, args.fad_model, ref_dir, audio_dir,
                          os.path.join(os.path.dirname(args.results), 'fad_raw.csv'),
                          args.fad_inf, args.sox_path)
        update_result(args.results, args.name,
                      {'fad': fad, 'fad_inf_r2': r2, 'fad_model': args.fad_model})
        return

    if args.resume:
        model, cfg, epoch = load_checkpoint(args.resume, args.config)
        device = torch.device(args.device) if args.device else get_device()
        print(f'Checkpoint {args.resume} (epoch {epoch}) on {device}')
        outs, stft_native, extra, (pmu, mu_f, lv_f, curve) = run_checkpoint(
            model, cfg, args.data_dir, rel_files, args.eval_sr, args.eval_len, device,
            args.batch_size, args.kl_thresh, not args.no_truncation)
        if curve:
            plot_truncation(curve, extra['trunc_k90'], os.path.join(out_dir, 'truncation.png'))
            with open(os.path.join(out_dir, 'truncation.json'), 'w') as f:
                json.dump({'curve': curve, 'k90': extra['trunc_k90']}, f, indent=1)
        row.update(extra, mode='checkpoint', checkpoint=args.resume, epoch=epoch)
        if not extra['test_clean']:
            print('WARNING: this checkpoint was not trained with fixed_split; most test '
                  'hits were probably in its training set, so its scores are optimistic.')
        plot_latent_pca(pmu, labels, os.path.join(out_dir, 'latent_pca.png'))
        plot_kl_per_dim(mu_f, lv_f, os.path.join(out_dir, 'kl_per_dim.png'),
                        free_bits=cfg.get('trainer', {}).get('free_bits'))
    else:
        outs = run_audio_dir(args.audio_dir, rel_files, args.eval_sr, args.eval_len)
        row['mode'] = 'audio_dir'
        stft_native = None

    for r, w in zip(rel_files, outs):
        write_wav(os.path.join(audio_dir, flat_name(r)), w.numpy(), args.eval_sr)

    rows = paired_metrics(refs, outs, args.eval_sr)
    row['stft'] = float(np.mean([r['stft'] for r in rows]))
    row.update(lat_summary(rows))
    with open(os.path.join(out_dir, 'per_hit.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'type', 'stft', 'stft_native', 'lat_ref', 'lat_sys'])
        for i, (rel, r) in enumerate(zip(rel_files, rows)):
            w.writerow([rel, labels[i], f'{r["stft"]:.5f}',
                        f'{stft_native[i]:.5f}' if stft_native else '',
                        f'{r["lat_ref"]:.4f}', f'{r["lat_sys"]:.4f}'])
    per_type_table(labels, rows, os.path.join(out_dir, 'per_type.txt'))

    rng = random.Random(0)
    picks = []
    for t in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == t]
        picks.append(rng.choice(idx))
    plot_recon_spectrograms(refs, outs, labels, [os.path.basename(r) for r in rel_files],
                            picks[:args.n_plot], os.path.join(out_dir, 'recon_spectrograms.png'),
                            n_fft=2048, hop=512)

    if args.fadtk:
        row['fad'], row['fad_inf_r2'] = run_fad(
            args.fadtk, args.fad_model, ref_dir, audio_dir,
            os.path.join(os.path.dirname(args.results), 'fad_raw.csv'),
            args.fad_inf, args.sox_path)
        row['fad_model'] = args.fad_model
    else:
        print('FAD skipped (no --fadtk); add it later with --fad_only.')

    append_result(args.results, row)
    print('  '.join(f'{k}={fmt(row[k])}' for k in
                    ['stft', 'stft_native', 'lat_med', 'lat_within2', 'fad', 'rtf_cpu',
                     'pca_hit_95', 'pca_frame_95', 'kl_dims', 'trunc_k90']
                    if row.get(k) not in (None, '')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Score one system on the fixed one-shot test split')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('-r', '--resume', help='RawAudioVAE checkpoint .pth')
    src.add_argument('--audio_dir', help='dir of system wavs named flat_name(relpath)')
    src.add_argument('--fad_only', action='store_true',
                     help='(re)compute only FAD for an existing <out_dir>/audio and '
                          'fill it into that run\'s row in --results')
    src.add_argument('--gen_dir', help='unpaired generated set to score with FAD (see docstring)')
    src.add_argument('--fad_sanity', action='store_true',
                     help='FAD between real test and real val audio, then exit')
    p.add_argument('-c', '--config', help='config json (default: beside the checkpoint)')
    p.add_argument('--data_dir', required=True, help='dataset root the split paths are relative to')
    p.add_argument('--split_file', default='splits/test.txt')
    p.add_argument('--name', help='row name in results.csv (required except for --fad_sanity)')
    p.add_argument('-o', '--out_dir', help='default eval/milestone/<name>')
    p.add_argument('--results', default='eval/results.csv')
    p.add_argument('--ref_dir', help='default eval/reference/<split>_<sr>_<len>')
    p.add_argument('--eval_sr', type=int, default=44100)
    p.add_argument('--eval_len', type=int, default=65536)
    p.add_argument('--max_items', type=int, default=0, help='first N test files only (smoke test)')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('-d', '--device')
    p.add_argument('--n_plot', type=int, default=7)
    p.add_argument('--kl_thresh', type=float, default=0.2,
                   help='nats; kl_dims counts dims above this (keep it above free_bits)')
    p.add_argument('--no_truncation', action='store_true',
                   help='skip the truncation curve (~10 extra decode passes)')
    p.add_argument('--fadtk', help='path to the fadtk executable (separate env)')
    p.add_argument('--fad_model', default='clap-laion-audio')
    p.add_argument('--fad_inf', action='store_true', help='also extrapolate FAD-inf')
    p.add_argument('--sox_path', help='passed to fadtk -s (default /usr/bin/sox)')
    args = p.parse_args()
    if not args.fad_sanity and not args.name:
        p.error('--name is required')
    if (args.fad_sanity or args.fad_only or args.gen_dir) and not args.fadtk:
        p.error('--fadtk is required for FAD modes')
    main(args)
