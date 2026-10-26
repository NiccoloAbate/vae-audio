import hashlib
import os
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from pathlib import Path
from torch.utils.data import Dataset


class CollectData(Dataset):
    def __init__(self, path_to_dataset, extension=['wav', 'mp3', 'npy', 'pth'], subset=None, transform=None):
        """
        Assume directory structure as:
        - dataset (the level which path_to_dataset indicates)
            - trainingdata
                - class A
                - class B
                ...
            - testdata
                - class A
                - class B
                ...
        :param path_to_dataset: a list of path to dataset directory; thus allows multiple datasets
        :param subset: None|'train'|'test'; if None, both train and test sets are loaded
        :param transform: see https://pytorch.org/docs/stable/torchvision/transforms.html
        """
        assert isinstance(path_to_dataset, list), "The input path_to_dataset should be a list."
        assert subset in [None, 'train', 'test'], "subset should be in [None, 'train', 'test']"

        full_path_to_dataset = []
        for data_dir in path_to_dataset:
            if not subset:  # load both train and test folders
                full_path_to_dataset.append(os.path.join(data_dir, 'trainingdata'))
                full_path_to_dataset.append(os.path.join(data_dir, 'testdata'))
            elif subset == 'train':
                full_path_to_dataset.append(os.path.join(data_dir, 'trainingdata'))
            else:
                full_path_to_dataset.append(os.path.join(data_dir, 'testdata'))

        aggr_data_path = []
        aggr_label = []
        for data_dir in full_path_to_dataset:
            for subclass in os.listdir(data_dir):
                for f in os.listdir(os.path.join(data_dir, subclass)):
                    if any(ext in f for ext in extension):
                        aggr_data_path.append(os.path.join(data_dir, subclass, f))
                        aggr_label.append(subclass)

        self.path_to_dataset = path_to_dataset
        self.extension = extension
        self.subset = subset
        self.transform = transform
        self.path_to_data = aggr_data_path
        self.labels = aggr_label

    def __len__(self):
        return len(self.path_to_data)

    def __getitem__(self, idx):
        if self.transform:
            return idx, self.labels[idx], self.transform(self.path_to_data[idx])

        return idx, self.labels[idx], self.path_to_data[idx]


class NsynthSubset(Dataset):
    """
    This Dataset is used to load *the filtered Nsynth (by nsynth_subset.py)*.
    In the filtered subset, file name follows the format:

        instrumentFamily_source_instrumentID-pitch-velocity

    which is true only when synthesizers are not included (a different format).
    """
    def __init__(self, path_to_dataset, file_ext='.wav', transform=None):
        path_to_data = Path(path_to_dataset)
        file_ext = '*' + file_ext
        y = []
        for f in path_to_data.glob(file_ext):
            f_name = f.stem
            instrument, pitch, vel = f_name.split('-')[0], int(f_name.split('-')[1]), int(f_name.split('-')[2])
            instrument_prop = instrument.split('_')
            assert len(instrument_prop) == 3
            assert instrument_prop[1] == 'acoustic'
            family, inst_id = instrument_prop[0], instrument_prop[2]
            y.append({'family': family, 'instrument': inst_id, 'pitch': pitch, 'velocity': vel})

        self.file_ext = file_ext
        self.path_to_data = list(path_to_data.glob(file_ext))
        self.labels = y
        self.transform = transform

        if len(self.path_to_data) == 0:
            raise Warning("The size of path_to_data is zero with the specified extension %s" % file_ext)

    def __len__(self):
        return len(self.path_to_data)

    def __getitem__(self, idx):
        if self.transform:
            return idx, self.labels[idx], self.transform(str(self.path_to_data[idx]))

        return idx, self.labels[idx], str(self.path_to_data[idx])


class RawAudioDataset(Dataset):
    """
    Loads raw audio wav files and serves fixed-length chunks.
    Expects the same directory structure as CollectData:
        data_dir/
            trainingdata/{class}/file.wav
            testdata/{class}/file.wav
    """
    def __init__(self, path_to_dataset, sr=22050, chunk_size=16384,
                 subset='train', **kwargs):
        assert isinstance(path_to_dataset, list)
        assert subset in [None, 'train', 'test']

        split_dirs = []
        for data_dir in path_to_dataset:
            if subset == 'train':
                split_dirs.append(os.path.join(data_dir, 'trainingdata'))
            elif subset == 'test':
                split_dirs.append(os.path.join(data_dir, 'testdata'))
            else:
                split_dirs.append(os.path.join(data_dir, 'trainingdata'))
                split_dirs.append(os.path.join(data_dir, 'testdata'))

        items = []   # (label, filepath, chunk_start_sample_at_target_sr, file_sr)
        for split_dir in split_dirs:
            if not os.path.isdir(split_dir):
                continue
            for label in sorted(os.listdir(split_dir)):
                label_dir = os.path.join(split_dir, label)
                if not os.path.isdir(label_dir):
                    continue
                for fname in sorted(os.listdir(label_dir)):
                    if not fname.lower().endswith('.wav'):
                        continue
                    fpath = os.path.join(label_dir, fname)
                    try:
                        info = sf.info(fpath)
                        n_samples = int(info.frames * sr / info.samplerate)
                        n_chunks  = n_samples // chunk_size
                        for c in range(n_chunks):
                            items.append((label, fpath, c * chunk_size, info.samplerate))
                    except Exception:
                        continue

        self.items      = items
        self.sr         = sr
        self.chunk_size = chunk_size
        self.path_to_data = [it[1] for it in items]
        self.labels       = [it[0] for it in items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        label, path, start, file_sr = self.items[idx]

        if file_sr == self.sr:
            # Load only the needed chunk — avoids allocating the entire file.
            y, _ = torchaudio.load(path, frame_offset=start, num_frames=self.chunk_size)
        else:
            # Resampling required: load full file, resample, then slice.
            y, _ = torchaudio.load(path)
            y = torchaudio.functional.resample(y, file_sr, self.sr)
            end = start + self.chunk_size
            if end > y.shape[1]:
                y = torch.nn.functional.pad(y, (0, end - y.shape[1]))
            y = y[:, start:end]

        if y.shape[0] > 1:
            y = y.mean(0, keepdim=True)
        if y.shape[1] < self.chunk_size:
            y = torch.nn.functional.pad(y, (0, self.chunk_size - y.shape[1]))
        return idx, label, y   # (1, chunk_size)


class NumpyChunkDataset(Dataset):
    """
    Fast dataset backed by a pre-extracted .npy array of shape (N, chunk_size).
    Zero file-seek overhead — ideal when torchaudio seeks are too slow.
    """
    def __init__(self, npy_path, label='cello', **kwargs):
        import numpy as np, os, pickle
        self.data = np.load(npy_path, mmap_mode='r')  # shape (N, chunk_size)
        self.label = label
        self.chunk_size = self.data.shape[1]

        # Load per-chunk metadata (real wav paths) if available
        meta_path = npy_path.replace('.npy', '_meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.items = meta['items']
        else:
            self.items = [(label, npy_path, i * self.chunk_size, 22050)
                          for i in range(len(self.data))]

        self.path_to_data = [it[1] for it in self.items]
        self.labels = [it[0] for it in self.items]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import numpy as np
        chunk = torch.from_numpy(self.data[idx].copy()).unsqueeze(0)  # (1, chunk_size)
        return idx, self.label, chunk


def oneshot_split(relpath, val=0.1, test=0.1):
    """Fixed train/val/test assignment for one file, from a hash of its path.

    `relpath` is relative to the dataset root, with '/' separators. The same
    file always lands in the same split, on any machine and whatever other
    files are added, removed or relabelled; the split is therefore not
    stratified, only approximately 80/10/10 within each class.
    """
    h = int(hashlib.md5(relpath.encode('utf-8')).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test:
        return 'test'
    if h < test + val:
        return 'val'
    return 'train'


class OneShotAudioDataset(Dataset):
    """Serves one fixed-length, attack-aligned chunk per audio file.

    Built for one-shot samples (e.g. drum hits) where each file is a single
    coherent event, usually *shorter* than the target length. Unlike
    RawAudioDataset (which slices long recordings into many interchangeable
    chunks and silently drops any file shorter than chunk_size), this dataset:

        1. recursively collects every audio file under the given dir(s),
        2. trims leading silence so the onset lands near t=0 (attack alignment),
        3. optionally peak-normalizes,
        4. zero-pads (or truncates) to exactly chunk_size.

    Returns (idx, label, waveform) with waveform shape (1, chunk_size) — the
    same contract as RawAudioDataset, so the RawAudioVAE trainers work unchanged.

    Args:
        path_to_dataset: dir or list of dirs (searched recursively).
        sr:              target sample rate (files are resampled to this).
        chunk_size:      output length in samples (must be divisible by the
                         model's total downsample factor, else the STFT recon
                         loss will hit a length mismatch).
        onset_thresh:    onset is the first sample exceeding this fraction of the
                         file peak. Set 0 to disable attack alignment.
        pre_pad:         samples of headroom kept before the detected onset so
                         the very start of the transient is not clipped.
        peak_norm:       peak-normalize each hit to ~0.95. Consistent levels help
                         the tanh/PQMF output, but this discards velocity
                         dynamics — set False if you want to model loudness.
        split:           None (all files) or 'train' / 'val' / 'test', using the
                         fixed hash split in oneshot_split(). Requires a single
                         dataset dir, since paths are taken relative to it.
        types:           keep only these drum types (see drum_types.classify),
                         e.g. ['kick', 'snare', 'clap', 'hat', 'cymbal', 'tom', 'perc'].
                         'exclude' files are always dropped when types is set.
        packs:           keep only files under these top-level folders.
    """
    def __init__(self, path_to_dataset, sr=22050, chunk_size=110080,
                 onset_thresh=0.02, pre_pad=256, peak_norm=True,
                 extensions=('.wav', '.flac', '.aif', '.aiff'),
                 split=None, types=None, packs=None, **kwargs):
        if isinstance(path_to_dataset, str):
            path_to_dataset = [path_to_dataset]

        files = []
        for d in path_to_dataset:
            for root, _, fnames in os.walk(d):
                for fn in sorted(fnames):
                    if fn.lower().endswith(extensions):
                        files.append(os.path.join(root, fn))
        files.sort()
        if len(files) == 0:
            raise RuntimeError(f"No audio files found under {path_to_dataset}")

        if split is not None or types is not None or packs is not None:
            assert len(path_to_dataset) == 1, "split/types/packs need a single dataset dir"
            root = path_to_dataset[0]
            from drum_types import classify
            kept = []
            for f in files:
                rel = os.path.relpath(f, root).replace(os.sep, '/')
                if split is not None and oneshot_split(rel) != split:
                    continue
                if packs is not None and rel.split('/')[0] not in packs:
                    continue
                if types is not None and classify(f, root=root) not in types:
                    continue
                kept.append(f)
            files = kept
            if len(files) == 0:
                raise RuntimeError(f"No files left after split={split} types={types} packs={packs}")
            print(f"OneShotAudioDataset: {len(files)} files (split={split}, types={types}, packs={packs})")

        self.files        = files
        self.sr           = sr
        self.chunk_size   = chunk_size
        self.onset_thresh = onset_thresh
        self.pre_pad      = pre_pad
        self.peak_norm    = peak_norm
        # label = immediate parent dir name (keeps per-class structure if present)
        self.labels       = [os.path.basename(os.path.dirname(f)) for f in files]
        self.path_to_data = files

    def __len__(self):
        return len(self.files)

    def _align_attack(self, y):
        """y: (T,) -> trimmed so the onset sits pre_pad samples from the start."""
        if self.onset_thresh <= 0:
            return y
        mag  = y.abs()
        peak = mag.max()
        if peak <= 0:
            return y
        above = (mag > self.onset_thresh * peak).nonzero()
        if len(above) == 0:
            return y
        onset = int(above[0])
        start = max(0, onset - self.pre_pad)
        return y[start:]

    def __getitem__(self, idx):
        path = self.files[idx]
        y, file_sr = torchaudio.load(path)             # (C, T)
        if y.shape[0] > 1:
            y = y.mean(0, keepdim=True)
        if file_sr != self.sr:
            y = torchaudio.functional.resample(y, file_sr, self.sr)
        y = y.squeeze(0)                               # (T,)

        y = self._align_attack(y)

        if y.shape[0] < self.chunk_size:
            y = F.pad(y, (0, self.chunk_size - y.shape[0]))
        else:
            y = y[:self.chunk_size]

        if self.peak_norm:
            peak = y.abs().max()
            if peak > 1e-6:
                y = y / peak * 0.95

        return idx, self.labels[idx], y.unsqueeze(0)   # (1, chunk_size)


if __name__ == '__main__':
    path_to_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'myAudioDataset/audio')
    d = CollectData([path_to_data], subset=None, transform=None)
    print("the number of data: %d" % len(d))
    try:
        print("the first five entries:")
        for n in range(5):
            print(d[n])
    except:
        raise IndexError("There is none or fewer than 5 data in the input path")
