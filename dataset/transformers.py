import numpy as np
import librosa
import torch


class AudioRead:
    def __init__(self, sr=22050, offset=0.0, duration=None):
        self.sr = sr
        self.offset = offset
        self.duration = duration

    def __call__(self, x):
        y, sr = librosa.load(x, sr=self.sr, duration=self.duration, offset=self.offset)
        return y


class TorchAudioRead:
    """Load audio using torchaudio + resample with torchaudio.
    Required for Vocos compatibility: torchaudio and librosa use different
    resampling algorithms, causing audible mel spectrogram differences.
    """
    def __init__(self, sr=24000, duration=None):
        import torchaudio as _ta
        self.sr = sr
        self.duration = duration

    def __call__(self, x):
        import torch
        import torchaudio
        y, src_sr = torchaudio.load(str(x))
        if src_sr != self.sr:
            y = torchaudio.functional.resample(y, src_sr, self.sr)
        if y.shape[0] > 1:
            y = y.mean(0, keepdim=True)
        if self.duration is not None:
            max_samples = int(self.duration * self.sr)
            y = y[:, :max_samples]
        return y.squeeze(0).numpy()


class Zscore:
    def __init__(self, divide_sigma=False):
        self.divide_sigma = divide_sigma

    def __call__(self, x):
        x -= x.mean()
        if self.divide_sigma:
            x /= x.std()
        return x


class PadAudio:
    def __init__(self, sr=22050, pad_to=30):
        """
        Pad the input audio with zeros.
        If the input is longer than the desired length, trim the audio.
        :param pad_to: the desired length of audio (seceond)
        """
        self.pad_to = pad_to
        self.sr = sr

    def __call__(self, x):
        target_len = int(self.pad_to * self.sr)
        pad_len = abs(len(x) - target_len)
        if len(x) < target_len:  # pad
            x = np.hstack([x, np.zeros(pad_len)])
        elif len(x) > target_len:  # trim
            x = x[:target_len]

        return x


class Spectrogram:
    def __init__(self, sr=22050, n_fft=2048, hop_size=735, n_band=64, spec_type='mel',
                 power=2, safe_log=False):
        """
        Derive spectrogram. Currently accept linear and Mel spectrogram.
        :param n_fft: size of short-time fourier transform
        :param hop_size: short-time window hop size
        :param n_band: number of frequency bins, ignored if spec_type='linear'
        :param power: mel spectrogram exponent (1=amplitude, 2=power)
        :param safe_log: if True, apply log(max(S, 1e-5)) instead of power_to_db.
                         When safe_log=True, uses torchaudio (norm=None) instead of
                         librosa (norm='slaney') to match Vocos's feature extractor exactly.
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.n_band = n_band
        self.spec_type = spec_type
        self.power = power
        self.safe_log = safe_log

        if safe_log and spec_type == 'mel':
            import torch
            import torchaudio
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=n_fft, hop_length=hop_size,
                n_mels=n_band, center=True, power=power
            )

    def __call__(self, x):
        assert self.spec_type in ['lin', 'mel', 'cqt'], "spec_type should be in ['lin', 'mel', 'cqt']"
        if self.spec_type == 'lin':
            S = librosa.core.stft(y=x, n_fft=self.n_fft, hop_length=self.hop_size)
            S = np.abs(S) ** 2  # power spectrogram

        elif self.spec_type == 'mel':
            if self.safe_log:
                import torch
                # Use torchaudio (norm=None) to match Vocos's feature extractor exactly.
                # librosa uses norm='slaney' by default which is incompatible with Vocos.
                y_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, T)
                S = self._mel_transform(y_t).squeeze(0).numpy()          # (n_mels, T)
                S = np.log(np.maximum(S, 1e-5))
            else:
                S = librosa.feature.melspectrogram(y=x, sr=self.sr, n_fft=self.n_fft,
                                                   hop_length=self.hop_size, n_mels=self.n_band,
                                                   power=self.power)
                S = librosa.core.power_to_db(S, ref=np.max)
        else:
            # TODO: implement CQT
            raise NotImplementedError

        return S


class MinMaxNorm:
    def __init__(self, min_val=0, max_val=1):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x):
        x -= x.mean()
        x_min = x.min()
        x_max = x.max()
        nom = x - x_min
        den = x_max - x_min

        if abs(den) > 1e-4:
            return (self.max_val - self.min_val) * (nom / den) + self.min_val
        else:
            return nom


class SpecChunking:
    def __init__(self, duration=0.5, sr=22050, hop_size=735, reverse=False):
        """
        Slice spectrogram into non-overlapping chunks. Discard chunks shorter than the specified duration.
        :params duration: the duration (in sec.) of each spectrogram chunk
        :params sr: sampling frequency used to read waveform; used to calculate the size of each spectrogram chunk
        :parms hop_size: hop size used to derive spectrogram; used to calculate the size of each spectrogram chunk
        :params reverse: reverse the spectrogram before chunking;
                         set True if the end is more important than the begin of spectrogram
        TODO:
            [] Allow an input argument to indicate the overlapping amount between chunks
        """
        self.duration = duration
        self.sr = sr
        self.hop_size = hop_size
        self.chunk_size = int(sr * duration) // hop_size
        self.reverse = reverse

    def __call__(self, x):
        time_dim = 1  # assume input spectrogram with shape n_freqBand * n_contextWin
        n_contextWin = x.shape[time_dim]  # context window size of the input spectrogram
        # TODO: overlapping window size; with the amount of overlap as an input argument
        indices = np.arange(self.chunk_size, n_contextWin, self.chunk_size)  # currently non-overlapping chunking

        # reverse to keep the end content of spectrogram intact in the later discard
        # this is only used when the end is more important than the begin of spectrogram
        if self.reverse:
            x = np.flip(x, time_dim)

        x_chunk = np.split(x, indices_or_sections=indices, axis=time_dim)

        # reverse back if self.reverse=True
        if self.reverse:
            x_chunk = [np.flip(i, time_dim) for i in x_chunk[::-1]]

        # discard those short chunks
        x_chunk = [x_i for x_i in x_chunk if x_i.shape[time_dim] == self.chunk_size]

        return np.array(x_chunk)


class LoadNumpyAry:
    def __call__(self, x):
        return np.load(x)


class NormalizeSpecDb:
    """
    Map dB mel spectrograms from (-80, 0) to (-1, 1) using the fixed linear transform
    Works for any number of mel bins — the dB range from power_to_db is always (-80, 0).
        x_norm = x / 40 + 1
    This matches the Tanh output range of the decoder.
    Inverse: x_db = (x_norm - 1) * 40
    """
    def __call__(self, x):
        return x / 40.0 + 1.0


class SafeLogNorm:
    """
    Map safe_log amplitude mel spectrograms to (-1, 1) using a fixed linear transform.
    safe_log values (log of amplitude mel with Vocos params) span roughly (-11, 5).
    We use the fixed map:  x_norm = (x + 3) / 8
    which maps (-11, 5) → (-1, 1).
    Inverse: x_safelog = x_norm * 8 - 3
    """
    def __call__(self, x):
        return (x + 3.0) / 8.0
