import os
from functools import lru_cache
from typing import Union

import ffmpeg
import numpy as np
import torch
import torch.nn.functional as F

try:
    from .utils import exact_div
except ImportError:
    from whisper.utils import exact_div

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
STREAM_CHUNK_LENGTH = 1
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: number of samples in a chunk of 30 seconds
STREAM_N_SAMPLES = STREAM_CHUNK_LENGTH * SAMPLE_RATE  # 1600: number of samples in a chunk of .1 seconds
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000: number of frames in a mel spectrogram input


class StreamAudio:
    """ Open an audio file and yields chunks as mono waveform, resampling as necessary
    to max_duration. """

    def __init__(
        self,
        file: Union[str, bytes, np.ndarray],
        sample_rate: int = SAMPLE_RATE,
        stream_chunk: int = STREAM_CHUNK_LENGTH,
        max_duration: int = CHUNK_LENGTH
    ):
        """ Init the generator.
        
        Parameters
        ----------
        file: str
            The audio file to open or read (str or bytes or np array)

        sample_rate: int
            The sample rate to resample the audio if necessary

        stream_chunk: int
            The duration of a chunk for streaming recognition

        max_len: int
            The maximum number of samples to yield

        """

        self._file = file
        self._sample_rate = sample_rate
        # self._stream_chunk = stream_chunk
        # self._max_duration = max_duration
        self._stream_n_samples = stream_chunk * sample_rate
        self._max_samples = max_duration * sample_rate
        self.closed = True

    def __enter__(self):

        """
        Enter the generator.
        """
        
        if isinstance(self._file, str):
            self._audio = load_audio(self._file, sr=self._sample_rate)
        elif isinstance(self._file, bytes):
            self._audio = np.frombuffer(self._file, np.int16).flatten().astype(np.float32) / 32768.0
        else:
            self._audio = self._file

        if not isinstance(self._audio, np.ndarray):
            # if it was str or byte (valid) this wouldn't occur
            raise TypeError("file must be a string or bytes or np array but is"
                            f"{type(self._file)}")

        self.closed = False
        self._audio_stream = np.array([])
        return self
    
    def __exit__(self, type, value, traceback):
        self.closed = True

    def generator(self):
        """Yields
        -------
        A NumPy array containing the audio waveform, in float32 dtype.
        """

        while len(self._audio):
            chunk = self._extract_first_n_elements()
            self._audio_stream = np.concatenate([self._audio_stream, chunk])
            if self._audio_stream.shape[0] > self._max_samples:
                # pad_or_trim would keep the first _max_samples while
                # we want to keep the last ones
                self._audio_stream = self._audio_stream[-self._max_samples:]
            yield self._audio_stream.astype(np.float32)

    def _extract_first_n_elements(
        self,
        delete: bool=True
    ):

        n = self._stream_n_samples
        if not isinstance(n, int):
            raise TypeError(f"n must be an integer but is {type(n)}")
        if not isinstance(self._audio, np.ndarray):
            raise TypeError(f"input should be an np array but is {type(self._audio)}")
        
        new_audio = self._audio[:n]
        if delete is True:
            self._audio = np.delete(self._audio, range(min(n, self._audio.shape[0])))

        return new_audio


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1, trim_start=False):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            if trim_start is False:
                array = array.index_select(dim=axis, index=torch.arange(length))
            else:
                # keep the last N_SAMPLES
                array = array.index_select(dim=axis, index=torch.arange(array.shape[axis] - length, array.shape[axis]))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            if trim_start is False:
                array = array.take(indices=range(length), axis=axis)
            else:
                # keep the last N_SAMPLES
                array = array.take(indices=range(array.shape[axis] - length, array.shape[axis]), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(audio: Union[str, np.ndarray, torch.Tensor], n_mels: int = N_MELS):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[:, :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
