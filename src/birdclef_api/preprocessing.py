"""Audio -> model input, identical to the preprocessing used to train the model.

Training pipeline (see github.com/youssef-khazal/birdclef-2025):
  1. load the recording as mono, resampled to 32 kHz
  2. cut it into consecutive 5 s segments (the last partial one is zero-padded)
  3. 64-band log-Mel spectrogram per segment (n_fft 1024, hop 256), in dB relative to the segment max
  4. saved as a grayscale PNG with matplotlib (min-max scaled to 0-255)
  5. at training time: 3 channels, resized to 224x224 (bilinear), scaled to [-1, 1]

Steps 4-5 are reproduced here in memory, without writing files. `tests/test_preprocessing.py`
checks that the result is pixel-identical to a spectrogram from the training set.
"""

from __future__ import annotations

import io
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from matplotlib import cm, colors
from PIL import Image

SAMPLE_RATE = 32_000
SEGMENT_SECONDS = 5
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
IMG_SIZE = 224

_GRAY = cm.ScalarMappable(cmap="gray")


def load_audio(source: str | Path | bytes, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file (path or raw bytes) as a mono float32 waveform at ``sr`` Hz.

    WAV / FLAC / OGG / MP3 are read with libsndfile; other formats (e.g. M4A from phones)
    fall back to librosa's decoder, which uses ffmpeg when it is installed.
    """
    data = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    try:
        y, file_sr = sf.read(data, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
    except Exception:
        if isinstance(data, io.BytesIO):
            data.seek(0)
        y, file_sr = librosa.load(data, sr=None, mono=True)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return np.asarray(y, dtype=np.float32)


def split_into_segments(y: np.ndarray, sr: int = SAMPLE_RATE, seconds: int = SEGMENT_SECONDS) -> list[np.ndarray]:
    """Consecutive fixed-length segments; the final partial segment is zero-padded."""
    seg_len = sr * seconds
    segments = [y[i : i + seg_len] for i in range(0, len(y) - seg_len + 1, seg_len)]
    remainder = len(y) % seg_len
    if remainder or not segments:
        tail = y[len(segments) * seg_len :]
        segments.append(np.pad(tail, (0, seg_len - len(tail))))
    return segments


def log_mel(segment: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=np.max)


def spectrogram_image(segment: np.ndarray) -> Image.Image:
    """Grayscale spectrogram image, byte-identical to ``plt.imsave(..., cmap="gray")`` used in training."""
    spec = log_mel(segment)
    _GRAY.set_norm(colors.Normalize())  # fresh min-max scaling per segment, as imsave does
    rgba = _GRAY.to_rgba(spec, bytes=True)
    return Image.fromarray(rgba, "RGBA").convert("L")


def image_to_input(img: Image.Image) -> np.ndarray:
    """Grayscale image -> float32 array [3, 224, 224] in [-1, 1] (torchvision Grayscale/Resize/Normalize)."""
    img = img.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return np.repeat(x[None], 3, axis=0)


def audio_to_batch(y: np.ndarray) -> np.ndarray:
    """Waveform -> batch [n_segments, 3, 224, 224] ready for the model."""
    return np.stack([image_to_input(spectrogram_image(s)) for s in split_into_segments(y)])
