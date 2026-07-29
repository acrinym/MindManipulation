from dataclasses import dataclass
from math import gcd
from pathlib import Path
import shutil
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .base import GenBase


def _decode_with_ffmpeg(path: Path, sample_rate: int) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            f"Decoding {path.suffix or 'this audio format'} requires FFmpeg to be installed and available on PATH"
        )
    command = [
        ffmpeg,
        "-v", "error",
        "-i", str(path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "2",
        "-ar", str(sample_rate),
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"FFmpeg could not decode {path.name}: {detail or 'unknown error'}")
    samples = np.frombuffer(result.stdout, dtype="<f4")
    if samples.size % 2:
        raise ValueError(f"FFmpeg returned malformed stereo audio for {path.name}")
    return samples.reshape((-1, 2)).copy()


def _stereo_data(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError(f"Audio data must be one- or two-dimensional, got {data.shape}")
    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    return data[:, :2]


def load_audio(path_value: str | Path, sample_rate: int = 44100) -> np.ndarray:
    """Load any audio format supported by SoundFile or the local FFmpeg installation."""
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        data, rate = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, TypeError, ValueError):
        return _decode_with_ffmpeg(path, sample_rate)

    data = _stereo_data(data)
    if rate != sample_rate:
        factor = gcd(rate, sample_rate)
        data = resample_poly(
            data,
            sample_rate // factor,
            rate // factor,
            axis=0,
        ).astype(np.float32)
    return data.astype(np.float32, copy=False)


@dataclass
class FileSpec(GenBase):
    path: str = ""
    loop: bool = False

    def _load(self) -> np.ndarray:
        return load_audio(self.path, self.sample_rate)

    def generator(self, duration: float):
        data = self._load()
        if len(data) == 0:
            raise ValueError(f"Audio file is empty: {self.path}")
        num = max(0, int(self.sample_rate * duration))
        if len(data) < num and self.loop:
            reps = int(np.ceil(num / len(data)))
            data = np.tile(data, (reps, 1))
        data = data[:num] * self._amp_scale()
        for i in range(0, len(data), self.frame):
            chunk = data[i : i + self.frame]
            yield chunk.astype(np.float32), {
                "type": "file",
                "path": Path(self.path).name,
            }
