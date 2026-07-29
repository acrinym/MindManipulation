from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from scipy.signal import resample_poly

from .base import GenBase


@dataclass
class FileSpec(GenBase):
    path: str = ""
    loop: bool = False

    def _load_mp3(self, path: Path) -> np.ndarray:
        segment = AudioSegment.from_mp3(path)
        segment = segment.set_frame_rate(self.sample_rate).set_channels(2)
        raw = np.asarray(segment.get_array_of_samples())
        data = raw.reshape((-1, 2)).astype(np.float32)
        scale = float(1 << (8 * segment.sample_width - 1))
        return data / scale

    @staticmethod
    def _stereo_data(data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            data = data[:, None]
        if data.shape[1] == 1:
            return np.repeat(data, 2, axis=1)
        return data[:, :2]

    def _load(self) -> np.ndarray:
        path = Path(self.path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if path.suffix.lower() == ".mp3":
            return self._load_mp3(path)

        data, rate = sf.read(path, dtype="float32", always_2d=True)
        data = self._stereo_data(data)
        if rate != self.sample_rate:
            factor = gcd(rate, self.sample_rate)
            data = resample_poly(
                data,
                self.sample_rate // factor,
                rate // factor,
                axis=0,
            ).astype(np.float32)
        return data

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
