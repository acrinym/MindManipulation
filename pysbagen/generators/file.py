from dataclasses import dataclass
from math import gcd
from pathlib import Path
import shutil
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .base import GenBase


@dataclass
class FileSpec(GenBase):
    path: str = ""
    loop: bool = False

    def _load_mp3(self, path: Path) -> np.ndarray:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("MP3 decoding requires FFmpeg to be installed and available on PATH")

        command = [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "2",
            "-ar",
            str(self.sample_rate),
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
