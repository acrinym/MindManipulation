import os
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from dataclasses import dataclass
from .base import GenBase

@dataclass
class FileSpec(GenBase):
    path: str = ""
    loop: bool = False

    def _load(self):
        if self.path.lower().endswith(".mp3"):
            seg = AudioSegment.from_mp3(self.path)
            seg = seg.set_frame_rate(self.sample_rate).set_channels(2)
            data = np.array(seg.get_array_of_samples(), dtype=np.int16).reshape(-1, 2)
            data = data.astype(np.float32) / 32768.0
            return data
        data, rate = sf.read(self.path, dtype="float32", always_2d=True)
        if rate != self.sample_rate:
            raise ValueError(f"Sample rate mismatch: {rate} != {self.sample_rate}")
        return data

    def generator(self, duration: float):
        data = self._load()
        num = int(self.sample_rate * duration)
        if len(data) < num and self.loop:
            reps = int(np.ceil(num / len(data)))
            data = np.tile(data, (reps, 1))
        data = data[:num] * self._amp_scale()
        for i in range(0, len(data), self.frame):
            chunk = data[i:i+self.frame]
            if chunk.shape[0] == 0: break
            yield chunk.astype(np.float32), {"type": "file", "path": os.path.basename(self.path)}
