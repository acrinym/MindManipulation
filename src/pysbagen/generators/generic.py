import numpy as np
from dataclasses import dataclass
from .base import GenBase

@dataclass
class GenericToneSpec(GenBase):
    freq: float = 200.0
    waveform: str = "sine"

    def generator(self, duration: float):
        num = int(self.sample_rate * duration)
        for i in range(0, num, self.frame):
            n = min(self.frame, num - i)
            t = np.linspace(i / self.sample_rate, (i + n) / self.sample_rate, n, endpoint=False)

            if self.waveform == "sine":
                wave = np.sin(2 * np.pi * self.freq * t)
            elif self.waveform == "square":
                wave = np.sign(np.sin(2 * np.pi * self.freq * t))
            elif self.waveform == "triangle":
                wave = 2 * np.abs(2 * (t * self.freq - np.floor(t * self.freq + 0.5))) - 1
            elif self.waveform == "sawtooth":
                wave = 2 * (t * self.freq - np.floor(t * self.freq + 0.5))
            else: # sine is the default
                wave = np.sin(2 * np.pi * self.freq * t)

            chunk = self._stereo(wave) * self._amp_scale()
            yield chunk.astype(np.float32), {"type": "generic", "freq": self.freq, "waveform": self.waveform}
