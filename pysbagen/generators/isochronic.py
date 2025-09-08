import numpy as np
from dataclasses import dataclass
from .base import GenBase

@dataclass
class IsochronicSpec(GenBase):
    freq: float = 200.0
    beat: float = 10.0

    def generator(self, duration: float):
        num = int(self.sample_rate * duration)
        for i in range(0, num, self.frame):
            n = min(self.frame, num - i)
            t = np.linspace(i / self.sample_rate, (i + n) / self.sample_rate, n, endpoint=False)
            tone = np.sin(2 * np.pi * self.freq * t)
            gate = (np.sin(2 * np.pi * self.beat * t) > 0).astype(np.float32)
            mod = tone * gate
            chunk = self._stereo(mod) * self._amp_scale()
            yield chunk.astype(np.float32), {"type": "isochronic", "freq": self.freq, "beat": self.beat}
