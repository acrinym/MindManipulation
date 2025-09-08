import numpy as np
from dataclasses import dataclass
from .base import GenBase

@dataclass
class ToneSpec(GenBase):
    base: float = 200.0
    beat: float = 10.0

    def generator(self, duration: float):
        num = int(self.sample_rate * duration)
        for i in range(0, num, self.frame):
            n = min(self.frame, num - i)
            t = np.linspace(i / self.sample_rate, (i + n) / self.sample_rate, n, endpoint=False)
            left  = np.sin(2 * np.pi * self.base * t)
            right = np.sin(2 * np.pi * (self.base + self.beat) * t)
            chunk = np.vstack((left, right)).T * self._amp_scale()
            yield chunk.astype(np.float32), {"type": "binaural", "base": self.base, "beat": self.beat}
