import numpy as np
from dataclasses import dataclass
from .base import GenBase

@dataclass
class HarmonicBoxSpec(GenBase):
    base: float = 180.0
    diff: float = 5.0
    mod: float = 8.0

    def generator(self, duration: float):
        num = int(self.sample_rate * duration)
        for i in range(0, num, self.frame):
            n = min(self.frame, num - i)
            t = np.linspace(i / self.sample_rate, (i + n) / self.sample_rate, n, endpoint=False)
            phases = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
            left = np.zeros_like(t)
            right = np.zeros_like(t)
            for ph in phases:
                gate = (np.sin(2 * np.pi * self.mod * t + ph) > 0).astype(np.float32)
                left += np.sin(2 * np.pi * self.base * t) * gate
                right += np.sin(2 * np.pi * (self.base + self.diff) * t) * gate
            chunk = np.vstack((left, right)).T * self._amp_scale() / len(phases)
            yield chunk.astype(np.float32), {"type": "harmonic_box", "base": self.base, "diff": self.diff, "mod": self.mod}
