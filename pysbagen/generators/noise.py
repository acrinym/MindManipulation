from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .base import GenBase


@dataclass
class NoiseSpec(GenBase):
    kind: str = "white"
    seed: int | None = None

    def generator(self, duration: float):
        num = max(0, int(self.sample_rate * duration))
        kind = self.kind.lower()
        if kind not in {"white", "pink"}:
            raise ValueError(f"Unsupported noise kind: {self.kind}")

        rng = np.random.default_rng(self.seed)
        b = np.array([1.0], dtype=np.float64)
        a_pink = np.array([1.0, -0.985], dtype=np.float64)
        zi = [np.zeros(max(len(a_pink), len(b)) - 1) for _ in range(2)]

        for i in range(0, num, self.frame):
            n = min(self.frame, num - i)
            samples = rng.normal(0, 1, (n, 2))
            if kind == "pink":
                for channel in range(2):
                    samples[:, channel], zi[channel] = lfilter(
                        b, a_pink, samples[:, channel], zi=zi[channel]
                    )
            yield (samples * self._amp_scale()).astype(np.float32), {
                "type": "noise",
                "kind": kind,
            }
