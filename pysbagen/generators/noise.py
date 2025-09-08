import numpy as np
from dataclasses import dataclass
from scipy.signal import lfilter
from .base import GenBase

@dataclass
class NoiseSpec(GenBase):
    kind: str = "white"  # "white" | "pink"

    def generator(self, duration: float):
        num = int(self.sample_rate * duration)
        a = self._amp_scale()
        # simple Voss-McCartney-ish pink: 1st-order filter (approx)
        b = [1.0]; a_pink = [1.0, -0.985]  # crude, fast
        for i in range(0, num, self.frame):
            n = min(self.frame, num - i)
            w = np.random.normal(0, 1, (n, 2))
            if self.kind.lower() == "pink":
                # filter each channel
                w[:,0] = lfilter(b, a_pink, w[:,0])
                w[:,1] = lfilter(b, a_pink, w[:,1])
            yield (w * a).astype(np.float32), {"type": "noise", "kind": self.kind}
