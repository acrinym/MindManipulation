from dataclasses import dataclass
import numpy as np

DEFAULT_SAMPLE_RATE = 44100
FRAME = 1024

@dataclass
class GenBase:
    amp: float = 100.0
    sample_rate: int = DEFAULT_SAMPLE_RATE
    frame: int = FRAME

    def _stereo(self, mono: np.ndarray) -> np.ndarray:
        return np.vstack((mono, mono)).T

    def _amp_scale(self) -> float:
        return max(self.amp, 0.0) / 100.0
