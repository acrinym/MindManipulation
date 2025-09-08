from dataclasses import dataclass
from typing import Generator, TypedDict, Protocol
import numpy as np

class ChunkInfo(TypedDict, total=False):
    type: str
    base: float
    beat: float
    freq: float
    path: str
    note: str

class AudioGen(Protocol):
    def generator(self, duration: float) -> Generator[tuple[np.ndarray, ChunkInfo], None, None]: ...
