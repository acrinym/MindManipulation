from dataclasses import dataclass
from typing import Generator, TypedDict, Protocol, Union
import numpy as np

from .generators import (
    ToneSpec,
    NoiseSpec,
    FileSpec,
    IsochronicSpec,
    HarmonicBoxSpec,
    GenericToneSpec,
)

class ChunkInfo(TypedDict, total=False):
    type: str
    base: float
    beat: float
    freq: float
    path: str
    note: str

class AudioGen(Protocol):
    def generator(self, duration: float) -> Generator[tuple[np.ndarray, ChunkInfo], None, None]: ...


AnySpec = Union[
    ToneSpec,
    NoiseSpec,
    FileSpec,
    IsochronicSpec,
    HarmonicBoxSpec,
    GenericToneSpec,
]
