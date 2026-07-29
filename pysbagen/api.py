from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import soundfile as sf

from .generators import FileSpec, HarmonicBoxSpec, IsochronicSpec, NoiseSpec, ToneSpec
from .mixer import SR, build_session_generator, mix_generators
from .parser import parse_sbg


@dataclass(frozen=True)
class RenderResult:
    outfile: Path
    frames: int
    sample_rate: int
    peak: float

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate


def render_specs(specs: Iterable, duration: float):
    return mix_generators(specs, duration)


def render_schedule(path: str | Path, duration: float | None = None):
    tone_sets, schedule = parse_sbg(path)
    return build_session_generator(tone_sets, schedule, duration)


def build_quick_specs(
    *,
    base: float | None = None,
    beat: float | None = None,
    isochronic: tuple[float, float] | None = None,
    harmonic_box: tuple[float, float, float] | None = None,
    noise: float | None = None,
    noise_kind: str = "white",
    music: str | None = None,
    music_amp: float = 100.0,
    loop_music: bool = False,
):
    if (base is None) != (beat is None):
        raise ValueError("base and beat must be supplied together")

    specs = []
    if base is not None and beat is not None:
        specs.append(ToneSpec(base=float(base), beat=float(beat)))
    if isochronic is not None:
        specs.append(IsochronicSpec(freq=float(isochronic[0]), beat=float(isochronic[1])))
    if harmonic_box is not None:
        specs.append(
            HarmonicBoxSpec(
                base=float(harmonic_box[0]),
                diff=float(harmonic_box[1]),
                mod=float(harmonic_box[2]),
            )
        )
    if noise is not None:
        specs.append(NoiseSpec(amp=float(noise), kind=noise_kind))
    if music:
        specs.append(FileSpec(path=music, amp=float(music_amp), loop=loop_music))
    if not specs:
        raise ValueError("At least one generator must be selected")
    return specs


def write_audio(
    chunks: Iterator[tuple[np.ndarray, list[dict]]],
    outfile: str | Path,
    sample_rate: int = SR,
) -> RenderResult:
    path = Path(outfile).expanduser()
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

    frames = 0
    peak = 0.0
    with sf.SoundFile(path, mode="w", samplerate=sample_rate, channels=2) as output:
        for chunk, _ in chunks:
            normalized = np.asarray(chunk, dtype=np.float32)
            if normalized.ndim != 2 or normalized.shape[1] != 2:
                raise ValueError(f"Writer expected stereo chunks, got {normalized.shape}")
            output.write(normalized)
            frames += len(normalized)
            if len(normalized):
                peak = max(peak, float(np.max(np.abs(normalized))))

    if frames == 0:
        path.unlink(missing_ok=True)
        raise ValueError("No audio was generated")
    return RenderResult(path.resolve(), frames, sample_rate, peak)
