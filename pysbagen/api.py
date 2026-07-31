from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import soundfile as sf

from .generators import FileSpec, HarmonicBoxSpec, IsochronicSpec, NoiseSpec, ToneSpec
from .mixer import SR, build_session_generator, mix_generators
from .parser import parse_sbg
from .sleep import SleepRequest, build_sleep_spec


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


def render_sleep(request: SleepRequest):
    return render_specs([build_sleep_spec(request)], request.duration_seconds)


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
    """Write atomically so a failed render never destroys an existing destination."""
    path = Path(outfile).expanduser()
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent if path.parent != Path("") else Path(".")
    suffix = path.suffix or ".wav"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem or 'pysbagen'}-",
        suffix=f".tmp{suffix}",
        dir=parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    frames = 0
    peak = 0.0
    try:
        with sf.SoundFile(
            temporary_path,
            mode="w",
            samplerate=sample_rate,
            channels=2,
            format="WAV" if not path.suffix else None,
        ) as output:
            for chunk, _ in chunks:
                normalized = np.asarray(chunk, dtype=np.float32)
                if normalized.ndim != 2 or normalized.shape[1] != 2:
                    raise ValueError(f"Writer expected stereo chunks, got {normalized.shape}")
                output.write(normalized)
                frames += len(normalized)
                if len(normalized):
                    peak = max(peak, float(np.max(np.abs(normalized))))

        if frames == 0:
            raise ValueError("No audio was generated")
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return RenderResult(path.resolve(), frames, sample_rate, peak)
