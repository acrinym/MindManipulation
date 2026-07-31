from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import soundfile as sf

from .generators import FileSpec, HarmonicBoxSpec, IsochronicSpec, NoiseSpec, ToneSpec
from .importers import ImportedArtifact, import_artifact
from .mixer import SR, build_session_generator, mix_generators
from .sleep import SleepRequest, build_sleep_spec


@dataclass(frozen=True)
class RenderResult:
    outfile: Path
    frames: int
    sample_rate: int
    peak: float
    manifest: Path | None = None

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate


@dataclass
class InspectedRender:
    artifact: ImportedArtifact
    duration: float
    path_qualification: dict | None
    chunks: Iterator[tuple[np.ndarray, list[dict]]]

    def __iter__(self):
        return iter(self.chunks)


def inspect_artifact(path: str | Path, *, preserve_to: str | Path | None = None) -> ImportedArtifact:
    """Import an SBG/DRG artifact without starting playback or rendering."""
    return import_artifact(path, preserve_to=preserve_to)


def render_specs(specs: Iterable, duration: float):
    return mix_generators(specs, duration)


def render_schedule(
    path: str | Path,
    duration: float | None = None,
    *,
    allow_disclosed_changes: bool = False,
    path_qualification: dict | None = None,
):
    """Render only after the canonical compatibility report permits it.

    DRG and SBG files share this path. Partial or approximated imports require an
    explicit acknowledgement; blocked and inspection-only imports never render.
    """
    artifact = import_artifact(path)
    artifact.report.require_renderable(allow_disclosed_changes=allow_disclosed_changes)
    render_duration = artifact.require_duration(duration)
    if path_qualification is not None and path_qualification.get("safe_to_start") is False:
        raise ValueError("Attached listening-path qualification blocks playback/rendering")
    return InspectedRender(
        artifact=artifact,
        duration=render_duration,
        path_qualification=path_qualification,
        chunks=build_session_generator(artifact.tone_sets, artifact.schedule, render_duration),
    )


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
    chunks: Iterator[tuple[np.ndarray, list[dict]]] | InspectedRender,
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

    inspected = chunks if isinstance(chunks, InspectedRender) else None
    frames = 0
    peak = 0.0
    manifest_path: Path | None = None
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
        if inspected is not None:
            manifest_path = path.with_name(path.name + ".pysbagen.json")
            payload = {
                "schema": "pysbagen.render-manifest.v1",
                "rendered_at": datetime.now(timezone.utc).isoformat(),
                "output": {
                    "path": str(path.resolve()),
                    "frames": frames,
                    "sample_rate": sample_rate,
                    "duration": frames / sample_rate,
                    "peak": peak,
                    "sha256": _sha256_file(path),
                },
                "source_import_report": inspected.artifact.report.to_dict(),
                "accepted_disclosed_changes": inspected.artifact.report.requires_acknowledgement,
                "path_qualification": inspected.path_qualification,
            }
            descriptor, temporary_manifest_name = tempfile.mkstemp(
                prefix=f".{manifest_path.name}-", suffix=".tmp", dir=parent
            )
            os.close(descriptor)
            temporary_manifest = Path(temporary_manifest_name)
            try:
                temporary_manifest.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary_manifest, manifest_path)
            finally:
                temporary_manifest.unlink(missing_ok=True)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return RenderResult(path.resolve(), frames, sample_rate, peak, manifest_path.resolve() if manifest_path else None)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
