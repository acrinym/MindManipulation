from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .compatibility import CompatibilityState
from .importers import ImportedArtifact


@dataclass(frozen=True)
class TimelineLayer:
    tone_set: str
    component_type: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class TimelineSegment:
    start: float
    end: float | None
    active_tone_sets: tuple[str, ...]
    layers: tuple[TimelineLayer, ...]
    transition_to_next: bool = False
    source_tokens: tuple[str, ...] = ()

    @property
    def duration(self) -> float | None:
        return None if self.end is None else max(0.0, self.end - self.start)


@dataclass
class AudioSourceReport:
    path: str
    exists: bool
    codec: str | None = None
    container: str | None = None
    channels: int | None = None
    sample_rate: int | None = None
    duration: float | None = None
    frames: int | None = None
    peak: float | None = None
    clipping_samples: int | None = None
    stereo_correlation: float | None = None
    near_mono: bool | None = None
    anti_phase: bool | None = None
    resampling_required: bool | None = None
    state: CompatibilityState = CompatibilityState.UNKNOWN
    suitability: str = "not analyzed"
    warnings: list[str] = field(default_factory=list)
    analysis_backend: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["state"] = self.state.value
        return result

    def to_text(self) -> str:
        lines = [
            f"Source: {self.path}",
            f"State: {self.state.value}",
            f"Suitability: {self.suitability}",
            f"Container/codec: {self.container or 'unknown'} / {self.codec or 'unknown'}",
            f"Channels: {self.channels if self.channels is not None else 'unknown'}",
            f"Sample rate: {self.sample_rate if self.sample_rate is not None else 'unknown'}",
            f"Duration: {self.duration:.3f}s" if self.duration is not None else "Duration: unknown",
            f"Peak: {self.peak:.6f}" if self.peak is not None else "Peak: not sampled",
            (
                f"Stereo correlation: {self.stereo_correlation:.6f}"
                if self.stereo_correlation is not None
                else "Stereo correlation: unavailable"
            ),
            f"Analysis backend: {self.analysis_backend or 'none'}",
        ]
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  {warning}" for warning in self.warnings)
        return "\n".join(lines)


@dataclass
class AudioPathQualification:
    method: str
    route: str
    channels: int
    sample_rate: int
    spatial_processing: bool
    normalization: bool
    bluetooth: bool
    state: CompatibilityState
    safe_to_start: bool
    findings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["state"] = self.state.value
        return result

    def to_text(self) -> str:
        lines = [
            f"Method: {self.method}",
            f"Route: {self.route}",
            f"Channels: {self.channels}",
            f"Sample rate: {self.sample_rate}",
            f"State: {self.state.value}",
            f"Safe to start: {'yes' if self.safe_to_start else 'no'}",
        ]
        lines.extend(f"  {finding}" for finding in self.findings)
        return "\n".join(lines)


def build_timeline(
    artifact: ImportedArtifact,
    *,
    duration: float | None = None,
) -> list[TimelineSegment]:
    schedule = sorted(artifact.schedule, key=lambda item: item[0])
    if not schedule:
        return []
    final_end = duration if duration is not None else artifact.report.inferred_duration
    active: list[str] = []
    segments: list[TimelineSegment] = []
    for index, (start, tokens) in enumerate(schedule):
        active = _apply_label_event(active, artifact.tone_sets, tokens)
        next_start = schedule[index + 1][0] if index + 1 < len(schedule) else final_end
        layers = tuple(
            TimelineLayer(label, type(spec).__name__, _spec_parameters(spec))
            for label in active
            for spec in artifact.tone_sets.get(label, [])
        )
        segments.append(
            TimelineSegment(
                float(start),
                float(next_start) if next_start is not None else None,
                tuple(active),
                layers,
                bool(tokens and tokens[-1] == "->"),
                tuple(tokens),
            )
        )
    return segments


def timeline_to_dict(timeline: list[TimelineSegment]) -> list[dict[str, Any]]:
    return [asdict(segment) for segment in timeline]


def timeline_to_text(timeline: list[TimelineSegment]) -> str:
    if not timeline:
        return "Timeline: no playable events"
    lines = ["Timeline:"]
    for segment in timeline:
        end = f"{segment.end:.3f}s" if segment.end is not None else "open"
        active = ", ".join(segment.active_tone_sets) or "silence"
        lines.append(
            f"  {segment.start:.3f}s – {end}: {active}"
            f"{' -> crossfade' if segment.transition_to_next else ''}"
        )
        for layer in segment.layers:
            parameters = ", ".join(
                f"{key}={value}" for key, value in layer.parameters.items()
            )
            lines.append(
                f"    {layer.tone_set}: {layer.component_type}({parameters})"
            )
    return "\n".join(lines)


def inspect_audio_source(
    path: str | Path,
    *,
    target_sample_rate: int = 44100,
) -> AudioSourceReport:
    source = Path(path).expanduser().resolve()
    report = AudioSourceReport(path=str(source), exists=source.is_file())
    if not report.exists:
        report.state = CompatibilityState.MISSING_SOURCE
        report.suitability = "source file is unavailable"
        report.warnings.append("The referenced audio file does not exist.")
        return report

    try:
        import soundfile as sf

        info = sf.info(source)
        report.container = info.format
        report.codec = info.subtype
        report.channels = info.channels
        report.sample_rate = info.samplerate
        report.frames = info.frames
        report.duration = info.duration
        report.analysis_backend = "soundfile"
        sample_limit = min(info.frames, max(info.samplerate * 30, 1))
        with sf.SoundFile(source) as handle:
            samples = handle.read(sample_limit, dtype="float32", always_2d=True)
            _analyze_samples(report, samples)
    except Exception as exc:
        report.warnings.append(f"SoundFile analysis was unavailable: {exc}")
        _inspect_with_ffprobe(report, source)

    if report.sample_rate is not None:
        report.resampling_required = report.sample_rate != target_sample_rate
        if report.resampling_required:
            report.warnings.append(
                f"Source will be resampled from {report.sample_rate} Hz "
                f"to {target_sample_rate} Hz."
            )
    _classify_source(report)
    return report


def qualify_audio_path(
    *,
    method: str,
    route: str,
    channels: int,
    sample_rate: int,
    spatial_processing: bool = False,
    normalization: bool = False,
    bluetooth: bool = False,
) -> AudioPathQualification:
    method = method.strip().lower()
    route = route.strip().lower()
    if channels <= 0:
        raise ValueError("Output channel count must be positive")
    if sample_rate <= 0:
        raise ValueError("Output sample rate must be positive")

    findings: list[str] = []
    state = CompatibilityState.SUPPORTED
    safe = True
    if channels < 2 and method in {"binaural", "mixed"}:
        findings.append(
            "Binaural content requires two independent output channels; "
            "mono/downmix routing blocks playback."
        )
        state = CompatibilityState.UNSAFE_TO_RENDER
        safe = False
    if method == "binaural" and route not in {"headphones", "earbuds"}:
        findings.append(
            "Binaural separation is not reliable on the selected route; "
            "use headphones or earbuds."
        )
        state = CompatibilityState.UNSAFE_TO_RENDER
        safe = False
    if spatial_processing:
        findings.append(
            "Spatial enhancement can cross-mix left and right channels and "
            "invalidate intended separation."
        )
        state = CompatibilityState.UNSAFE_TO_RENDER
        safe = False
    if sample_rate < 32000:
        findings.append(
            "The negotiated sample rate is unusually low for this product path."
        )
        if safe:
            state = CompatibilityState.PARTIAL
    if normalization:
        findings.append(
            "Loudness normalization may alter relative layer amplitudes; "
            "disable it for protocol fidelity."
        )
        if safe and state is CompatibilityState.SUPPORTED:
            state = CompatibilityState.PARTIAL
    if bluetooth:
        findings.append(
            "Bluetooth may add codec processing, resampling, and channel "
            "handling outside PySbagen's control."
        )
        if safe and state is CompatibilityState.SUPPORTED:
            state = CompatibilityState.PARTIAL
    if not findings:
        findings.append(
            "Stereo route and processing declarations are compatible with "
            "the selected method."
        )
    return AudioPathQualification(
        method,
        route,
        int(channels),
        int(sample_rate),
        bool(spatial_processing),
        bool(normalization),
        bool(bluetooth),
        state,
        safe,
        findings,
    )


def _apply_label_event(
    active: list[str],
    tone_sets: dict[str, list[Any]],
    tokens: list[str],
) -> list[str]:
    items = tokens[:-1] if tokens and tokens[-1] == "->" else tokens
    if not items:
        return list(active)
    active = (
        []
        if items[0].lower() in {"-", "off", "alloff"}
        or not items[0].startswith(("+", "-"))
        else list(active)
    )
    for token in items:
        lowered = token.lower()
        if lowered in {"-", "off", "alloff"}:
            if lowered in {"-", "alloff"}:
                active = []
            continue
        operation = token[0] if token.startswith(("+", "-")) else "+"
        label = token.lstrip("+-")
        if label not in tone_sets:
            continue
        if operation == "-":
            active = [item for item in active if item != label]
        elif label not in active:
            active.append(label)
    return active


def _spec_parameters(spec: Any) -> dict[str, Any]:
    if is_dataclass(spec):
        values = asdict(spec)
    elif hasattr(spec, "__dict__"):
        values = {
            key: value
            for key, value in vars(spec).items()
            if not key.startswith("_")
        }
    else:
        values = {"value": repr(spec)}
    return {
        key: (
            str(value)
            if isinstance(value, Path)
            else value
            if isinstance(value, (str, int, float, bool)) or value is None
            else repr(value)
        )
        for key, value in values.items()
    }


def _analyze_samples(report: AudioSourceReport, samples: np.ndarray) -> None:
    if samples.size == 0:
        report.peak = 0.0
        report.clipping_samples = 0
        return
    report.peak = float(np.max(np.abs(samples)))
    report.clipping_samples = int(np.count_nonzero(np.abs(samples) >= 0.999))
    if samples.shape[1] < 2:
        return

    left = samples[:, 0].astype(np.float64)
    right = samples[:, 1].astype(np.float64)
    if np.std(left) > 1e-12 and np.std(right) > 1e-12:
        correlation = float(np.corrcoef(left, right)[0, 1])
        report.stereo_correlation = correlation
        report.near_mono = correlation >= 0.995
        report.anti_phase = correlation <= -0.995
    else:
        report.near_mono = bool(np.allclose(left, right, atol=1e-6))
        report.anti_phase = False
        report.stereo_correlation = 1.0 if report.near_mono else None


def _inspect_with_ffprobe(report: AudioSourceReport, source: Path) -> None:
    executable = shutil.which("ffprobe")
    if executable is None:
        report.warnings.append(
            "ffprobe is not installed, so only file existence could be verified."
        )
        return
    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,channels,sample_rate,duration,nb_frames:"
        "format=format_name,duration",
        "-of",
        "json",
        str(source),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        stream = (payload.get("streams") or [{}])[0]
        format_info = payload.get("format") or {}
        report.codec = stream.get("codec_name")
        report.container = format_info.get("format_name")
        report.channels = _optional_int(stream.get("channels"))
        report.sample_rate = _optional_int(stream.get("sample_rate"))
        report.frames = _optional_int(stream.get("nb_frames"))
        report.duration = _optional_float(
            stream.get("duration") or format_info.get("duration")
        )
        report.analysis_backend = "ffprobe-metadata"
        report.warnings.append(
            "Waveform suitability could not be sampled; metadata-only "
            "qualification was used."
        )
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        report.warnings.append(f"ffprobe analysis failed: {exc}")


def _classify_source(report: AudioSourceReport) -> None:
    if report.channels is None or report.sample_rate is None:
        report.state = CompatibilityState.UNKNOWN
        report.suitability = "metadata is insufficient for reliable qualification"
        return

    state = CompatibilityState.SUPPORTED
    suitability = "usable as a local audio layer"
    if report.channels < 2:
        state = CompatibilityState.PARTIAL
        suitability = (
            "usable as a mono bed, not as an independently separated "
            "binaural source"
        )
        report.warnings.append(
            "Source is mono; it cannot preserve independent left/right source content."
        )
    if report.near_mono:
        state = CompatibilityState.PARTIAL
        suitability = "stereo container is effectively mono or strongly correlated"
        report.warnings.append("Left and right channels are nearly identical.")
    if report.anti_phase:
        state = CompatibilityState.PARTIAL
        suitability = (
            "strongly anti-correlated stereo; usable with mono-downmix caution"
        )
        report.warnings.append(
            "Left and right channels are strongly anti-correlated; a mono "
            "downmix may cancel much of the signal."
        )
    if report.clipping_samples:
        state = CompatibilityState.PARTIAL
        report.warnings.append(
            f"Detected {report.clipping_samples} near-clipping samples in the "
            "analyzed window."
        )
    if report.resampling_required and state is CompatibilityState.SUPPORTED:
        state = CompatibilityState.EQUIVALENT
        suitability = "usable with disclosed resampling"
    report.state = state
    report.suitability = suitability


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None
