from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .generators import FileSpec, HarmonicBoxSpec, IsochronicSpec, NoiseSpec, ToneSpec
from .types import AnySpec

_AUDIO_EXTENSIONS = {".wav", ".ogg", ".flac", ".aif", ".aiff", ".mp3", ".m4a", ".aac", ".opus", ".wma", ".caf"}
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def _parse_hms(value: str) -> int:
    parts = value.split(":")
    if not 1 <= len(parts) <= 3 or any(not part.isdigit() for part in parts):
        raise ValueError(f"Invalid schedule time: {value!r}")
    numbers = [int(part) for part in parts]
    if len(numbers) == 1:
        return numbers[0]
    if numbers[-1] >= 60:
        raise ValueError(f"Schedule seconds must be below 60: {value!r}")
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    if numbers[1] >= 60:
        raise ValueError(f"Schedule minutes must be below 60: {value!r}")
    return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]


def _split_params_and_amp(value: str, default_amp: float = 100.0) -> tuple[str, float]:
    if "/" not in value:
        return value, default_amp
    params, amp_text = value.rsplit("/", 1)
    if not _NUMBER.match(amp_text.strip()):
        return value, default_amp
    return params, float(amp_text)


def _resolve_audio_path(path_text: str, base_dir: Path | None) -> Path:
    path = Path(os.path.expandvars(path_text)).expanduser()
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def _parse_file_component(spec: str, base_dir: Path | None) -> Optional[FileSpec]:
    if "/" in spec:
        path_text, amp_text = spec.rsplit("/", 1)
        if _NUMBER.match(amp_text.strip()):
            candidate = _resolve_audio_path(path_text, base_dir)
            if candidate.suffix.lower() in _AUDIO_EXTENSIONS or candidate.is_file():
                return FileSpec(path=str(candidate), amp=float(amp_text))
    candidate = _resolve_audio_path(spec, base_dir)
    if candidate.suffix.lower() in _AUDIO_EXTENSIONS or candidate.is_file():
        return FileSpec(path=str(candidate))
    return None


def parse_tone_component(spec: str, base_dir: str | Path | None = None) -> Optional[AnySpec]:
    spec = spec.strip()
    if not spec:
        raise ValueError("Tone component cannot be empty")
    if spec.lower() in {"-", "off"}:
        return None
    resolved_base = Path(base_dir) if base_dir is not None else None
    prefix, separator, rest = spec.partition(":")
    recognized_prefix = prefix.lower() if separator else ""
    if recognized_prefix == "iso":
        params_text, amp = _split_params_and_amp(rest)
        params = [float(value) for value in params_text.split(",")]
        if len(params) != 2:
            raise ValueError("iso requires frequency and beat: iso:FREQ,BEAT[/AMP]")
        return IsochronicSpec(freq=params[0], beat=params[1], amp=amp)
    if recognized_prefix == "hbox":
        params_text, amp = _split_params_and_amp(rest)
        params = [float(value) for value in params_text.split(",")]
        if len(params) != 3:
            raise ValueError("hbox requires base, difference, and modulation")
        return HarmonicBoxSpec(base=params[0], diff=params[1], mod=params[2], amp=amp)
    if recognized_prefix == "file":
        parsed_file = _parse_file_component(rest, resolved_base)
        if parsed_file is None:
            raise ValueError(f"Unsupported audio file component: {rest!r}")
        return parsed_file
    if recognized_prefix in {"spin", "slide"}:
        spec = rest
    noise_match = re.fullmatch(r"(?i)(pink|white)(?:/([+-]?(?:\d+(?:\.\d*)?|\.\d+)))?", spec)
    if noise_match:
        return NoiseSpec(kind=noise_match.group(1).lower(), amp=float(noise_match.group(2) or 100.0))
    parsed_file = _parse_file_component(spec, resolved_base)
    if parsed_file is not None:
        return parsed_file
    core_spec, amp = _split_params_and_amp(spec)
    beat = 0.0
    if "+" in core_spec:
        base_text, beat_text = core_spec.split("+", 1)
        beat = float(beat_text)
    elif "-" in core_spec and core_spec.count("-") == 1 and not core_spec.startswith("-"):
        base_text, beat_text = core_spec.split("-", 1)
        beat = -float(beat_text)
    else:
        base_text = core_spec
    return ToneSpec(base=float(base_text), beat=beat, amp=amp)


def parse_sbg_from_string(source: str, base_dir: str | Path | None = None) -> tuple[Dict[str, List[AnySpec]], List[Tuple[float, List[str]]]]:
    tone_sets: Dict[str, List[AnySpec]] = {}
    schedule: List[Tuple[float, List[str]]] = []
    resolved_base = Path(base_dir) if base_dir is not None else None
    for line_number, raw in enumerate(source.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            is_schedule = line.upper().startswith("NOW") or line[0].isdigit() or line.startswith("+")
            if not is_schedule and ":" in line:
                label, rest = line.split(":", 1)
                label = label.strip()
                if not label:
                    raise ValueError("Tone-set label cannot be empty")
                parts = shlex.split(rest, comments=True, posix=True)
                tone_sets[label] = [component for part in parts if (component := parse_tone_component(part, resolved_base)) is not None]
                continue
            if line.upper().startswith("NOW"):
                time_value = 0
                rest = line[3:].strip()
            else:
                time_text, rest = line.split(maxsplit=1)
                time_value = _parse_hms(time_text.lstrip("+"))
            raw_items = [item for item in shlex.split(rest) if item]
            transition_positions = [index for index, item in enumerate(raw_items) if item == "->"]
            if transition_positions and transition_positions != [len(raw_items) - 1]:
                raise ValueError("'->' must end a schedule line and transitions to the next timed event")
            transition = bool(transition_positions)
            if transition:
                raw_items.pop()
            if raw_items and raw_items[0] in {"==", "--", "<>", "<-", "->", "=-", "-="}:
                raw_items.pop(0)
            items = raw_items + (["->"] if transition else [])
            if not raw_items:
                raise ValueError("Schedule event must name at least one tone set or off")
            schedule.append((float(time_value), items))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid SBG line {line_number}: {raw.strip()} ({exc})") from exc
    schedule.sort(key=lambda item: item[0])
    return tone_sets, schedule


def parse_sbg(path: str | Path):
    schedule_path = Path(path).expanduser().resolve()
    with schedule_path.open("r", encoding="latin-1") as handle:
        return parse_sbg_from_string(handle.read(), base_dir=schedule_path.parent)
