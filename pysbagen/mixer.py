from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Tuple

import numpy as np

FRAME = 1024
SR = 44100


def _stereo_float32(chunk: np.ndarray) -> np.ndarray:
    array = np.asarray(chunk, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Audio chunks must be one- or two-dimensional, got {array.shape}")
    if array.shape[1] == 1:
        array = np.repeat(array, 2, axis=1)
    elif array.shape[1] > 2:
        array = array[:, :2]
    return array


@dataclass
class _StreamState:
    iterator: Iterator
    pending: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    info: dict | None = None
    exhausted: bool = False

    def read(self, frame_count: int) -> tuple[np.ndarray, dict | None, bool]:
        pieces: list[np.ndarray] = []
        collected = 0
        produced_audio = False

        while collected < frame_count:
            if len(self.pending):
                take = min(frame_count - collected, len(self.pending))
                pieces.append(self.pending[:take])
                self.pending = self.pending[take:]
                collected += take
                produced_audio = True
                continue

            if self.exhausted:
                break

            try:
                chunk, info = next(self.iterator)
            except StopIteration:
                self.exhausted = True
                continue

            normalized = _stereo_float32(chunk)
            if len(normalized) == 0:
                continue
            self.pending = normalized
            self.info = info

        output = np.zeros((frame_count, 2), dtype=np.float32)
        if pieces:
            combined = np.vstack(pieces)
            output[: len(combined)] = combined
        return output, self.info if produced_audio else None, produced_audio


def mix_generators(
    gens: Iterable, duration: float
) -> Iterator[Tuple[np.ndarray, list[dict]]]:
    """Mix generators for exactly ``duration`` seconds, padding exhausted streams with silence."""
    if duration <= 0:
        return

    specs = list(gens)
    for spec in specs:
        sample_rate = getattr(spec, "sample_rate", SR)
        if sample_rate != SR:
            raise ValueError(f"Generator sample rate {sample_rate} does not match mixer rate {SR}")

    states = [_StreamState(iter(spec.generator(duration))) for spec in specs]
    total_frames = int(SR * duration)

    for offset in range(0, total_frames, FRAME):
        frame_count = min(FRAME, total_frames - offset)
        accumulator = np.zeros((frame_count, 2), dtype=np.float32)
        infos: List[dict] = []

        for state in states:
            chunk, info, produced_audio = state.read(frame_count)
            if produced_audio:
                accumulator += chunk
                if info is not None:
                    infos.append(info)

        peak = float(np.max(np.abs(accumulator))) if len(accumulator) else 0.0
        if peak > 1.0:
            accumulator /= peak
        yield accumulator, infos


def _apply_schedule_event(active: list, tone_sets: dict, names: list[str]) -> list:
    if not names:
        return active

    if not names[0].startswith(("+", "-")):
        active = []

    for token in names:
        lowered = token.lower()
        if lowered in {"-", "off", "alloff"}:
            if lowered == "alloff":
                active = []
            continue

        operation = token[0] if token[0] in "+-" else "+"
        name = token.lstrip("+-")
        if name not in tone_sets:
            raise ValueError(f"Schedule references unknown tone set: {name}")
        selected = tone_sets[name]
        if operation == "-":
            active = [generator for generator in active if generator not in selected]
        else:
            active.extend(selected)
    return active


def build_session_generator(tone_sets, schedule, duration=None):
    """Render a schedule without collapsing silent gaps or overrunning an explicit duration."""
    if not schedule:
        return

    ordered = sorted(schedule, key=lambda item: item[0])
    if duration is None:
        duration = float(ordered[-1][0])
    if duration < 0:
        raise ValueError("duration must be non-negative")

    active: list = []
    cursor = 0.0

    for start, names in ordered:
        start = float(start)
        if start < cursor:
            raise ValueError("Schedule times must be non-decreasing")
        if start > duration:
            break

        segment_duration = start - cursor
        if segment_duration > 0:
            yield from mix_generators(active, segment_duration)

        active = _apply_schedule_event(active, tone_sets, names)
        cursor = start

    if cursor < duration:
        yield from mix_generators(active, duration - cursor)
