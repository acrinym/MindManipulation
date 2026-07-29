from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, List, Tuple

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
    spec: object
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


def _new_state(spec, duration: float) -> _StreamState:
    sample_rate = getattr(spec, "sample_rate", SR)
    if sample_rate != SR:
        raise ValueError(f"Generator sample rate {sample_rate} does not match mixer rate {SR}")
    return _StreamState(spec=spec, iterator=iter(spec.generator(max(duration, 0.0))))


def _read_states(states: list[_StreamState], frame_count: int) -> tuple[np.ndarray, list[dict]]:
    accumulator = np.zeros((frame_count, 2), dtype=np.float32)
    infos: list[dict] = []
    for state in states:
        chunk, info, produced_audio = state.read(frame_count)
        if produced_audio:
            accumulator += chunk
            if info is not None:
                infos.append(info)
    return accumulator, infos


def _mix_state_groups_frames(
    groups: list[tuple[list[_StreamState], Callable[[np.ndarray], np.ndarray]]],
    total_frames: int,
) -> Iterator[Tuple[np.ndarray, list[dict]]]:
    if total_frames <= 0:
        return

    for offset in range(0, total_frames, FRAME):
        frame_count = min(FRAME, total_frames - offset)
        progress = (offset + np.arange(frame_count, dtype=np.float64)) / max(total_frames - 1, 1)
        accumulator = np.zeros((frame_count, 2), dtype=np.float32)
        infos: list[dict] = []
        for states, gain_fn in groups:
            chunk, group_infos = _read_states(states, frame_count)
            gain = np.asarray(gain_fn(progress), dtype=np.float32)
            if gain.ndim == 0:
                gain = np.full(frame_count, float(gain), dtype=np.float32)
            accumulator += chunk * gain[:, None]
            infos.extend(group_infos)

        peak = float(np.max(np.abs(accumulator))) if len(accumulator) else 0.0
        if peak > 1.0:
            accumulator /= peak
        yield accumulator, infos


def _mix_state_groups(
    groups: list[tuple[list[_StreamState], Callable[[np.ndarray], np.ndarray]]],
    duration: float,
) -> Iterator[Tuple[np.ndarray, list[dict]]]:
    yield from _mix_state_groups_frames(groups, int(SR * duration))


def _constant_gain(value: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda progress: np.full_like(progress, value, dtype=np.float64)


def mix_generators(
    gens: Iterable, duration: float
) -> Iterator[Tuple[np.ndarray, list[dict]]]:
    """Mix generators for exactly ``duration`` seconds, padding exhausted streams with silence."""
    states = [_new_state(spec, duration) for spec in list(gens)]
    yield from _mix_state_groups([(states, _constant_gain(1.0))], duration)


def _schedule_tokens(names: list[str]) -> tuple[list[str], bool]:
    transition = bool(names and names[-1] == "->")
    tokens = names[:-1] if transition else names
    return tokens, transition


def _apply_schedule_event(active: list, tone_sets: dict, names: list[str]) -> list:
    names, _ = _schedule_tokens(names)
    if not names:
        return active

    first_lower = names[0].lower()
    if first_lower in {"-", "off", "alloff"} or not names[0].startswith(("+", "-")):
        active = []

    for token in names:
        lowered = token.lower()
        if lowered in {"-", "off", "alloff"}:
            if lowered in {"-", "alloff"}:
                active = []
            continue

        operation = token[0] if token[0] in "+-" else "+"
        name = token.lstrip("+-")
        if name not in tone_sets:
            raise ValueError(f"Schedule references unknown tone set: {name}")
        selected = tone_sets[name]
        if operation == "-":
            selected_ids = {id(generator) for generator in selected}
            active = [generator for generator in active if id(generator) not in selected_ids]
        else:
            active.extend(selected)
    return active


def _reconcile_states(
    existing: list[_StreamState],
    specs: list,
    remaining_duration: float,
) -> list[_StreamState]:
    by_identity = {id(state.spec): state for state in existing}
    return [
        by_identity.get(id(spec)) or _new_state(spec, remaining_duration)
        for spec in specs
    ]


def _transition_groups(
    source_states: list[_StreamState],
    target_specs: list,
    remaining_duration: float,
) -> tuple[list[tuple[list[_StreamState], Callable[[np.ndarray], np.ndarray]]], list[_StreamState]]:
    source_by_id = {id(state.spec): state for state in source_states}
    target_ids = {id(spec) for spec in target_specs}
    common = [state for state in source_states if id(state.spec) in target_ids]
    source_only = [state for state in source_states if id(state.spec) not in target_ids]
    target_only = [
        _new_state(spec, remaining_duration)
        for spec in target_specs
        if id(spec) not in source_by_id
    ]
    target_by_id = {id(state.spec): state for state in common + target_only}
    final_states = [target_by_id[id(spec)] for spec in target_specs]

    groups: list[tuple[list[_StreamState], Callable[[np.ndarray], np.ndarray]]] = []
    if common:
        groups.append((common, _constant_gain(1.0)))
    if source_only:
        groups.append((source_only, lambda progress: 1.0 - progress))
    if target_only:
        groups.append((target_only, lambda progress: progress))
    if not groups:
        groups.append(([], _constant_gain(1.0)))
    return groups, final_states


def build_session_generator(tone_sets, schedule, duration=None):
    """Render schedules with persistent streams, real silence, and full-interval ``->`` crossfades."""
    if not schedule:
        return

    ordered = sorted(schedule, key=lambda item: item[0])
    if duration is None:
        duration = float(ordered[-1][0])
    duration = float(duration)
    if duration < 0:
        raise ValueError("duration must be non-negative")
    duration_frames = int(SR * duration)

    event_frames = [(int(SR * float(time_value)), names) for time_value, names in ordered]
    first_frame = event_frames[0][0]
    if first_frame < 0:
        raise ValueError("Schedule times must be non-negative")
    if first_frame > 0:
        yield from _mix_state_groups_frames(
            [([], _constant_gain(1.0))], min(first_frame, duration_frames)
        )
    if first_frame >= duration_frames:
        return

    active_specs = _apply_schedule_event([], tone_sets, event_frames[0][1])
    active_states = _reconcile_states(
        [], active_specs, (duration_frames - first_frame) / SR
    )

    for index, (start_frame, names) in enumerate(event_frames):
        if index and start_frame < event_frames[index - 1][0]:
            raise ValueError("Schedule times must be non-decreasing")
        if start_frame >= duration_frames:
            break
        next_frame = duration_frames
        if index + 1 < len(event_frames):
            next_frame = min(event_frames[index + 1][0], duration_frames)
        segment_frames = max(0, next_frame - start_frame)
        _, transition = _schedule_tokens(names)

        if transition and index + 1 < len(event_frames) and segment_frames > 0:
            target_specs = _apply_schedule_event(
                list(active_specs), tone_sets, event_frames[index + 1][1]
            )
            groups, final_states = _transition_groups(
                active_states,
                target_specs,
                (duration_frames - start_frame) / SR,
            )
            yield from _mix_state_groups_frames(groups, segment_frames)
            active_specs = target_specs
            active_states = final_states
        else:
            yield from _mix_state_groups_frames(
                [(active_states, _constant_gain(1.0))], segment_frames
            )
            if index + 1 < len(event_frames) and next_frame < duration_frames:
                active_specs = _apply_schedule_event(
                    list(active_specs), tone_sets, event_frames[index + 1][1]
                )
                active_states = _reconcile_states(
                    active_states,
                    active_specs,
                    (duration_frames - next_frame) / SR,
                )
