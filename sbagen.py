#!/usr/bin/env python3
"""Python rewrite of SBAGEN.

The goal of this script is to provide a reasonably complete reimplementation of
SBAGEN's core features in Python.  It supports reading ``.sbg`` schedule files,
mixing multiple tone sets, adding noise or sound files and writing the result to
a WAV file.  The format implemented here is a subset of the original SBAGEN
syntax but is compatible with many of the example session files that ship with
SBAGEN.


This code is released under the terms of the GNU General Public License v2.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100
FADE_TIME = 1.0  # seconds


@dataclass
class ToneSpec:
    base: float
    beat: float
    amp: float

    def generate(self, duration: float) -> np.ndarray:
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        left = np.sin(2 * np.pi * self.base * t)
        right = np.sin(2 * np.pi * (self.base + self.beat) * t)
        stereo = np.vstack((left, right)).T * (self.amp / 100.0)
        return stereo


@dataclass
class NoiseSpec:
    amp: float

    def generate(self, duration: float) -> np.ndarray:
        return np.random.normal(scale=self.amp / 100.0, size=(int(SAMPLE_RATE * duration), 2))


@dataclass
class FileSpec:
    path: str
    amp: float

    def generate(self, duration: float) -> np.ndarray:
        data, rate = sf.read(self.path)
        if rate != SAMPLE_RATE:
            raise ValueError(f"Sample rate mismatch: {rate} != {SAMPLE_RATE}")
        if data.ndim == 1:
            data = np.stack([data, data], axis=1)
        if len(data) < int(duration * SAMPLE_RATE):
            repeat = int(np.ceil(int(duration * SAMPLE_RATE) / len(data)))
            data = np.tile(data, (repeat, 1))
        data = data[: int(duration * SAMPLE_RATE)]
        return data * (self.amp / 100.0)


ToneGenerator = Tuple[str, List]


def parse_tone_component(spec: str):
    if ':' in spec and not spec[0].isdigit():
        # Remove modifiers like spin:, slide:, pure: etc
        _, spec = spec.split(':', 1)
    if spec.startswith("pink") or spec.startswith("white"):
        _, amp = spec.split("/")
        return NoiseSpec(float(amp))
    if "/" in spec and os.path.isfile(spec.split("/")[0]):
        path, amp = spec.split("/")
        return FileSpec(path, float(amp))
    beat = 0.0
    amp = 100.0
    if '+' in spec:
        base, rest = spec.split('+', 1)
    elif '-' in spec:
        base, rest = spec.split('-', 1)
        beat = -float(rest.split('/')[0])
        if '/' in rest:
            beat_part, amp = rest.split('/')
            beat = -float(beat_part)
        return ToneSpec(float(base), beat, float(amp))
    else:
        if '/' in spec:
            base, amp = spec.split('/')
        else:
            base = spec
        return ToneSpec(float(base), beat, float(amp))

    if '/' in rest:
        beat_part, amp = rest.split('/')
        beat = float(beat_part)
    else:
        beat = float(rest)
    return ToneSpec(float(base), beat, float(amp))


def parse_sbg(path: str) -> Tuple[Dict[str, List], List[Tuple[float, List[str]]]]:
    tone_sets: Dict[str, List] = {}
    schedule: List[Tuple[float, List[str]]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line and not line[0].isdigit():
                label, rest = line.split(":", 1)
                parts = [p for p in rest.split() if p]
                tone_sets[label.strip()] = [parse_tone_component(p) for p in parts]
            else:
                time_str, rest = line.split(maxsplit=1)
                h, m = [int(x) for x in time_str.split(":")]
                t = h * 3600 + m * 60
                items = rest.split()
                schedule.append((t, items))
    schedule.sort(key=lambda x: x[0])
    return tone_sets, schedule


def mix_generators(generators: List, duration: float) -> np.ndarray:
    total = np.zeros((int(duration * SAMPLE_RATE), 2), dtype=np.float32)
    for gen in generators:
        total += gen.generate(duration)
    return total


def build_session(tone_sets: Dict[str, List], schedule: List[Tuple[float, List[str]]], duration: float) -> np.ndarray:
    if not schedule:
        return np.zeros((0, 2))

    segments = []
    times = [t for t, _ in schedule]
    if duration is None:
        duration = times[-1] - times[0]
    times.append(times[-1] + duration)  # end marker

    for idx, (start, names) in enumerate(schedule):
        end = times[idx + 1]
        seg_dur = max(0, end - start)
        gens = []
        if "off" not in names:
            for n in names:
                if n in tone_sets:
                    gens.extend(tone_sets[n])
        segment = mix_generators(gens, seg_dur + FADE_TIME)
        segments.append(segment)

    # cross-fade segments
    result = segments[0]
    for seg in segments[1:]:
        fade_len = int(FADE_TIME * SAMPLE_RATE)
        a = result.shape[0]
        result[a - fade_len :] = (
            result[a - fade_len :] * np.linspace(1, 0, fade_len)[:, None]
            + seg[:fade_len] * np.linspace(0, 1, fade_len)[:, None]
        )
        result = np.vstack([result[: a - fade_len], seg])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="SBAGEN compatible binaural beat generator")
    parser.add_argument("schedule", nargs="?", help=".sbg schedule file")
    parser.add_argument("--outfile", required=True, help="Output WAV file")
    parser.add_argument("--duration", type=float, help="Override session duration in seconds")
    parser.add_argument("--base", type=float, help="Quick tone base frequency")
    parser.add_argument("--beat", type=float, help="Quick tone beat frequency")
    parser.add_argument("--noise", action="store_true", help="Add white noise to quick tone")
    args = parser.parse_args()

    if args.schedule:
        tones, sched = parse_sbg(args.schedule)
        audio = build_session(tones, sched, args.duration)
    else:
        if args.base is None or args.beat is None or args.duration is None:
            parser.error("base, beat and duration required when not using schedule")
        gens = [ToneSpec(args.base, args.beat, 100.0)]
        if args.noise:
            gens.append(NoiseSpec(20.0))
        audio = mix_generators(gens, args.duration)

    sf.write(args.outfile, audio, SAMPLE_RATE)


def generate_tone(base_freq: float, beat_freq: float, duration: float) -> np.ndarray:
    """Generate a stereo binaural beat tone."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    left = np.sin(2 * np.pi * base_freq * t)
    right = np.sin(2 * np.pi * (base_freq + beat_freq) * t)
    return np.vstack((left, right)).T


def generate_noise(duration: float, volume: float) -> np.ndarray:
    """Return stereo white noise at the given volume."""
    samples = np.random.normal(scale=volume, size=(int(SAMPLE_RATE * duration), 2))
    return samples


def main():
    parser = argparse.ArgumentParser(description="Simple SBAGEN-like binaural beat generator")
    parser.add_argument("--base", type=float, default=200.0, help="Base frequency for left ear")
    parser.add_argument("--beat", type=float, default=10.0, help="Binaural beat difference")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds")
    parser.add_argument("--noise", action="store_true", help="Mix in white noise")
    parser.add_argument("--outfile", required=True, help="Output WAV file")
    args = parser.parse_args()

    tone = generate_tone(args.base, args.beat, args.duration)
    if args.noise:
        tone += generate_noise(args.duration, 0.2)
    sf.write(args.outfile, tone, SAMPLE_RATE)


if __name__ == "__main__":
    main()
