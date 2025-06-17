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
from typing import Dict, List, Tuple, Optional

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


@dataclass
class IsochronicSpec:
    freq: float
    beat: float
    amp: float

    def generate(self, duration: float) -> np.ndarray:
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        tone = np.sin(2 * np.pi * self.freq * t)
        gate = (np.sin(2 * np.pi * self.beat * t) > 0).astype(np.float32)
        mod = tone * gate
        stereo = np.vstack((mod, mod)).T * (self.amp / 100.0)
        return stereo


@dataclass
class HarmonicBoxSpec:
    base: float
    diff: float
    mod: float
    amp: float

    def generate(self, duration: float) -> np.ndarray:
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        phases = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        left = np.zeros_like(t)
        right = np.zeros_like(t)
        for ph in phases:
            gate = (np.sin(2 * np.pi * self.mod * t + ph) > 0).astype(np.float32)
            left += np.sin(2 * np.pi * self.base * t) * gate
            right += np.sin(2 * np.pi * (self.base + self.diff) * t) * gate
        out = np.vstack((left, right)).T
        out *= (self.amp / 100.0) / len(phases)
        return out


ToneGenerator = Tuple[str, List]


def parse_tone_component(spec: str):
    if spec.strip() in {"-", "off", "OFF"}:
        return None
    if ':' in spec and not spec[0].isdigit():
        # Remove modifiers like spin:, slide:, pure: etc
        prefix, spec = spec.split(':', 1)
        if prefix == "iso":
            if '/' in spec:
                params, amp = spec.split('/')
            else:
                params, amp = spec, '100'
            freq, beat = [float(x) for x in params.split(',')]
            return IsochronicSpec(freq, beat, float(amp))
        if prefix == "hbox":
            if '/' in spec:
                params, amp = spec.split('/')
            else:
                params, amp = spec, '100'
            base, diff, mod = [float(x) for x in params.split(',')]
            return HarmonicBoxSpec(base, diff, mod, float(amp))
        # fallback for other modifiers
        spec = spec
    if spec.startswith("pink") or spec.startswith("white"):
        _, amp = spec.split("/")
        return NoiseSpec(float(amp))

    if "/" in spec and os.path.isfile(spec.split("/")[0]):
        path, amp = spec.split("/")
        return FileSpec(path, float(amp))

    amp = 100.0
    core = spec
    if '/' in spec:
        core, amp_str = spec.rsplit('/', 1)
        amp = float(amp_str)

    beat = 0.0
    if '+' in core:
        base_str, beat_str = core.split('+', 1)
        beat = float(beat_str)
    elif '-' in core:
        base_str, beat_str = core.split('-', 1)
        beat = -float(beat_str)
    else:
        base_str = core

    return ToneSpec(float(base_str), beat, amp)


def parse_sbg(path: str) -> Tuple[Dict[str, List], List[Tuple[float, List[str]]]]:
    tone_sets: Dict[str, List] = {}
    schedule: List[Tuple[float, List[str]]] = []
    # SBAGEN example files may use Windows-1252 encoding, so use latin-1 to
    # avoid decoding errors while still returning str objects.
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line and not line[0].isdigit() and not line.startswith("+"):
                label, rest = line.split(":", 1)
                parts = [p for p in rest.split() if p]
                comps = [parse_tone_component(p) for p in parts]
                comps = [c for c in comps if c is not None]
                tone_sets[label.strip()] = comps
            else:
                if line.startswith("NOW"):
                    time_val = 0
                    rest = line[3:].strip()
                else:
                    time_str, rest = line.split(maxsplit=1)
                    if time_str.startswith("+"):
                        time_str = time_str[1:]
                    tparts = [int(x) for x in time_str.split(":")]
                    if len(tparts) == 3:
                        h, m, s = tparts
                    else:
                        h, m = tparts
                        s = 0
                    time_val = h * 3600 + m * 60 + s
                items = [i for i in rest.replace("->", "").split() if i]
                schedule.append((time_val, items))
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
        off_tokens = {n.lower() for n in names}
        if "off" not in off_tokens and "alloff" not in off_tokens:
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
    parser.add_argument("--isochronic", nargs=2, metavar=("FREQ", "BEAT"), type=float,
                        help="Generate isochronic tone with base frequency and beat rate")
    parser.add_argument("--harmonic-box", nargs=3, metavar=("BASE", "DIFF", "MOD"), type=float,
                        help="Generate Harmonic Box X with base frequency, binaural difference and modulation rate")
    parser.add_argument("--music", help="Background WAV file to mix")
    parser.add_argument("--music-amp", type=float, default=100.0, help="Volume percent for background music")
    args = parser.parse_args()

    if args.schedule:
        tones, sched = parse_sbg(args.schedule)
        audio = build_session(tones, sched, args.duration)
        dur = len(audio) / SAMPLE_RATE
        if args.music:
            music = FileSpec(args.music, args.music_amp).generate(dur)
            audio = audio + music[: len(audio)]
    else:
        gens: List = []
        duration = args.duration
        if args.harmonic_box:
            base, diff, mod = args.harmonic_box
            gens.append(HarmonicBoxSpec(base, diff, mod, 100.0))
        elif args.isochronic:
            freq, beat = args.isochronic
            gens.append(IsochronicSpec(freq, beat, 100.0))
        else:
            if args.base is None or args.beat is None or args.duration is None:
                parser.error("base, beat and duration required when not using schedule")
            gens.append(ToneSpec(args.base, args.beat, 100.0))
        if args.noise:
            gens.append(NoiseSpec(20.0))
        if args.music:
            gens.append(FileSpec(args.music, args.music_amp))
        if duration is None:
            parser.error("duration required")
        audio = mix_generators(gens, duration)

    sf.write(args.outfile, audio, SAMPLE_RATE)


if __name__ == "__main__":
    main()
