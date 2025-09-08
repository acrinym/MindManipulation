#!/usr/bin/env python3
"""Python rewrite of SBAGEN.

The goal of this script is to provide a reasonably complete reimplementation of
SBAGEN's core features in Python. It supports reading ``.sbg`` schedule files,
mixing multiple tone sets, adding noise or sound files and writing the result to
a WAV file. The format implemented here is a subset of the original SBAGEN
syntax but is compatible with many of the example session files that ship with
SBAGEN.

This code is released under the terms of the GNU General Public License v2.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100
FADE_TIME = 1.0  # seconds


# --- Data classes for different sound generators ---

@dataclass
class ToneSpec:
    """Represents a standard binaural beat tone."""
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
    """Represents pink or white noise."""
    amp: float
    # Note: This implementation uses white noise for both 'pink' and 'white'.
    # A proper pink noise generator would require filtering.
    def generate(self, duration: float) -> np.ndarray:
        return np.random.normal(scale=self.amp / 100.0, size=(int(SAMPLE_RATE * duration), 2))


@dataclass
class FileSpec:
    """Represents a sound file to be mixed."""
    path: str
    amp: float

    def generate(self, duration: float) -> np.ndarray:
        data, rate = sf.read(self.path)
        if rate != SAMPLE_RATE:
            # A more advanced version could resample here.
            raise ValueError(f"Sample rate mismatch for {self.path}: file is {rate}Hz, need {SAMPLE_RATE}Hz")
        if data.ndim == 1:
            data = np.stack([data, data], axis=1) # convert mono to stereo

        num_samples = int(duration * SAMPLE_RATE)
        if len(data) < num_samples:
            # Tile the audio if it's shorter than the required duration
            repeat = int(np.ceil(num_samples / len(data)))
            data = np.tile(data, (repeat, 1))
        
        return data[:num_samples] * (self.amp / 100.0)


@dataclass
class IsochronicSpec:
    """Represents an isochronic tone."""
    freq: float
    beat: float
    amp: float

    def generate(self, duration: float) -> np.ndarray:
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        tone = np.sin(2 * np.pi * self.freq * t)
        # Create a square wave to act as a gate
        gate = (np.sin(2 * np.pi * self.beat * t) > 0).astype(np.float32)
        mod = tone * gate
        stereo = np.vstack((mod, mod)).T * (self.amp / 100.0)
        return stereo


@dataclass
class HarmonicBoxSpec:
    """Represents a Harmonic Box X tone."""
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
        out *= (self.amp / 100.0) / len(phases) # Normalize amplitude
        return out

# Type hint for any of our sound generator classes
AnySpec = Union[ToneSpec, NoiseSpec, FileSpec, IsochronicSpec, HarmonicBoxSpec]


def parse_tone_component(spec: str) -> Optional[AnySpec]:
    """Parses a single component string from an SBG file into a generator spec."""
    spec = spec.strip()
    if spec.lower() in {"-", "off"}:
        return None

    if ':' in spec and not spec[0].isdigit():
        # Handle special prefixes like 'iso:', 'hbox:', 'spin:', etc.
        prefix, rest = spec.split(':', 1)
        prefix = prefix.lower()

        if prefix == "iso":
            params_str, amp_str = rest.split('/') if '/' in rest else (rest, '100')
            freq, beat = [float(x) for x in params_str.split(',')]
            return IsochronicSpec(freq, beat, float(amp_str))
        
        if prefix == "hbox":
            params_str, amp_str = rest.split('/') if '/' in rest else (rest, '100')
            base, diff, mod = [float(x) for x in params_str.split(',')]
            return HarmonicBoxSpec(base, diff, mod, float(amp_str))
        
        # For other unsupported prefixes (spin, slide), just use the core spec.
        spec = rest

    if spec.lower().startswith("pink") or spec.lower().startswith("white"):
        _, amp_str = spec.split("/")
        return NoiseSpec(float(amp_str))

    if "/" in spec and os.path.isfile(spec.split("/")[0]):
        path, amp_str = spec.split("/")
        return FileSpec(path, float(amp_str))

    # Default case: standard binaural tone (e.g., "200+10/50")
    amp = 100.0
    core_spec = spec
    if '/' in spec:
        core_spec, amp_str = spec.rsplit('/', 1)
        amp = float(amp_str)

    beat = 0.0
    if '+' in core_spec:
        base_str, beat_str = core_spec.split('+', 1)
        beat = float(beat_str)
    elif '-' in core_spec and core_spec.count('-') == 1 and not core_spec.startswith('-'):
        # Handle subtraction for beat frequency, but avoid negative base frequencies
        base_str, beat_str = core_spec.split('-', 1)
        beat = -float(beat_str)
    else:
        base_str = core_spec

    return ToneSpec(float(base_str), beat, amp)


def parse_sbg(path: str) -> Tuple[Dict[str, List[AnySpec]], List[Tuple[float, List[str]]]]:
    """Parses an .sbg file into tone sets and a schedule."""
    tone_sets: Dict[str, List[AnySpec]] = {}
    schedule: List[Tuple[float, List[str]]] = []
    
    # SBAGEN example files may use Windows-1252, so use latin-1 to avoid errors.
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if ":" in line and not line[0].isdigit() and not line.startswith("+"):
                # This line defines a tone set (e.g., "alpha: 200+10 pink/10")
                label, rest = line.split(":", 1)
                parts = [p for p in rest.split() if p]
                components = [comp for p in parts if (comp := parse_tone_component(p)) is not None]
                tone_sets[label.strip()] = components
            else:
                # This line is a schedule entry (e.g., "0:05 alpha -> beta")
                if line.upper().startswith("NOW"):
                    time_val = 0
                    rest = line[3:].strip()
                else:
                    time_str, rest = line.split(maxsplit=1)
                    if time_str.startswith("+"):
                        time_str = time_str[1:]
                    
                    t_parts = [int(x) for x in time_str.split(":")]
                    if len(t_parts) == 3:
                        h, m, s = t_parts
                    elif len(t_parts) == 2:
                        h, m, s = 0, t_parts[0], t_parts[1]
                    else:
                        h, m, s = 0, 0, t_parts[0]
                    time_val = h * 3600 + m * 60 + s
                
                items = [item for item in rest.replace("->", " ").split() if item]
                schedule.append((time_val, items))
                
    schedule.sort(key=lambda x: x[0])
    return tone_sets, schedule


def mix_generators(generators: List[AnySpec], duration: float) -> np.ndarray:
    """Mixes the output of several generators for a given duration."""
    total = np.zeros((int(duration * SAMPLE_RATE), 2), dtype=np.float32)
    if not generators or duration <= 0:
        return total

    for gen in generators:
        total += gen.generate(duration)
    return total


def build_session(tone_sets: Dict[str, List[AnySpec]], schedule: List[Tuple[float, List[str]]], duration: Optional[float]) -> np.ndarray:
    """Builds a full audio session from a schedule by mixing and cross-fading segments."""
    if not schedule:
        return np.zeros((0, 2))

    segments = []
    times = [t for t, _ in schedule]

    if duration is None:
        duration = times[-1]

    # Add a final event to mark the end of the last segment
    schedule.append((duration, ["off"]))
    
    active_generators = []
    last_time = 0.0

    for i, (start_time, names) in enumerate(schedule):
        seg_dur = start_time - last_time
        if seg_dur > 0:
            segment = mix_generators(active_generators, seg_dur)
            segments.append(segment)

        # Update the list of active generators based on the current schedule event
        is_absolute = not names[0].startswith('+') and not names[0].startswith('-')
        if is_absolute:
            active_generators.clear()
        
        off_tokens = {n.lower() for n in names}
        if "alloff" in off_tokens:
             active_generators.clear()

        for name in names:
            if name.lower() in {"-", "off", "alloff"}:
                continue
            
            clean_name = name.lstrip('+-')
            if name.startswith('-'): # Remove tone set
                active_generators = [gen for gen in active_generators if gen not in tone_sets.get(clean_name, [])]
            else: # Add tone set
                active_generators.extend(tone_sets.get(clean_name, []))
        
        last_time = start_time

    # Combine all generated segments with cross-fading
    if not segments:
        return np.zeros((0, 2))

    fade_len = int(FADE_TIME * SAMPLE_RATE)
    output = []
    for i, seg in enumerate(segments):
        if i > 0 and len(seg) >= fade_len and len(output) >= fade_len:
            # Create fade-out ramp
            output[-fade_len:] *= np.linspace(1, 0, fade_len)[:, None]
            # Create fade-in ramp and add
            output[-fade_len:] += seg[:fade_len] * np.linspace(0, 1, fade_len)[:, None]
            # Append the rest of the new segment
            output.append(seg[fade_len:])
        else:
            output.append(seg)

    return np.vstack(output)


def main() -> None:
    """Main function to parse arguments and generate the audio file."""
    parser = argparse.ArgumentParser(description="SBAGEN compatible binaural beat generator")
    parser.add_argument("schedule", nargs="?", help=".sbg schedule file to process")
    parser.add_argument("-o", "--outfile", required=True, help="Output WAV file")
    parser.add_argument("-d", "--duration", type=float, help="Override total session duration in seconds")

    # Quick-generate options (if no schedule file is provided)
    parser.add_argument("--base", type=float, help="Quick tone: base frequency (e.g., 200)")
    parser.add_argument("--beat", type=float, help="Quick tone: beat frequency (e.g., 10)")
    parser.add_argument("--noise", type=float, metavar="AMP", help="Quick tone: add white noise with amplitude (0-100)")
    parser.add_argument("--isochronic", nargs=2, metavar=("FREQ", "BEAT"), type=float,
                        help="Quick tone: generate an isochronic tone")
    parser.add_argument("--harmonic-box", nargs=3, metavar=("BASE", "DIFF", "MOD"), type=float,
                        help="Quick tone: generate a Harmonic Box X tone")

    # Background music options
    parser.add_argument("--music", help="Background WAV file to mix in")
    parser.add_argument("--music-amp", type=float, default=100.0, help="Volume percent for background music (0-100)")

    args = parser.parse_args()

    audio = None
    if args.schedule:
        if not os.path.exists(args.schedule):
            parser.error(f"Schedule file not found: {args.schedule}")
        tones, sched = parse_sbg(args.schedule)
        audio = build_session(tones, sched, args.duration)

        # Mix in background music if provided
        if args.music:
            dur = len(audio) / SAMPLE_RATE
            music_gen = FileSpec(args.music, args.music_amp)
            music_track = music_gen.generate(dur)
            audio = audio + music_track[:len(audio)]
    else:
        # Handle quick-generate options
        if args.duration is None:
            parser.error("--duration is required when not using a schedule file.")
        
        gens: List[AnySpec] = []
        if args.harmonic_box:
            base, diff, mod = args.harmonic_box
            gens.append(HarmonicBoxSpec(base, diff, mod, 100.0))
        elif args.isochronic:
            freq, beat = args.isochronic
            gens.append(IsochronicSpec(freq, beat, 100.0))
        elif args.base is not None and args.beat is not None:
            gens.append(ToneSpec(args.base, args.beat, 100.0))
        
        if args.noise is not None:
            gens.append(NoiseSpec(args.noise))
        if args.music:
            gens.append(FileSpec(args.music, args.music_amp))
            
        if not gens:
            parser.error("You must specify a tone to generate (e.g., --base and --beat) if not using a schedule.")
            
        audio = mix_generators(gens, args.duration)

    if audio is not None:
        print(f"Writing {len(audio) / SAMPLE_RATE:.2f} seconds of audio to {args.outfile}...")
        # Normalize to prevent clipping before writing
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio /= max_val
        sf.write(args.outfile, audio, SAMPLE_RATE)
        print("Done.")
    else:
        print("No audio generated.")

if __name__ == "__main__":
    main()
