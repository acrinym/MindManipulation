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

    def generator(self, duration: float):
        num_samples = int(SAMPLE_RATE * duration)
        for i in range(0, num_samples, 1024):
            chunk_size = min(1024, num_samples - i)
            t = np.linspace(i / SAMPLE_RATE, (i + chunk_size) / SAMPLE_RATE, chunk_size, endpoint=False)
            left = np.sin(2 * np.pi * self.base * t)
            right = np.sin(2 * np.pi * (self.base + self.beat) * t)
            stereo = np.vstack((left, right)).T * (self.amp / 100.0)
            yield stereo, {"type": "binaural", "base": self.base, "beat": self.beat}


@dataclass
class NoiseSpec:
    """Represents pink or white noise."""
    amp: float
    # Note: This implementation uses white noise for both 'pink' and 'white'.
    # A proper pink noise generator would require filtering.
    def generator(self, duration: float):
        num_samples = int(SAMPLE_RATE * duration)
        for i in range(0, num_samples, 1024):
            chunk_size = min(1024, num_samples - i)
            chunk = np.random.normal(scale=self.amp / 100.0, size=(chunk_size, 2))
            yield chunk, {"type": "noise", "amp": self.amp}


from pydub import AudioSegment

@dataclass
class FileSpec:
    """Represents a sound file to be mixed."""
    path: str
    amp: float

    def generator(self, duration: float):
        if self.path.lower().endswith(".mp3"):
            audio = AudioSegment.from_mp3(self.path)
            if audio.frame_rate != SAMPLE_RATE:
                audio = audio.set_frame_rate(SAMPLE_RATE)
            if audio.channels == 1:
                audio = audio.set_channels(2)
            # Convert to numpy array
            data = np.array(audio.get_array_of_samples()).reshape(-1, 2)
            # Normalize to float
            data = data / (2**(audio.sample_width * 8 - 1))
        else:
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
        
        data = data[:num_samples] * (self.amp / 100.0)

        for i in range(0, len(data), 1024):
            chunk = data[i:i+1024]
            yield chunk, {"type": "file", "path": self.path}


@dataclass
class IsochronicSpec:
    """Represents an isochronic tone."""
    freq: float
    beat: float
    amp: float

    def generator(self, duration: float):
        num_samples = int(SAMPLE_RATE * duration)
        for i in range(0, num_samples, 1024):
            chunk_size = min(1024, num_samples - i)
            t = np.linspace(i / SAMPLE_RATE, (i + chunk_size) / SAMPLE_RATE, chunk_size, endpoint=False)
            tone = np.sin(2 * np.pi * self.freq * t)
            # Create a square wave to act as a gate
            gate = (np.sin(2 * np.pi * self.beat * t) > 0).astype(np.float32)
            mod = tone * gate
            stereo = np.vstack((mod, mod)).T * (self.amp / 100.0)
            yield stereo, {"type": "isochronic", "freq": self.freq, "beat": self.beat}


@dataclass
class HarmonicBoxSpec:
    """Represents a Harmonic Box X tone."""
    base: float
    diff: float
    mod: float
    amp: float

    def generator(self, duration: float):
        num_samples = int(SAMPLE_RATE * duration)
        for i in range(0, num_samples, 1024):
            chunk_size = min(1024, num_samples - i)
            t = np.linspace(i / SAMPLE_RATE, (i + chunk_size) / SAMPLE_RATE, chunk_size, endpoint=False)
            phases = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
            left = np.zeros_like(t)
            right = np.zeros_like(t)
            for ph in phases:
                gate = (np.sin(2 * np.pi * self.mod * t + ph) > 0).astype(np.float32)
                left += np.sin(2 * np.pi * self.base * t) * gate
                right += np.sin(2 * np.pi * (self.base + self.diff) * t) * gate
            out = np.vstack((left, right)).T
            out *= (self.amp / 100.0) / len(phases) # Normalize amplitude
            yield out, {"type": "harmonic_box", "base": self.base, "diff": self.diff, "mod": self.mod}

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


def mix_generators(generators: List[AnySpec], duration: float):
    """Mixes the output of several generators for a given duration."""
    if not generators or duration <= 0:
        return

    gens = [g.generator(duration) for g in generators]

    while True:
        try:
            mixed_chunk = np.zeros((1024, 2), dtype=np.float32)
            freq_info = []

            for g in gens:
                chunk, info = next(g)
                # Ensure chunks are the same size for mixing
                if len(chunk) < len(mixed_chunk):
                    padded_chunk = np.zeros_like(mixed_chunk)
                    padded_chunk[:len(chunk)] = chunk
                    mixed_chunk += padded_chunk
                else:
                    mixed_chunk += chunk
                freq_info.append(info)

            yield mixed_chunk, freq_info
        except StopIteration:
            break


def build_session_generator(tone_sets: Dict[str, List[AnySpec]], schedule: List[Tuple[float, List[str]]], duration: Optional[float]):
    """A generator that yields chunks of audio and frequency information from a schedule."""
    if not schedule:
        return

    times = [t for t, _ in schedule]
    if duration is None:
        duration = times[-1]

    schedule.append((duration, ["off"]))
    
    active_generators = []
    last_time = 0.0

    for i, (start_time, names) in enumerate(schedule):
        seg_dur = start_time - last_time
        if seg_dur > 0:
            yield from mix_generators(active_generators, seg_dur)

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
            if name.startswith('-'):
                active_generators = [gen for gen in active_generators if gen not in tone_sets.get(clean_name, [])]
            else:
                active_generators.extend(tone_sets.get(clean_name, []))
        
        last_time = start_time


def generate_audio(args: argparse.Namespace) -> Optional[np.ndarray]:
    """Generates audio based on command line arguments, returns the raw audio data."""
    if args.schedule:
        if not os.path.exists(args.schedule):
            raise FileNotFoundError(f"Schedule file not found: {args.schedule}")
        tones, sched = parse_sbg(args.schedule)
        audio_generator = build_session_generator(tones, sched, args.duration)
    else:
        if args.duration is None:
            raise ValueError("--duration is required when not using a schedule file.")

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
            raise ValueError("You must specify a tone to generate (e.g., --base and --beat) if not using a schedule.")

        audio_generator = mix_generators(gens, args.duration)

    if audio_generator is None:
        return None

    audio_chunks = [chunk for chunk, info in audio_generator]
    if not audio_chunks:
        return None

    audio = np.vstack(audio_chunks)

    # Note: background music is not yet handled in the generator pipeline
    # This would require a more complex mixing strategy.
    # For now, we add it to the final generated audio.
    if args.music:
        dur = len(audio) / SAMPLE_RATE
        music_gen = FileSpec(args.music, args.music_amp)
        # This is not ideal as it reads the whole file again
        music_track = next(music_gen.generator(dur))[0]
        # This part needs more work to properly mix chunk by chunk
        # For now, just adding it to the beginning for testing
        if len(music_track) > len(audio):
            music_track = music_track[:len(audio)]
        audio[:len(music_track)] += music_track

    return audio

def generate_audio_and_viz(args: argparse.Namespace):
    """A generator that yields audio chunks and frequency info for real-time processing."""
    if args.schedule:
        if not os.path.exists(args.schedule):
            raise FileNotFoundError(f"Schedule file not found: {args.schedule}")
        tones, sched = parse_sbg(args.schedule)
        yield from build_session_generator(tones, sched, args.duration)
    else:
        if args.duration is None:
            raise ValueError("--duration is required when not using a schedule file.")
        
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
            raise ValueError("You must specify a tone to generate (e.g., --base and --beat) if not using a schedule.")
            
        yield from mix_generators(gens, args.duration)


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

    try:
        audio = generate_audio(args)
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
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
