"""Simplified Python rewrite of SBAGEN for generating binaural beats.

This script is a minimal re-implementation inspired by the original SBAGEN
(Copyright (c) 1999-2011 Jim Peters). It supports basic binaural tone
generation and optional noise, writing the result to a WAV file.

This code is released under the terms of the GNU General Public License v2.
"""

import argparse
import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100


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
