#!/usr/bin/env python3
"""
Python rewrite of SBAGEN - Command-Line Interface.

This script provides the command-line functionality for generating SBAGEN
sessions. It uses the core logic from the `audio_engine` module and handles
user arguments for file-based generation and quick-tone creation.

This code is released under the terms of the GNU General Public License v2.
"""

import argparse
import os
import numpy as np
import soundfile as sf

# Import the core audio generation logic
from audio_engine import (
    SAMPLE_RATE,
    AnySpec,
    FileSpec,
    HarmonicBoxSpec,
    IsochronicSpec,
    NoiseSpec,
    ToneSpec,
    parse_sbg,
    build_session,
    mix_generators
)


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
        
        gens: list[AnySpec] = []
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
