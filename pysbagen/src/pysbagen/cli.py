import argparse, os, numpy as np, soundfile as sf
from .parser import parse_sbg
from .mixer import build_session_generator
from .generators.binaural import ToneSpec
from .generators.noise import NoiseSpec
from .generators.file import FileSpec
from .generators.isochronic import IsochronicSpec
from .generators.harmonic_box import HarmonicBoxSpec

SR = 44100

def main():
    p = argparse.ArgumentParser(description="SBaGen-compatible generator (Python)")
    p.add_argument("schedule", nargs="?", help=".sbg schedule file")
    p.add_argument("-o","--outfile", required=True)
    p.add_argument("-d","--duration", type=float)
    p.add_argument("--base", type=float)
    p.add_argument("--beat", type=float)
    p.add_argument("--noise", type=float, metavar="AMP")
    p.add_argument("--noise-kind", default="white", choices=["white","pink"])
    p.add_argument("--isochronic", nargs=2, metavar=("FREQ","BEAT"), type=float)
    p.add_argument("--harmonic-box", nargs=3, metavar=("BASE","DIFF","MOD"), type=float)
    p.add_argument("--music", help="Background file")
    p.add_argument("--music-amp", type=float, default=100.0)
    args = p.parse_args()

    if args.schedule:
        tones, sched = parse_sbg(args.schedule)
        gen = build_session_generator(tones, sched, args.duration)
    else:
        if args.duration is None:
            p.error("--duration is required without a schedule")
        gens = []
        if args.base is not None and args.beat is not None:
            gens.append(ToneSpec(base=args.base, beat=args.beat, amp=100.0))
        if args.isochronic is not None:
            gens.append(IsochronicSpec(freq=args.isochronic[0], beat=args.isochronic[1], amp=100.0))
        if args.harmonic_box is not None:
            gens.append(HarmonicBoxSpec(base=args.harmonic_box[0], diff=args.harmonic_box[1], mod=args.harmonic_box[2], amp=100.0))
        if args.noise is not None:
            gens.append(NoiseSpec(amp=args.noise, kind=args.noise_kind))
        if args.music:
            gens.append(FileSpec(path=args.music, amp=args.music_amp))
        if not gens:
            p.error("No generators specified.")
        from .mixer import mix_generators
        gen = mix_generators(gens, args.duration)

    # stream -> buffer -> write
    chunks = [c for c,_ in gen]
    if not chunks:
        print("No audio generated.")
        return
    audio = np.vstack(chunks)
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio /= peak
    sf.write(args.outfile, audio, SR)
    print(f"Wrote {len(audio)/SR:.2f}s to {args.outfile}")
