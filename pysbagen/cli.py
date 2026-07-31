from __future__ import annotations

import argparse

from .api import build_quick_specs, render_schedule, render_specs, write_audio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SBaGen-compatible generator (Python)")
    parser.add_argument("schedule", nargs="?", help=".sbg schedule file")
    parser.add_argument("-o", "--outfile", required=True)
    parser.add_argument("-d", "--duration", type=float)
    parser.add_argument("--base", type=float)
    parser.add_argument("--beat", type=float)
    parser.add_argument("--noise", type=float, metavar="AMP")
    parser.add_argument("--noise-kind", default="white", choices=["white", "pink"])
    parser.add_argument("--isochronic", nargs=2, metavar=("FREQ", "BEAT"), type=float)
    parser.add_argument("--harmonic-box", nargs=3, metavar=("BASE", "DIFF", "MOD"), type=float)
    parser.add_argument("--music", help="Background audio file")
    parser.add_argument("--music-amp", type=float, default=100.0)
    parser.add_argument("--loop-music", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.schedule:
            chunks = render_schedule(args.schedule, args.duration)
        else:
            if args.duration is None or args.duration <= 0:
                parser.error("--duration must be positive without a schedule")
            specs = build_quick_specs(
                base=args.base,
                beat=args.beat,
                isochronic=tuple(args.isochronic) if args.isochronic else None,
                harmonic_box=tuple(args.harmonic_box) if args.harmonic_box else None,
                noise=args.noise,
                noise_kind=args.noise_kind,
                music=args.music,
                music_amp=args.music_amp,
                loop_music=args.loop_music,
            )
            chunks = render_specs(specs, args.duration)

        result = write_audio(chunks, args.outfile)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Wrote {result.duration:.2f}s to {result.outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
