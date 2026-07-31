from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import build_quick_specs, render_schedule, render_specs, write_audio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SBaGen-compatible generator (Python)")
    parser.add_argument("schedule", nargs="?", help=".sbg or .drg schedule artifact")
    parser.add_argument("-o", "--outfile", required=True)
    parser.add_argument("-d", "--duration", type=float)
    parser.add_argument(
        "--allow-disclosed-changes",
        action="store_true",
        help="Render only after accepting partial/equivalent/approximated findings from sbgpy-inspect.",
    )
    parser.add_argument("--base", type=float)
    parser.add_argument("--beat", type=float)
    parser.add_argument("--noise", type=float, metavar="AMP")
    parser.add_argument("--noise-kind", default="white", choices=["white", "pink"])
    parser.add_argument("--isochronic", nargs=2, metavar=("FREQ", "BEAT"), type=float)
    parser.add_argument("--harmonic-box", nargs=3, metavar=("BASE", "DIFF", "MOD"), type=float)
    parser.add_argument("--music", help="Background audio file")
    parser.add_argument("--music-amp", type=float, default=100.0)
    parser.add_argument("--loop-music", action="store_true")
    parser.add_argument(
        "--path-qualification",
        help="JSON file produced by sbgpy-inspect path; a blocked route prevents rendering.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.schedule:
            path_qualification = None
            if args.path_qualification:
                path_qualification = json.loads(Path(args.path_qualification).read_text(encoding="utf-8"))
            chunks = render_schedule(
                args.schedule,
                args.duration,
                allow_disclosed_changes=args.allow_disclosed_changes,
                path_qualification=path_qualification,
            )
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
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print(f"Wrote {result.duration:.2f}s to {result.outfile}")
    if result.manifest:
        print(f"Compatibility manifest: {result.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
