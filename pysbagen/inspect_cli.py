from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .compatibility import RenderDisposition
from .importers import import_artifact
from .inspector import build_timeline, inspect_audio_source, qualify_audio_path, timeline_to_dict, timeline_to_text
from .interoperability import inspect_with_sbagenx
from .library import LocalLibrary
from .sbagenx_backend import probe_sbagenx
from .sbagenx_native import SBaGenXNativeError
from .sbgf import import_sbgf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sbgpy-inspect", description="Inspect SBG/SBGF/DRG compatibility, audio sources, output routes, optional SBaGenX backends, and the local library.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="Inspect an SBG, SBGF, or DRG artifact")
    inspect_parser.add_argument("source")
    inspect_parser.add_argument("--json", action="store_true", dest="as_json")
    inspect_parser.add_argument("--timeline-json", action="store_true")
    inspect_parser.add_argument("--duration", type=float)
    inspect_parser.add_argument("--preserve-to")
    inspect_parser.add_argument("--add-to-library", action="store_true")
    inspect_parser.add_argument("--library-root")
    source_parser = subparsers.add_parser("source", help="Qualify a user-provided audio source")
    source_parser.add_argument("source")
    source_parser.add_argument("--target-sample-rate", type=int, default=44100)
    source_parser.add_argument("--json", action="store_true", dest="as_json")
    path_parser = subparsers.add_parser("path", help="Qualify a listening/rendering path")
    path_parser.add_argument("--method", choices=["binaural", "monaural", "isochronic", "mixed"], required=True)
    path_parser.add_argument("--route", choices=["headphones", "earbuds", "speakers", "device"], required=True)
    path_parser.add_argument("--channels", type=int, required=True)
    path_parser.add_argument("--sample-rate", type=int, required=True)
    path_parser.add_argument("--spatial-processing", action="store_true")
    path_parser.add_argument("--normalization", action="store_true")
    path_parser.add_argument("--bluetooth", action="store_true")
    path_parser.add_argument("--json", action="store_true", dest="as_json")
    path_parser.add_argument("--save")
    backend_parser = subparsers.add_parser("backend", help="Discover, qualify, or validate through an optional SBaGenX native library")
    backend_parser.add_argument("--executable", help="Explicit SBaGenX executable path; otherwise use SBAGENX_BIN/PATH")
    backend_parser.add_argument("--library", help="Explicit sbagenxlib path; otherwise use SBAGENXLIB_PATH/system lookup")
    backend_parser.add_argument("--discover-only", action="store_true", help="Locate candidates without executing or loading them")
    backend_parser.add_argument("--validate", metavar="SOURCE", help="Inspect and compare a .sbg or .sbgf source through PySbagen and qualified API 47")
    backend_parser.add_argument("--json", action="store_true", dest="as_json")
    library_parser = subparsers.add_parser("library", help="Inspect or verify the local-first library")
    library_parser.add_argument("action", choices=["list", "show", "verify", "archive", "export"])
    library_parser.add_argument("item_id", nargs="?")
    library_parser.add_argument("--root")
    library_parser.add_argument("--json", action="store_true", dest="as_json")
    library_parser.add_argument("--destination")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        return _inspect_command(args)
    if args.command == "source":
        report = inspect_audio_source(args.source, target_sample_rate=args.target_sample_rate)
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True) if args.as_json else report.to_text())
        return 0 if report.state.value in {"supported", "equivalent", "partial"} else 2
    if args.command == "path":
        report = qualify_audio_path(method=args.method, route=args.route, channels=args.channels, sample_rate=args.sample_rate, spatial_processing=args.spatial_processing, normalization=args.normalization, bluetooth=args.bluetooth)
        payload = report.to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True) if args.as_json else report.to_text())
        if args.save:
            Path(args.save).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"Saved path qualification: {Path(args.save).resolve()}")
        return 0 if report.safe_to_start else 2
    if args.command == "backend":
        if args.validate:
            if args.discover_only:
                print("--discover-only cannot be combined with --validate", file=sys.stderr)
                return 2
            try:
                report = inspect_with_sbagenx(args.validate, library=args.library)
            except (OSError, ValueError, SBaGenXNativeError) as exc:
                print(f"SBaGenX interoperability validation failed: {exc}", file=sys.stderr)
                return 2
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True) if args.as_json else report.to_text())
            if not report.source_identity_matches or not report.native_valid:
                return 2
            if report.discrepancies or report.pysbagen_disposition is not RenderDisposition.SAFE:
                return 1
            return 0
        report = probe_sbagenx(
            executable=args.executable,
            library=args.library,
            query_executable=not args.discover_only,
            load_library=not args.discover_only,
        )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True) if args.as_json else report.to_text())
        qualified = report.candidate_found if args.discover_only else report.usable
        return 0 if qualified else 2
    if args.command == "library":
        return _library_command(args)
    raise AssertionError(f"Unhandled command: {args.command}")


def _inspect_command(args: argparse.Namespace) -> int:
    source_path = Path(args.source).expanduser().resolve()
    artifact = import_sbgf(source_path) if source_path.suffix.lower() == ".sbgf" else import_artifact(source_path, preserve_to=args.preserve_to)

    if source_path.suffix.lower() == ".sbgf":
        if args.as_json:
            print(json.dumps(artifact.report.to_dict(), indent=2, sort_keys=True))
        else:
            print(artifact.report.to_text())
        if args.timeline_json:
            destination = Path("timeline.json")
            destination.write_text(
                json.dumps(
                    {
                        "schema": "pysbagen.sbgf-structure.v1",
                        "source_sha256": artifact.report.source_sha256,
                        "source_type": "sbgf",
                        "timeline": None,
                        "reason": "SBGF is function-driven; use native curve sampling rather than an invented SBG timeline.",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"SBGF structure receipt: {destination.resolve()}")
        if args.add_to_library:
            item = LocalLibrary(args.library_root).add(artifact)
            print(f"Library item: {item.item_id} ({item.state}) at {item.path}")
        return 2

    timeline = build_timeline(artifact, duration=args.duration)
    if args.as_json:
        payload = artifact.report.to_dict()
        payload["timeline"] = timeline_to_dict(timeline)
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(artifact.report.to_text())
        print()
        print(timeline_to_text(timeline))
    if args.timeline_json:
        destination = Path("timeline.json")
        destination.write_text(json.dumps(timeline_to_dict(timeline), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Timeline JSON: {destination.resolve()}")
    if args.add_to_library:
        item = LocalLibrary(args.library_root).add(artifact)
        print(f"Library item: {item.item_id} ({item.state}) at {item.path}")
    if artifact.report.render_disposition is RenderDisposition.SAFE:
        return 0
    if artifact.report.render_disposition is RenderDisposition.SAFE_WITH_DISCLOSED_CHANGES:
        return 1
    return 2


def _library_command(args: argparse.Namespace) -> int:
    library = LocalLibrary(args.root)
    if args.action == "list":
        items = library.list_items()
        if args.as_json:
            print(json.dumps([item.manifest for item in items], indent=2, sort_keys=True))
        else:
            for item in items:
                print(f"{item.item_id}  {item.state:14}  {item.manifest.get('display_name', '')}")
        return 0
    if not args.item_id:
        raise SystemExit("library show/verify/archive/export requires item_id")
    if args.action == "show":
        print(json.dumps(library.get(args.item_id).manifest, indent=2, sort_keys=True))
        return 0
    if args.action == "verify":
        result = library.verify(args.item_id)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 2
    if args.action == "archive":
        print(f"Archived {library.set_state(args.item_id, 'archived').item_id}")
        return 0
    if args.action == "export":
        exported = library.export_manifest(args.item_id, args.destination or f"{args.item_id}.pysbagen-library.json")
        print(f"Exported {exported.resolve()}")
        return 0
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
