"""CLI for exporting and opening offline Living Session constellations."""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

from .constellation import (
    build_constellation,
    write_constellation_html,
    write_constellation_json,
)
from .living_sessions import LivingSessionArchive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sbgpy-constellation",
        description=(
            "Build a self-contained offline navigator for Living Session lineages, "
            "mutations, echoes, outcomes, backends, and provenance."
        ),
    )
    parser.add_argument("--root", help="Living-session archive root")
    parser.add_argument("--lineage", help="Restrict the snapshot to one lineage ID")
    parser.add_argument(
        "--session",
        dest="focus_session_id",
        help="Focus one session and automatically restrict to its lineage",
    )
    parser.add_argument(
        "--format",
        choices=["html", "json"],
        default="html",
        help="Snapshot format (default: html)",
    )
    parser.add_argument("-o", "--outfile", help="Destination path")
    parser.add_argument(
        "--redact-notes",
        action="store_true",
        help="Remove free-text notes, event labels/payloads, and user-audio paths",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        dest="open_after",
        help="Open the generated HTML in the default browser",
    )
    parser.add_argument(
        "--summary-json",
        action="store_true",
        help="Print the snapshot summary and identity as JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.open_after and args.format != "html":
        raise SystemExit("sbgpy-constellation: --open requires --format html")

    archive = LivingSessionArchive(args.root)
    try:
        graph = build_constellation(
            archive,
            lineage_id=args.lineage,
            focus_session_id=args.focus_session_id,
        )
        default_name = (
            f"living-constellation-{args.focus_session_id[:8]}.html"
            if args.focus_session_id and args.format == "html"
            else f"living-constellation.{args.format}"
        )
        destination = Path(args.outfile or default_name)
        if args.format == "html":
            written = write_constellation_html(
                graph,
                destination,
                redact_notes=args.redact_notes,
            )
        else:
            written = write_constellation_json(
                graph,
                destination,
                redact_notes=args.redact_notes,
            )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"sbgpy-constellation: {exc}") from exc

    summary = graph.to_dict(redact_notes=args.redact_notes)["summary"]
    receipt = {
        "schema": "pysbagen.constellation-export-receipt.v1",
        "path": str(written),
        "format": args.format,
        "graph_sha256": graph.graph_sha256,
        "focus_session_id": graph.focus_session_id,
        "lineage_filter": args.lineage,
        "notes_redacted": args.redact_notes,
        **summary,
    }
    if args.summary_json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(
            f"Wrote {summary['session_count']} session(s), {summary['edge_count']} relationship(s), "
            f"and {summary['echo_count']} echo(es) to {written}"
        )
        print(f"Snapshot SHA-256: {graph.graph_sha256}")
        if summary["warning_count"]:
            print(f"Integrity warnings preserved: {summary['warning_count']}")

    if args.open_after:
        webbrowser.open(written.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
