"""CLI for Living Sessions Confluence experiences."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .confluence import (
    TRAIT_KEYS,
    build_confluence_constellation,
    create_confluence_session,
    describe_confluence,
    suggest_confluence,
)
from .constellation import constellation_to_text, render_constellation_html
from .living_sessions import LivingSessionArchive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sbgpy-confluence",
        description="Combine two remembered Living Sessions into a new reproducible experience.",
    )
    parser.add_argument("--root", help="Living-session archive root")
    commands = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("suggest", "Preview inheritance without writing a session"),
        ("create", "Create and remember a Confluence session"),
    ):
        cmd = commands.add_parser(name, help=help_text)
        cmd.add_argument("parent_a")
        cmd.add_argument("parent_b")
        cmd.add_argument("--from-a", default="", help=f"Comma-separated: {', '.join(TRAIT_KEYS)}")
        cmd.add_argument("--from-b", default="", help=f"Comma-separated: {', '.join(TRAIT_KEYS)}")
        cmd.add_argument("--json", action="store_true", dest="as_json")

    show = commands.add_parser("show", help="Show both ancestors and inheritance")
    show.add_argument("session_id")
    show.add_argument("--json", action="store_true", dest="as_json")

    graph = commands.add_parser("constellation", help="Show dual-parent ancestry")
    graph.add_argument("--focus")
    graph.add_argument("--html", nargs="?", const="living-session-confluence-constellation.html")
    graph.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive = LivingSessionArchive(args.root)
    try:
        if args.command == "suggest":
            a, b = archive.get(args.parent_a), archive.get(args.parent_b)
            suggestion = suggest_confluence(
                a, b, from_a=_traits(args.from_a), from_b=_traits(args.from_b)
            )
            payload = suggestion.to_dict()
            _emit(payload, args.as_json, header=f"{a.plan.memory_phrase} x {b.plan.memory_phrase}")
            return 0

        if args.command == "create":
            stored = create_confluence_session(
                archive,
                args.parent_a,
                args.parent_b,
                from_a=_traits(args.from_a),
                from_b=_traits(args.from_b),
            )
            payload = describe_confluence(stored)
            _emit(payload, args.as_json, header=f"Created {payload['memory_phrase']}")
            if not args.as_json:
                print("Use sbgpy-session render/mark/finish with this session ID, or reuse it as a later ancestor.")
            return 0

        if args.command == "show":
            payload = describe_confluence(archive.get(args.session_id))
            _emit(payload, args.as_json, header=payload["memory_phrase"])
            return 0

        graph = build_confluence_constellation(archive, focus_session_id=args.focus)
        html_export = None
        if args.html:
            path = Path(args.html).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render_constellation_html(graph), encoding="utf-8")
            html_export = {"path": str(path.resolve()), "sha256": _sha256(path)}
        if args.as_json:
            payload = {"constellation": graph}
            if html_export:
                payload["html_export"] = html_export
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(constellation_to_text(graph))
            print(
                "Confluence second-parent connections: "
                f"{graph['counts'].get('confluence_second_parent_edges', 0)}"
            )
            for edge in graph["edges"]:
                if edge["mode"] == "confluence-b":
                    print(f"  B {edge['source'][:8]} -> {edge['target'][:8]}")
            if html_export:
                print(f"Offline navigator: {html_export['path']}")
                print(f"HTML SHA-256: {html_export['sha256']}")
        return 0
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"sbgpy-confluence: {exc}") from exc


def _traits(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _emit(payload: dict, as_json: bool, *, header: str) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(header)
    if "session_id" in payload:
        print(f"Session: {payload['session_id']}")
    for item in payload.get("assignments", payload.get("inheritance", [])):
        print(f"  {item['trait']} <- {item['source']}: {json.dumps(item['value'], sort_keys=True)}")
        print(f"    {item['reason']}")
    for tension in payload.get("tensions", []):
        print(f"  tension: {tension}")
    if payload.get("rationale"):
        print(payload["rationale"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
