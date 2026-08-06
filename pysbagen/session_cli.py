"""Command-line loop for memorable, evolving, local-first PySbagen sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from .api import render_sleep, write_audio
from .constellation import (
    build_constellation,
    constellation_to_text,
    write_constellation_html,
)
from .living_sessions import (
    AffectSnapshot,
    LivingSessionArchive,
    SessionOutcome,
    create_child_sleep_plan,
    create_sleep_plan,
    recommend_child_mode,
    sleep_request_from_manifest,
)
from .sleep import (
    INTENSITY_LABELS,
    PROBLEM_LABELS,
    SOUND_WORLD_LABELS,
    SleepRequest,
    build_sleep_recipe,
    write_recipe_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sbgpy-session",
        description="Create living session lineages, echoes, outcomes, and a local personal atlas.",
    )
    parser.add_argument("--root", help="Living-session archive root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_parser = subparsers.add_parser("new-sleep", help="Create a memorable root sleep session")
    new_parser.add_argument("--problem", choices=sorted(PROBLEM_LABELS), required=True)
    new_parser.add_argument("--sound-world", choices=sorted(SOUND_WORLD_LABELS), required=True)
    new_parser.add_argument("--intensity", choices=sorted(INTENSITY_LABELS), default="balanced")
    new_parser.add_argument("--duration", type=float, default=45.0)
    new_parser.add_argument("--user-audio")
    new_parser.add_argument("--seed", type=int, default=0)
    new_parser.add_argument("--backend-policy", choices=["python", "sbagenx", "auto"], default="python")
    _add_affect_arguments(new_parser, "pre")
    new_parser.add_argument("--json", action="store_true", dest="as_json")

    next_parser = subparsers.add_parser("next", help="Create the next return, branch, contrast, or wander")
    next_parser.add_argument("session_id")
    next_parser.add_argument("--mode", choices=["auto", "return", "branch", "contrast", "wander"], default="auto")
    _add_affect_arguments(next_parser, "pre")
    next_parser.add_argument("--json", action="store_true", dest="as_json")

    show_parser = subparsers.add_parser("show", help="Show one plan, its echoes/events, and outcome")
    show_parser.add_argument("session_id")
    show_parser.add_argument("--json", action="store_true", dest="as_json")

    list_parser = subparsers.add_parser("list", help="List the local session atlas chronologically")
    list_parser.add_argument("--json", action="store_true", dest="as_json")

    mark_parser = subparsers.add_parser("mark", help="Attach a memorable echo or event")
    mark_parser.add_argument("session_id")
    mark_parser.add_argument("--kind", choices=["echo", "shift", "insight", "discomfort", "custom"], default="echo")
    mark_parser.add_argument("--label", required=True)
    mark_parser.add_argument("--at", type=float, dest="position_seconds")
    mark_parser.add_argument("--payload", help="Optional JSON object with extra local details")

    finish_parser = subparsers.add_parser("finish", help="Record an optional non-medical outcome check-in")
    finish_parser.add_argument("session_id")
    finish_parser.add_argument("--rating", type=int, choices=range(1, 6), required=True)
    finish_parser.add_argument("--would-repeat", choices=["yes", "no"], required=True)
    finish_parser.add_argument("--comfort", choices=["comfortable", "neutral", "uncomfortable"], required=True)
    finish_parser.add_argument("--note")
    finish_parser.add_argument("--tag", action="append", default=[])
    _add_affect_arguments(finish_parser, "post")

    render_parser = subparsers.add_parser("render", help="Render the exact stored sleep recipe")
    render_parser.add_argument("session_id")
    render_parser.add_argument("-o", "--outfile")

    atlas_parser = subparsers.add_parser("atlas", help="Summarize lineages, echoes, and descriptive local patterns")
    atlas_parser.add_argument("--json", action="store_true", dest="as_json")

    constellation_parser = subparsers.add_parser(
        "constellation",
        help="Navigate session ancestry, changes, echoes, outcomes, backends, and provenance",
    )
    constellation_parser.add_argument("--lineage", help="Limit the snapshot to one lineage ID")
    constellation_parser.add_argument("--focus", help="Select and center one session when the HTML opens")
    constellation_parser.add_argument(
        "--html",
        nargs="?",
        const="living-session-constellation.html",
        metavar="PATH",
        help="Write a self-contained offline HTML navigator (default filename when omitted)",
    )
    constellation_parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive = LivingSessionArchive(args.root)
    try:
        if args.command == "new-sleep":
            request = SleepRequest(
                problem=args.problem,
                sound_world=args.sound_world,
                intensity=args.intensity,
                duration_minutes=args.duration,
                user_audio=args.user_audio,
                seed=args.seed,
            )
            plan = create_sleep_plan(
                request,
                pre_affect=_affect_from_args(args, "pre"),
                backend_policy=args.backend_policy,
            )
            archive.create(plan)
            _print_plan(plan, args.as_json)
            return 0

        if args.command == "next":
            parent = archive.get(args.session_id)
            mode = recommend_child_mode(parent, archive) if args.mode == "auto" else args.mode
            plan = create_child_sleep_plan(
                parent.plan,
                mode=mode,
                archive=archive,
                pre_affect=_affect_from_args(args, "pre"),
            )
            archive.create(plan)
            _print_plan(plan, args.as_json)
            return 0

        if args.command == "show":
            stored = archive.get(args.session_id)
            if args.as_json:
                print(
                    json.dumps(
                        {
                            "plan": stored.plan.to_dict(),
                            "status": stored.status,
                            "events": [event.to_dict() for event in stored.events],
                            "outcome": stored.outcome.to_dict() if stored.outcome else None,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                _print_stored(stored)
            return 0

        if args.command == "list":
            sessions = archive.list_sessions()
            if args.as_json:
                print(
                    json.dumps(
                        [
                            {
                                "session_id": item.plan.session_id,
                                "lineage_id": item.plan.lineage_id,
                                "title": item.plan.title,
                                "memory_phrase": item.plan.memory_phrase,
                                "mode": item.plan.mode,
                                "generation": item.plan.generation,
                                "status": item.status,
                                "rating": item.outcome.rating if item.outcome else None,
                            }
                            for item in sessions
                        ],
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                for item in sessions:
                    rating = f" rating={item.outcome.rating}" if item.outcome else ""
                    print(
                        f"{item.plan.session_id}  {item.status:9}  "
                        f"g{item.plan.generation} {item.plan.mode:8}  {item.plan.memory_phrase}{rating}"
                    )
            return 0

        if args.command == "mark":
            payload = json.loads(args.payload) if args.payload else {}
            if not isinstance(payload, dict):
                raise ValueError("--payload must decode to a JSON object")
            event = archive.append_event(
                args.session_id,
                kind=args.kind,
                label=args.label,
                position_seconds=args.position_seconds,
                payload=payload,
            )
            print(f"Recorded {event.kind} {event.event_id}: {event.label}")
            return 0

        if args.command == "finish":
            outcome = SessionOutcome(
                session_id=args.session_id,
                completed_at=_utc_now(),
                rating=args.rating,
                would_repeat=args.would_repeat == "yes",
                comfort=args.comfort,
                post_affect=_affect_from_args(args, "post"),
                note=args.note,
                tags=tuple(sorted(set(args.tag))),
            )
            stored = archive.finish(outcome)
            print(
                f"Completed {stored.plan.memory_phrase}: rating {outcome.rating}/5, "
                f"would repeat={'yes' if outcome.would_repeat else 'no'}"
            )
            print("Use `sbgpy-session next SESSION_ID --mode auto` for a transparent next step.")
            return 0

        if args.command == "render":
            stored = archive.get(args.session_id)
            if stored.plan.backend_policy == "sbagenx":
                raise ValueError(
                    "This plan requires SBaGenX, but native rendering is not qualified yet; "
                    "use a Python plan or wait for the typed native-render receipt train."
                )
            request = sleep_request_from_manifest(stored.plan.recipe_manifest)
            outfile = Path(args.outfile or f"{_slug(stored.plan.title)}-{stored.plan.session_id[:8]}.wav")
            result = write_audio(render_sleep(request), outfile)
            recipe_path = write_recipe_manifest(build_sleep_recipe(request), result.outfile)
            archive.append_event(
                args.session_id,
                kind="render",
                label="Rendered exact living-session recipe",
                payload={
                    "output_path": str(result.outfile),
                    "output_sha256": _sha256_file(result.outfile),
                    "frames": result.frames,
                    "sample_rate": result.sample_rate,
                    "duration": result.duration,
                    "peak": result.peak,
                    "recipe_manifest_path": str(recipe_path.resolve()),
                    "recipe_sha256": stored.plan.recipe_sha256,
                    "backend_policy": stored.plan.backend_policy,
                    "actual_backend": "python",
                    "backend_reason": (
                        "explicit Python policy"
                        if stored.plan.backend_policy == "python"
                        else "auto policy selected the qualified portable Python backend; native rendering is not enabled"
                    ),
                },
            )
            print(f"Rendered {stored.plan.memory_phrase}")
            print(f"Audio: {result.outfile}")
            print(f"Recipe: {recipe_path}")
            return 0

        if args.command == "atlas":
            atlas = archive.atlas()
            if args.as_json:
                print(json.dumps(atlas, indent=2, sort_keys=True))
            else:
                _print_atlas(atlas)
            return 0

        if args.command == "constellation":
            html_export = None
            if args.html:
                html_path, graph = write_constellation_html(
                    archive,
                    args.html,
                    lineage_id=args.lineage,
                    focus_session_id=args.focus,
                )
                html_export = {
                    "path": str(html_path),
                    "sha256": _sha256_file(html_path),
                    "snapshot_sha256": graph["snapshot_sha256"],
                }
            else:
                graph = build_constellation(
                    archive,
                    lineage_id=args.lineage,
                    focus_session_id=args.focus,
                )
            if args.as_json:
                payload = (
                    {"constellation": graph, "html_export": html_export}
                    if html_export is not None
                    else graph
                )
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(constellation_to_text(graph))
                if html_export is not None:
                    print(f"Offline navigator: {html_export['path']}")
                    print(f"HTML SHA-256: {html_export['sha256']}")
            return 0
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"sbgpy-session: {exc}") from exc
    raise AssertionError(f"Unhandled command: {args.command}")


def _add_affect_arguments(parser: argparse.ArgumentParser, prefix: str) -> None:
    parser.add_argument(f"--{prefix}-valence", type=float)
    parser.add_argument(f"--{prefix}-arousal", type=float)
    parser.add_argument(f"--{prefix}-agency", type=float)
    parser.add_argument(f"--{prefix}-note")


def _affect_from_args(args: argparse.Namespace, prefix: str) -> AffectSnapshot | None:
    values = [
        getattr(args, f"{prefix}_valence"),
        getattr(args, f"{prefix}_arousal"),
        getattr(args, f"{prefix}_agency"),
    ]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"Supply all three --{prefix}-valence/--{prefix}-arousal/--{prefix}-agency values")
    return AffectSnapshot(
        valence=float(values[0]),
        arousal=float(values[1]),
        agency=float(values[2]),
        note=getattr(args, f"{prefix}_note"),
    )


def _print_plan(plan, as_json: bool) -> None:
    if as_json:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return
    print(plan.memory_phrase)
    print(f"Session: {plan.session_id}")
    print(f"Lineage: {plan.lineage_id} · generation {plan.generation} · mode {plan.mode}")
    print(f"Recipe SHA-256: {plan.recipe_sha256}")
    print(plan.rationale)
    if plan.mutations:
        print("Disclosed changes:")
        for mutation in plan.mutations:
            print(f"  {mutation.key}: {mutation.before!r} -> {mutation.after!r}")
            print(f"    {mutation.reason}")
    else:
        print("Disclosed changes: none")
    if plan.experimental:
        print("Exploration note: this wander combines changes and is less causally interpretable.")


def _print_stored(stored) -> None:
    _print_plan(stored.plan, False)
    print(f"Status: {stored.status}")
    if stored.events:
        print("Events:")
        for event in stored.events:
            at = f" @ {event.position_seconds:.2f}s" if event.position_seconds is not None else ""
            print(f"  {event.kind}{at}: {event.label}")
    else:
        print("Events: none")
    if stored.outcome:
        print(
            f"Outcome: {stored.outcome.rating}/5, comfort={stored.outcome.comfort}, "
            f"would-repeat={'yes' if stored.outcome.would_repeat else 'no'}"
        )
    else:
        print("Outcome: not recorded")


def _print_atlas(atlas: dict) -> None:
    print("PySbagen Living Session Atlas")
    print(
        f"Sessions: {atlas['session_count']} · completed: {atlas['completed_count']} · "
        f"lineages: {atlas['lineage_count']} · echoes: {atlas['echo_count']}"
    )
    if atlas["average_rating"] is not None:
        print(f"Average local rating: {atlas['average_rating']:.2f}/5")
    if atlas["would_repeat_rate"] is not None:
        print(f"Would-repeat rate: {atlas['would_repeat_rate']:.0%}")
    if atlas["average_affect_delta"]:
        delta = atlas["average_affect_delta"]
        print(
            "Average recorded state delta: "
            f"valence {delta['valence']:+.2f}, arousal {delta['arousal']:+.2f}, agency {delta['agency']:+.2f}"
        )
    if atlas["lineages"]:
        print("Lineages:")
        for lineage in atlas["lineages"]:
            print(
                f"  {lineage['lineage_id']}: {lineage['session_count']} session(s), "
                f"{lineage['completed_count']} completed · {' -> '.join(lineage['titles'])}"
            )
    if atlas["pattern_candidates"]:
        print("Descriptive local pattern candidates:")
        for pattern in atlas["pattern_candidates"]:
            print(
                f"  {pattern['kind']}={pattern['label']}: {pattern['average_rating']:.2f}/5 "
                f"across {pattern['observations']} observations"
            )
    print("These are personal descriptive records, not medical-efficacy conclusions.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "living-session"


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
