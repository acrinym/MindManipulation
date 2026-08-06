from __future__ import annotations

from pathlib import Path

import pytest

from pysbagen.constellation import (
    build_constellation,
    constellation_to_text,
    render_constellation_html,
)
from pysbagen.living_sessions import (
    AffectSnapshot,
    LivingSessionArchive,
    SessionOutcome,
    create_child_sleep_plan,
    create_sleep_plan,
)
from pysbagen.session_cli import main as session_main
from pysbagen.sleep import SleepRequest


def _request() -> SleepRequest:
    return SleepRequest(
        problem="racing_mind",
        sound_world="rain_room",
        intensity="balanced",
        duration_minutes=45,
        seed=17,
    )


def _lineage(archive: LivingSessionArchive):
    root = create_sleep_plan(
        _request(),
        pre_affect=AffectSnapshot(valence=-0.5, arousal=0.9, agency=0.2),
        created_at="2026-08-05T20:00:00+00:00",
    )
    archive.create(root)
    archive.append_event(
        root.session_id,
        kind="render",
        label="Rendered exact living-session recipe",
        payload={
            "actual_backend": "python",
            "output_path": "/tmp/root.wav",
            "output_sha256": "f" * 64,
            "backend_reason": "explicit Python policy",
        },
        created_at="2026-08-05T20:01:00+00:00",
    )
    archive.finish(
        SessionOutcome(
            session_id=root.session_id,
            completed_at="2026-08-05T21:00:00+00:00",
            rating=5,
            would_repeat=True,
            comfort="comfortable",
            post_affect=AffectSnapshot(valence=0.4, arousal=0.2, agency=0.7),
            tags=("rain", "settled"),
        )
    )
    branch = create_child_sleep_plan(
        root,
        mode="branch",
        archive=archive,
        created_at="2026-08-05T22:00:00+00:00",
    )
    archive.create(branch)
    archive.append_event(
        branch.session_id,
        kind="echo",
        label="The rain became a room",
        position_seconds=123.5,
        created_at="2026-08-05T22:02:00+00:00",
    )
    returned = create_child_sleep_plan(
        branch,
        mode="return",
        archive=archive,
        created_at="2026-08-05T23:00:00+00:00",
    )
    archive.create(returned)
    return root, branch, returned


def test_constellation_exposes_identity_changes_memory_outcome_and_backend(tmp_path: Path):
    archive = LivingSessionArchive(tmp_path / "living")
    root, branch, returned = _lineage(archive)

    graph = build_constellation(archive, focus_session_id=branch.session_id)
    nodes = {node["session_id"]: node for node in graph["nodes"]}
    edges = {edge["target"]: edge for edge in graph["edges"]}

    assert graph["counts"] == {
        "sessions": 3,
        "edges": 2,
        "lineages": 1,
        "completed": 1,
        "echoes": 1,
        "warnings": 0,
    }
    assert len(graph["snapshot_sha256"]) == 64
    assert graph == build_constellation(archive, focus_session_id=branch.session_id)
    assert nodes[root.session_id]["backend"]["actual_backends"] == ["python"]
    assert nodes[root.session_id]["backend"]["latest_output_sha256"] == "f" * 64
    assert nodes[root.session_id]["outcome"]["affect_delta"]["valence"] == pytest.approx(0.9)
    assert nodes[branch.session_id]["echo_count"] == 1
    assert nodes[branch.session_id]["echoes"][0]["label"] == "The rain became a room"
    assert edges[branch.session_id]["change_count"] == 1
    assert edges[branch.session_id]["causal_interpretability"] == "high"
    assert edges[returned.session_id]["recipe_identity_preserved"]
    assert edges[returned.session_id]["short_label"] == "exact return"
    assert nodes[branch.session_id]["layout"]["x"] > nodes[root.session_id]["layout"]["x"]

    text = constellation_to_text(graph)
    assert root.title in text
    assert branch.mutations[0].key in text
    assert "personal descriptive records" in text


def test_constellation_html_is_self_contained_searchable_and_script_safe(tmp_path: Path):
    archive = LivingSessionArchive(tmp_path / "living")
    _, branch, _ = _lineage(archive)
    archive.append_event(
        branch.session_id,
        kind="echo",
        label="Unsafe </script><script>alert(1)</script> memory",
        created_at="2026-08-05T22:03:00+00:00",
    )

    document = render_constellation_html(build_constellation(archive))

    assert '<script src=' not in document
    assert '<link rel=' not in document
    assert "constellation-data" in document
    assert "Search title, motif, echo, hash" in document
    assert "Filter lineage" in document
    assert "\\u003c/script\\u003e" in document
    assert "Unsafe </script>" not in document


def test_constellation_lineage_filter_and_focus_are_fail_closed(tmp_path: Path):
    archive = LivingSessionArchive(tmp_path / "living")
    root, branch, _ = _lineage(archive)
    other = create_sleep_plan(
        SleepRequest(problem="cannot_cross", sound_world="deep_night", duration_minutes=30),
        created_at="2026-08-06T00:00:00+00:00",
    )
    archive.create(other)

    graph = build_constellation(
        archive,
        lineage_id=root.lineage_id,
        focus_session_id=branch.session_id,
    )

    assert graph["counts"]["sessions"] == 3
    assert {node["lineage_id"] for node in graph["nodes"]} == {root.lineage_id}
    with pytest.raises(KeyError, match="lineage not found"):
        build_constellation(archive, lineage_id="missing-lineage")
    with pytest.raises(KeyError, match="not present"):
        build_constellation(archive, lineage_id=root.lineage_id, focus_session_id=other.session_id)


def test_constellation_surfaces_parent_absence_instead_of_inventing_an_edge(tmp_path: Path):
    root = create_sleep_plan(_request(), created_at="2026-08-05T20:00:00+00:00")
    child = create_child_sleep_plan(
        root,
        mode="branch",
        created_at="2026-08-05T21:00:00+00:00",
    )
    orphan_archive = LivingSessionArchive(tmp_path / "orphan")
    orphan_archive.create(child)

    graph = build_constellation(orphan_archive)

    assert graph["counts"]["edges"] == 0
    assert graph["counts"]["warnings"] == 1
    assert graph["warnings"][0]["code"] == "parent-not-in-snapshot"
    assert graph["nodes"][0]["warnings"][0]["session_id"] == child.session_id


def test_constellation_cli_writes_offline_navigator_and_prints_receipts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    archive_root = tmp_path / "living"
    archive = LivingSessionArchive(archive_root)
    root, _, _ = _lineage(archive)
    destination = tmp_path / "constellation.html"

    assert session_main(
        [
            "--root",
            str(archive_root),
            "constellation",
            "--focus",
            root.session_id,
            "--html",
            str(destination),
        ]
    ) == 0

    output = capsys.readouterr().out
    assert destination.is_file()
    assert "Offline navigator:" in output
    assert "HTML SHA-256:" in output
    assert "Snapshot:" in output
