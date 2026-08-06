from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from pysbagen.constellation import (
    build_constellation,
    render_constellation_html,
    write_constellation_json,
)
from pysbagen.constellation_cli import main as constellation_main
from pysbagen.living_sessions import (
    AffectSnapshot,
    LivingSessionArchive,
    SessionOutcome,
    create_child_sleep_plan,
    create_sleep_plan,
)
from pysbagen.sleep import SleepRequest


def _archive_with_lineage(tmp_path: Path) -> tuple[LivingSessionArchive, object, object, object]:
    archive = LivingSessionArchive(tmp_path / "archive")
    root = create_sleep_plan(
        SleepRequest(
            problem="racing_mind",
            sound_world="rain_room",
            intensity="balanced",
            duration_minutes=45,
            seed=11,
        ),
        pre_affect=AffectSnapshot(-0.4, 0.8, 0.3, "Busy evening"),
        created_at="2026-08-05T20:00:00+00:00",
    )
    archive.create(root)
    branch = create_child_sleep_plan(
        root,
        mode="branch",
        archive=archive,
        created_at="2026-08-05T21:00:00+00:00",
    )
    archive.create(branch)
    returned = create_child_sleep_plan(
        root,
        mode="return",
        archive=archive,
        created_at="2026-08-05T22:00:00+00:00",
    )
    archive.create(returned)
    archive.append_event(
        root.session_id,
        kind="echo",
        label="The rain became a room",
        position_seconds=123.5,
        payload={"certainty": "approximate"},
        created_at="2026-08-05T20:10:00+00:00",
    )
    archive.append_event(
        branch.session_id,
        kind="render",
        label="Rendered exact living-session recipe",
        payload={"actual_backend": "python", "output_sha256": "abc"},
        created_at="2026-08-05T21:10:00+00:00",
    )
    archive.finish(
        SessionOutcome(
            session_id=root.session_id,
            completed_at="2026-08-05T20:50:00+00:00",
            rating=5,
            would_repeat=True,
            comfort="comfortable",
            post_affect=AffectSnapshot(0.4, 0.2, 0.7, "Settled"),
            note="Worked well tonight",
            tags=("rain", "settled"),
        )
    )
    return archive, root, branch, returned


def test_constellation_derives_nodes_edges_and_provenance(tmp_path: Path):
    archive, root, branch, returned = _archive_with_lineage(tmp_path)

    graph = build_constellation(archive)
    payload = graph.to_dict()

    assert payload["summary"] == {
        "session_count": 3,
        "edge_count": 2,
        "lineage_count": 1,
        "completed_count": 1,
        "echo_count": 1,
        "warning_count": 0,
    }
    assert len(graph.graph_sha256) == 64
    assert {node.session_id for node in graph.nodes} == {
        root.session_id,
        branch.session_id,
        returned.session_id,
    }
    root_node = next(node for node in graph.nodes if node.session_id == root.session_id)
    branch_node = next(node for node in graph.nodes if node.session_id == branch.session_id)
    assert root_node.x == 70
    assert branch_node.x == 370
    assert root_node.echo_count == 1
    assert root_node.rating == 5
    assert branch_node.render_count == 1
    assert any(edge.target_session_id == branch.session_id and edge.mutations for edge in graph.edges)
    assert any(edge.target_session_id == returned.session_id and edge.label == "return" for edge in graph.edges)


def test_focus_session_restricts_to_its_lineage(tmp_path: Path):
    archive, root, branch, _ = _archive_with_lineage(tmp_path)
    other = create_sleep_plan(
        SleepRequest(
            problem="cannot_cross",
            sound_world="deep_night",
            intensity="gentle",
            duration_minutes=30,
            seed=99,
        ),
        created_at="2026-08-06T00:00:00+00:00",
    )
    archive.create(other)

    graph = build_constellation(archive, focus_session_id=branch.session_id)

    assert graph.focus_session_id == branch.session_id
    assert {node.lineage_id for node in graph.nodes} == {root.lineage_id}
    assert other.session_id not in {node.session_id for node in graph.nodes}


def test_html_is_self_contained_searchable_and_script_safe(tmp_path: Path):
    archive, root, _, _ = _archive_with_lineage(tmp_path)
    archive.append_event(
        root.session_id,
        kind="custom",
        label="</script><script>alert('x')</script>",
        payload={"note": "private"},
        created_at="2026-08-05T20:20:00+00:00",
    )
    graph = build_constellation(archive, focus_session_id=root.session_id)

    rendered = render_constellation_html(graph, redact_notes=True)

    assert "PySbagen Living Session Constellation" in rendered
    assert 'id="search"' in rendered
    assert 'id="lineage"' in rendered
    assert root.session_id in rendered
    assert "</script><script>alert" not in rendered
    assert "[redacted]" in rendered
    assert '"notes_redacted":true' in rendered
    assert "https://" not in rendered


def test_json_export_has_snapshot_identity_and_redaction(tmp_path: Path):
    archive, root, _, _ = _archive_with_lineage(tmp_path)
    graph = build_constellation(archive)
    destination = write_constellation_json(
        graph,
        tmp_path / "constellation.json",
        redact_notes=True,
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    node = next(item for item in payload["nodes"] if item["session_id"] == root.session_id)
    assert payload["graph_sha256"] == graph.graph_sha256
    assert payload["privacy"]["notes_redacted"] is True
    assert node["outcome"]["note"] is None
    assert node["pre_affect"]["note"] is None
    assert node["events"][0]["label"] == "[redacted]"


def test_missing_parent_is_preserved_as_integrity_warning(tmp_path: Path):
    archive = LivingSessionArchive(tmp_path / "archive")
    root = create_sleep_plan(
        SleepRequest(
            problem="cannot_cross",
            sound_world="warm_ambient",
            duration_minutes=30,
        ),
        created_at="2026-08-05T20:00:00+00:00",
    )
    orphan = replace(
        root,
        session_id="orphan-session-00000001",
        parent_session_id="missing-parent-000000",
        generation=1,
        created_at="2026-08-05T21:00:00+00:00",
    )
    archive.create(orphan)

    graph = build_constellation(archive)

    assert graph.edges == []
    assert any("parent missing-parent-000000" in warning for warning in graph.integrity_warnings)


def test_constellation_cli_writes_machine_readable_snapshot(tmp_path: Path, capsys):
    archive, root, _, _ = _archive_with_lineage(tmp_path)
    destination = tmp_path / "out.json"

    code = constellation_main(
        [
            "--root",
            str(archive.root),
            "--session",
            root.session_id,
            "--format",
            "json",
            "--redact-notes",
            "--summary-json",
            "-o",
            str(destination),
        ]
    )

    assert code == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["format"] == "json"
    assert receipt["session_count"] == 3
    assert receipt["notes_redacted"] is True
    assert destination.is_file()
