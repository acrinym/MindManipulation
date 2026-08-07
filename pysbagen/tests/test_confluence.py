from __future__ import annotations

import pytest

from pysbagen.confluence import (
    build_confluence_constellation,
    confluence_metadata,
    create_confluence_plan,
    create_confluence_session,
    describe_confluence,
    suggest_confluence,
)
from pysbagen.living_sessions import LivingSessionArchive, SessionOutcome, create_sleep_plan
from pysbagen.sleep import SleepLayers, SleepRequest


def _parents(tmp_path):
    archive = LivingSessionArchive(tmp_path / "living")
    plan_a = create_sleep_plan(
        SleepRequest(
            problem="racing_mind",
            sound_world="rain_room",
            intensity="gentle",
            duration_minutes=30,
            layers=SleepLayers(binaural=True, monaural=False, isochronic=False, harmonic_box=False),
            seed=11,
        ),
        created_at="2026-08-07T10:00:00+00:00",
    )
    plan_b = create_sleep_plan(
        SleepRequest(
            problem="cannot_cross",
            sound_world="deep_night",
            intensity="immersive",
            duration_minutes=90,
            layers=SleepLayers(binaural=False, monaural=True, isochronic=True, harmonic_box=False),
            seed=22,
        ),
        created_at="2026-08-07T10:01:00+00:00",
    )
    archive.create(plan_a)
    archive.create(plan_b)
    archive.append_event(plan_a.session_id, kind="echo", label="Rain became a room", position_seconds=120.0, created_at="2026-08-07T10:02:00+00:00")
    archive.append_event(plan_b.session_id, kind="echo", label="Deep low opening", position_seconds=60.0, created_at="2026-08-07T10:03:00+00:00")
    archive.append_event(plan_b.session_id, kind="echo", label="Threshold softened", position_seconds=240.0, created_at="2026-08-07T10:04:00+00:00")
    archive.finish(SessionOutcome(session_id=plan_a.session_id, completed_at="2026-08-07T11:00:00+00:00", rating=5, would_repeat=True, comfort="comfortable"))
    archive.finish(SessionOutcome(session_id=plan_b.session_id, completed_at="2026-08-07T11:01:00+00:00", rating=4, would_repeat=True, comfort="comfortable"))
    return archive, archive.get(plan_a.session_id), archive.get(plan_b.session_id)


def test_suggestion_is_dual_parent_and_creates_real_bridge_traits(tmp_path):
    _, parent_a, parent_b = _parents(tmp_path)
    suggestion = suggest_confluence(parent_a, parent_b)
    sources = {item.source for item in suggestion.assignments}
    assert {"A", "B", "new"} <= sources
    by_trait = {item.trait: item for item in suggestion.assignments}
    assert by_trait["intensity"].source == "new"
    assert by_trait["intensity"].value == "balanced"
    assert by_trait["duration_minutes"].source == "new"
    assert by_trait["duration_minutes"].value == 60.0
    assert "generation seed is new" in suggestion.rationale


def test_explicit_inheritance_overrides_suggestion_without_hiding_conflicts(tmp_path):
    _, parent_a, parent_b = _parents(tmp_path)
    suggestion = suggest_confluence(parent_a, parent_b, from_a=("sound_world",), from_b=("problem",))
    by_trait = {item.trait: item for item in suggestion.assignments}
    assert by_trait["sound_world"].source == "A"
    assert by_trait["sound_world"].value == "rain_room"
    assert by_trait["problem"].source == "B"
    assert by_trait["problem"].value == "cannot_cross"
    assert any("sound world differs" in item for item in suggestion.tensions)
    assert any("intent differs" in item for item in suggestion.tensions)


def test_same_trait_cannot_be_forced_from_both_parents(tmp_path):
    _, parent_a, parent_b = _parents(tmp_path)
    with pytest.raises(ValueError, match="cannot be explicitly inherited from both"):
        suggest_confluence(parent_a, parent_b, from_a=("sound_world",), from_b=("sound_world",))


def test_create_plan_has_new_identity_and_new_seed(tmp_path):
    _, parent_a, parent_b = _parents(tmp_path)
    plan, suggestion = create_confluence_plan(parent_a, parent_b, created_at="2026-08-07T12:00:00+00:00")
    assert plan.mode == "confluence"
    assert plan.parent_session_id == parent_a.plan.session_id
    assert plan.lineage_id not in {parent_a.plan.lineage_id, parent_b.plan.lineage_id}
    assert plan.title not in {parent_a.plan.title, parent_b.plan.title}
    assert plan.recipe_manifest["request"]["seed"] not in {
        parent_a.plan.recipe_manifest["request"]["seed"],
        parent_b.plan.recipe_manifest["request"]["seed"],
    }
    assert plan.recipe_sha256 not in {parent_a.plan.recipe_sha256, parent_b.plan.recipe_sha256}
    assert suggestion.parent_b_session_id == parent_b.plan.session_id


def test_persisted_confluence_remembers_both_ancestors_and_inheritance(tmp_path):
    archive, parent_a, parent_b = _parents(tmp_path)
    stored = create_confluence_session(archive, parent_a.plan.session_id, parent_b.plan.session_id, created_at="2026-08-07T12:00:00+00:00")
    metadata = confluence_metadata(stored)
    assert metadata is not None
    assert metadata["parent_a"]["session_id"] == parent_a.plan.session_id
    assert metadata["parent_b"]["session_id"] == parent_b.plan.session_id
    assert metadata["schema"] == "pysbagen.living-session-confluence.v1"
    assert {item["source"] for item in metadata["inheritance"]} >= {"A", "B", "new"}
    description = describe_confluence(stored)
    assert description["parent_a"]["title"] == parent_a.plan.title
    assert description["parent_b"]["title"] == parent_b.plan.title


def test_confluence_can_become_a_future_ancestor(tmp_path):
    archive, parent_a, parent_b = _parents(tmp_path)
    first = create_confluence_session(archive, parent_a.plan.session_id, parent_b.plan.session_id, created_at="2026-08-07T12:00:00+00:00")
    second = create_confluence_session(archive, first.plan.session_id, parent_a.plan.session_id, created_at="2026-08-07T13:00:00+00:00")
    metadata = confluence_metadata(second)
    assert metadata is not None
    assert metadata["parent_a"]["session_id"] == first.plan.session_id
    assert second.plan.generation == max(first.plan.generation, parent_a.plan.generation) + 1


def test_constellation_contains_second_parent_edge_without_false_lineage_warning(tmp_path):
    archive, parent_a, parent_b = _parents(tmp_path)
    confluence = create_confluence_session(archive, parent_a.plan.session_id, parent_b.plan.session_id, created_at="2026-08-07T12:00:00+00:00")
    graph = build_confluence_constellation(archive, focus_session_id=confluence.plan.session_id)
    second_edges = [edge for edge in graph["edges"] if edge["mode"] == "confluence-b" and edge["source"] == parent_b.plan.session_id and edge["target"] == confluence.plan.session_id]
    assert len(second_edges) == 1
    assert graph["counts"]["confluence_second_parent_edges"] == 1
    warning_codes = {warning["code"] for warning in graph["warnings"] if warning["session_id"] == confluence.plan.session_id}
    assert "parent-lineage-mismatch" not in warning_codes
    assert "generation-gap" not in warning_codes


def test_confluence_rejects_using_the_same_session_twice(tmp_path):
    _, parent_a, _ = _parents(tmp_path)
    with pytest.raises(ValueError, match="two distinct remembered sessions"):
        suggest_confluence(parent_a, parent_a)
