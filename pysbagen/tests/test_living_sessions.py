from __future__ import annotations

from pathlib import Path

import pytest

from pysbagen.living_sessions import (
    AffectSnapshot,
    LivingSessionArchive,
    SessionOutcome,
    create_child_sleep_plan,
    create_sleep_plan,
    recommend_child_mode,
    sleep_request_from_manifest,
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


def test_root_identity_is_memorable_and_reproducible():
    before = AffectSnapshot(valence=-0.4, arousal=0.8, agency=0.3)
    first = create_sleep_plan(_request(), pre_affect=before, created_at="2026-07-31T19:00:00+00:00")
    second = create_sleep_plan(_request(), pre_affect=before, created_at="2026-07-31T19:00:00+00:00")

    assert first == second
    assert len(first.title.split()) == 2
    assert len(first.motif) == 3
    assert first.memory_phrase.startswith(first.title)
    assert len(first.recipe_sha256) == 64
    assert first.mode == "root"


def test_branch_changes_exactly_one_disclosed_dimension():
    root = create_sleep_plan(_request(), created_at="2026-07-31T19:00:00+00:00")
    child = create_child_sleep_plan(
        root,
        mode="branch",
        created_at="2026-07-31T19:01:00+00:00",
    )

    assert child.parent_session_id == root.session_id
    assert child.lineage_id == root.lineage_id
    assert child.generation == 1
    assert len(child.mutations) == 1
    assert child.recipe_sha256 != root.recipe_sha256
    sleep_request_from_manifest(child.recipe_manifest).validate()


def test_return_preserves_recipe_and_identity():
    root = create_sleep_plan(_request(), created_at="2026-07-31T19:00:00+00:00")
    returned = create_child_sleep_plan(
        root,
        mode="return",
        created_at="2026-07-31T19:02:00+00:00",
    )

    assert returned.recipe_sha256 == root.recipe_sha256
    assert returned.title == root.title
    assert returned.motif == root.motif
    assert returned.mutations == ()
    assert returned.session_id != root.session_id


def test_wander_combines_no_more_than_two_disclosed_changes():
    root = create_sleep_plan(_request(), created_at="2026-07-31T19:00:00+00:00")
    child = create_child_sleep_plan(
        root,
        mode="wander",
        created_at="2026-07-31T19:03:00+00:00",
    )

    assert child.experimental
    assert 1 <= len(child.mutations) <= 2
    assert len({item.key for item in child.mutations}) == len(child.mutations)
    sleep_request_from_manifest(child.recipe_manifest).validate()


def test_contrast_uses_one_audible_product_dimension():
    root = create_sleep_plan(_request(), created_at="2026-07-31T19:00:00+00:00")
    child = create_child_sleep_plan(
        root,
        mode="contrast",
        created_at="2026-07-31T19:04:00+00:00",
    )

    assert len(child.mutations) == 1
    assert child.mutations[0].key != "request.seed"
    sleep_request_from_manifest(child.recipe_manifest).validate()


def test_archive_keeps_echoes_outcomes_and_descriptive_atlas(tmp_path: Path):
    archive = LivingSessionArchive(tmp_path / "living")
    root = create_sleep_plan(
        _request(),
        pre_affect=AffectSnapshot(valence=-0.5, arousal=0.9, agency=0.2),
        created_at="2026-07-31T19:00:00+00:00",
    )
    archive.create(root)
    archive.append_event(
        root.session_id,
        kind="echo",
        label="The rain became a room",
        position_seconds=123.5,
        created_at="2026-07-31T19:02:00+00:00",
    )
    outcome = SessionOutcome(
        session_id=root.session_id,
        completed_at="2026-07-31T20:00:00+00:00",
        rating=5,
        would_repeat=True,
        comfort="comfortable",
        post_affect=AffectSnapshot(valence=0.4, arousal=0.2, agency=0.7),
        tags=("rain", "settled"),
    )
    stored = archive.finish(outcome)
    atlas = archive.atlas()

    assert stored.status == "completed"
    assert archive.echoes()[0]["event"]["label"] == "The rain became a room"
    assert atlas["session_count"] == 1
    assert atlas["completed_count"] == 1
    assert atlas["echo_count"] == 1
    assert atlas["average_rating"] == 5
    assert atlas["average_affect_delta"]["valence"] == pytest.approx(0.9)


def test_auto_mode_returns_after_a_strong_first_outcome_then_branches(tmp_path: Path):
    archive = LivingSessionArchive(tmp_path / "living")
    root = create_sleep_plan(_request(), created_at="2026-07-31T19:00:00+00:00")
    archive.create(root)
    archive.finish(
        SessionOutcome(
            session_id=root.session_id,
            completed_at="2026-07-31T20:00:00+00:00",
            rating=5,
            would_repeat=True,
            comfort="comfortable",
        )
    )
    stored_root = archive.get(root.session_id)
    assert recommend_child_mode(stored_root, archive) == "return"

    returned = create_child_sleep_plan(
        root,
        mode="return",
        archive=archive,
        created_at="2026-07-31T21:00:00+00:00",
    )
    archive.create(returned)
    archive.finish(
        SessionOutcome(
            session_id=returned.session_id,
            completed_at="2026-07-31T22:00:00+00:00",
            rating=5,
            would_repeat=True,
            comfort="comfortable",
        )
    )
    assert recommend_child_mode(archive.get(returned.session_id), archive) == "branch"


def test_outcome_is_immutable(tmp_path: Path):
    archive = LivingSessionArchive(tmp_path / "living")
    root = create_sleep_plan(_request(), created_at="2026-07-31T19:00:00+00:00")
    archive.create(root)
    archive.finish(
        SessionOutcome(
            session_id=root.session_id,
            completed_at="2026-07-31T20:00:00+00:00",
            rating=4,
            would_repeat=True,
            comfort="comfortable",
        )
    )
    with pytest.raises(ValueError, match="immutable"):
        archive.finish(
            SessionOutcome(
                session_id=root.session_id,
                completed_at="2026-07-31T20:00:00+00:00",
                rating=1,
                would_repeat=False,
                comfort="uncomfortable",
            )
        )


def test_native_required_plan_refuses_python_render(tmp_path: Path):
    archive_root = tmp_path / "living"
    archive = LivingSessionArchive(archive_root)
    root = create_sleep_plan(
        _request(),
        backend_policy="sbagenx",
        created_at="2026-07-31T19:00:00+00:00",
    )
    archive.create(root)

    with pytest.raises(SystemExit, match="requires SBaGenX"):
        session_main(["--root", str(archive_root), "render", root.session_id])
