"""Centralized variation policy hardening for Living Sessions.

The first Living Sessions model deliberately keeps its data and archive machinery
in ``living_sessions``. This module owns the product-level mutation selection
policy so retention behavior can evolve without coupling it to storage.
"""

from __future__ import annotations

import hashlib
from typing import Any

from . import living_sessions as _living
from .living_sessions import SessionMutation


def install_living_session_policy() -> None:
    """Install the current explicit mutation and identity policies.

    The assignment is intentional: ``create_child_sleep_plan`` resolves the
    selector from its module globals at call time. Keeping the policy here makes
    high-salience contrast rules independently testable and prevents storage
    code from accumulating engagement heuristics.
    """

    _living._select_mutations = select_mutations  # type: ignore[attr-defined]
    _living._identity = memorable_identity  # type: ignore[attr-defined]


def select_mutations(
    candidates: list[SessionMutation],
    usage: dict[str, int],
    mode: str,
    parent_session_id: str,
) -> list[SessionMutation]:
    """Select disclosed changes while keeping contrast audibly meaningful."""

    if not candidates:
        raise ValueError("No valid living-session mutations are available")

    eligible = candidates
    if mode == "contrast":
        audible = [item for item in candidates if item.key != "request.seed"]
        eligible = audible or candidates

    ranked = sorted(
        eligible,
        key=lambda item: (
            usage.get(item.key, 0),
            contrast_priority(item.key) if mode == "contrast" else 0,
            _living._short_hash(  # type: ignore[attr-defined]
                {"parent": parent_session_id, "mode": mode, "key": item.key},
                16,
            ),
        ),
    )
    if mode != "wander":
        return ranked[:1]

    first = ranked[0]
    for candidate in ranked[1:]:
        strong_pair = {first.key, candidate.key}
        if strong_pair in (
            {"request.intensity", "request.layers.harmonic_box"},
            {"request.intensity", "request.layers.isochronic"},
        ):
            continue
        return [first, candidate]
    return [first]


def contrast_priority(key: str) -> int:
    """Rank audible contrast dimensions ahead of subtle or temporal changes."""

    if key == "request.sound_world_bundle":
        return 0
    if key == "request.intensity":
        return 1
    if key.startswith("request.layers."):
        return 2
    if key == "request.duration_minutes":
        return 3
    return 9


def memorable_identity(recipe_sha256: str, *, generation: int) -> tuple[str, tuple[str, ...]]:
    """Create a stable human identity and always provide three unique motifs."""

    digest = hashlib.sha256(f"{recipe_sha256}:{generation}".encode("utf-8")).digest()
    title = (
        f"{_living._ADJECTIVES[digest[0] % len(_living._ADJECTIVES)]} "  # type: ignore[attr-defined]
        f"{_living._NOUNS[digest[1] % len(_living._NOUNS)]}"  # type: ignore[attr-defined]
    )
    motifs: list[str] = []
    for value in digest[2:]:
        motif = _living._MOTIFS[value % len(_living._MOTIFS)]  # type: ignore[attr-defined]
        if motif not in motifs:
            motifs.append(motif)
        if len(motifs) == 3:
            break
    if len(motifs) < 3:
        for motif in _living._MOTIFS:  # type: ignore[attr-defined]
            if motif not in motifs:
                motifs.append(motif)
            if len(motifs) == 3:
                break
    return title, tuple(motifs)
