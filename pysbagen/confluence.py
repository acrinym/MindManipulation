"""Dual-ancestor Living Session synthesis for memorable hybrid experiences."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal

from .constellation import build_constellation
from .living_sessions import AffectSnapshot, LivingSessionArchive, LivingSessionPlan, StoredSession
from .sleep import SleepLayers, SleepRequest, build_sleep_recipe, recipe_manifest

InheritanceSource = Literal["A", "B", "both", "new"]

TRAIT_KEYS = ("problem", "sound_world", "intensity", "duration_minutes", "layers")
_TRAIT_LABELS = {
    "problem": "intent",
    "sound_world": "sound world",
    "intensity": "presence",
    "duration_minutes": "duration",
    "layers": "layer blend",
}


@dataclass(frozen=True)
class ConfluenceInheritance:
    """One experiential dimension carried into a Confluence session."""

    trait: str
    source: InheritanceSource
    value: Any
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConfluenceSuggestion:
    """Transparent suggestion for how two remembered sessions might meet."""

    parent_a_session_id: str
    parent_b_session_id: str
    assignments: tuple[ConfluenceInheritance, ...]
    tensions: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_a_session_id": self.parent_a_session_id,
            "parent_b_session_id": self.parent_b_session_id,
            "assignments": [item.to_dict() for item in self.assignments],
            "tensions": list(self.tensions),
            "rationale": self.rationale,
        }


def suggest_confluence(
    parent_a: StoredSession,
    parent_b: StoredSession,
    *,
    from_a: Iterable[str] = (),
    from_b: Iterable[str] = (),
) -> ConfluenceSuggestion:
    """Suggest a comprehensible blend using memory, outcome, and audible difference."""

    _validate_parents(parent_a, parent_b)
    request_a = _request_from_plan(parent_a.plan)
    request_b = _request_from_plan(parent_b.plan)
    explicit_a = _normalize_trait_set(from_a)
    explicit_b = _normalize_trait_set(from_b)
    overlap = explicit_a & explicit_b
    if overlap:
        raise ValueError(
            "A trait cannot be explicitly inherited from both parents: "
            + ", ".join(sorted(overlap))
        )

    values_a = _trait_values(request_a)
    values_b = _trait_values(request_b)
    score_a = _memory_score(parent_a)
    score_b = _memory_score(parent_b)
    preferred = "A" if score_a >= score_b else "B"
    assignments: list[ConfluenceInheritance] = []
    tensions: list[str] = []

    for trait in TRAIT_KEYS:
        value_a = values_a[trait]
        value_b = values_b[trait]
        if value_a == value_b:
            assignments.append(
                ConfluenceInheritance(
                    trait,
                    "both",
                    value_a,
                    f"Both memories already share this {_TRAIT_LABELS[trait]}.",
                )
            )
            continue

        tensions.append(
            f"{_TRAIT_LABELS[trait]} differs: A={_display(value_a)} · B={_display(value_b)}"
        )

        if trait in explicit_a:
            assignments.append(
                ConfluenceInheritance(
                    trait,
                    "A",
                    value_a,
                    f"Explicitly keep Parent A's {_TRAIT_LABELS[trait]}.",
                )
            )
            continue
        if trait in explicit_b:
            assignments.append(
                ConfluenceInheritance(
                    trait,
                    "B",
                    value_b,
                    f"Explicitly keep Parent B's {_TRAIT_LABELS[trait]}.",
                )
            )
            continue

        bridge = _bridge_value(trait, value_a, value_b)
        if bridge is not None:
            assignments.append(
                ConfluenceInheritance(
                    trait,
                    "new",
                    bridge,
                    f"Create a bridge between the two remembered {_TRAIT_LABELS[trait]} choices.",
                )
            )
            continue

        source = _suggest_source(
            trait,
            preferred=preferred,
            parent_a=parent_a,
            parent_b=parent_b,
            existing=assignments,
        )
        value = value_a if source == "A" else value_b
        assignments.append(
            ConfluenceInheritance(
                trait,
                source,
                value,
                _inheritance_reason(trait, source, parent_a, parent_b),
            )
        )

    assignments = _ensure_both_parents_are_present(
        assignments,
        values_a=values_a,
        values_b=values_b,
        parent_a=parent_a,
        parent_b=parent_b,
        protected_a=explicit_a,
        protected_b=explicit_b,
    )
    inherited_a = [_TRAIT_LABELS[x.trait] for x in assignments if x.source == "A"]
    inherited_b = [_TRAIT_LABELS[x.trait] for x in assignments if x.source == "B"]
    newly_bridged = [_TRAIT_LABELS[x.trait] for x in assignments if x.source == "new"]
    shared = [_TRAIT_LABELS[x.trait] for x in assignments if x.source == "both"]
    parts = []
    if inherited_a:
        parts.append("A carries " + ", ".join(inherited_a))
    if inherited_b:
        parts.append("B carries " + ", ".join(inherited_b))
    if newly_bridged:
        parts.append("the meeting creates " + ", ".join(newly_bridged))
    if shared:
        parts.append("both already share " + ", ".join(shared))
    parts.append("the generation seed is new so the result is its own reproducible experience")
    return ConfluenceSuggestion(
        parent_a_session_id=parent_a.plan.session_id,
        parent_b_session_id=parent_b.plan.session_id,
        assignments=tuple(assignments),
        tensions=tuple(tensions),
        rationale="; ".join(parts) + ".",
    )


def create_confluence_plan(
    parent_a: StoredSession,
    parent_b: StoredSession,
    *,
    suggestion: ConfluenceSuggestion | None = None,
    from_a: Iterable[str] = (),
    from_b: Iterable[str] = (),
    pre_affect: AffectSnapshot | None = None,
    created_at: str | None = None,
) -> tuple[LivingSessionPlan, ConfluenceSuggestion]:
    """Create a dual-ancestor plan without writing it to an archive."""

    _validate_parents(parent_a, parent_b)
    suggestion = suggestion or suggest_confluence(
        parent_a,
        parent_b,
        from_a=from_a,
        from_b=from_b,
    )
    if suggestion.parent_a_session_id != parent_a.plan.session_id:
        raise ValueError("Confluence suggestion does not belong to Parent A")
    if suggestion.parent_b_session_id != parent_b.plan.session_id:
        raise ValueError("Confluence suggestion does not belong to Parent B")

    request_a = _request_from_plan(parent_a.plan)
    request_b = _request_from_plan(parent_b.plan)
    values_a = _trait_values(request_a)
    values_b = _trait_values(request_b)
    selected: dict[str, Any] = {}
    for item in suggestion.assignments:
        if item.trait not in TRAIT_KEYS:
            raise ValueError(f"Unsupported Confluence trait: {item.trait}")
        if item.source == "A":
            selected[item.trait] = values_a[item.trait]
        elif item.source == "B":
            selected[item.trait] = values_b[item.trait]
        elif item.source == "both":
            if values_a[item.trait] != values_b[item.trait]:
                raise ValueError(f"Trait {item.trait} is not shared by both parents")
            selected[item.trait] = values_a[item.trait]
        elif item.source == "new":
            selected[item.trait] = item.value
        else:
            raise ValueError(f"Unknown Confluence source: {item.source}")

    missing = [trait for trait in TRAIT_KEYS if trait not in selected]
    if missing:
        raise ValueError("Confluence suggestion is incomplete: " + ", ".join(missing))

    seed = _new_seed(parent_a.plan, parent_b.plan, selected)
    request = SleepRequest(
        problem=str(selected["problem"]),
        sound_world=str(selected["sound_world"]),
        intensity=str(selected["intensity"]),
        duration_minutes=float(selected["duration_minutes"]),
        user_audio=_selected_user_audio(
            selected["sound_world"],
            request_a=request_a,
            request_b=request_b,
            assignments=suggestion.assignments,
        ),
        layers=_layers_from_value(selected["layers"]),
        seed=seed,
    )
    request.validate()
    manifest = recipe_manifest(build_sleep_recipe(request))
    recipe_sha = _hash(manifest)
    timestamp = created_at or _utc_now()
    generation = max(parent_a.plan.generation, parent_b.plan.generation) + 1
    lineage_id = _hash(
        {
            "kind": "confluence",
            "parent_a_lineage": parent_a.plan.lineage_id,
            "parent_b_lineage": parent_b.plan.lineage_id,
            "recipe_sha256": recipe_sha,
        },
        24,
    )
    title, motif = _hybrid_identity(parent_a.plan, parent_b.plan, recipe_sha)
    session_id = _hash(
        {
            "kind": "confluence-session",
            "lineage_id": lineage_id,
            "parent_a": parent_a.plan.session_id,
            "parent_b": parent_b.plan.session_id,
            "generation": generation,
            "recipe_sha256": recipe_sha,
            "created_at": timestamp,
        },
        24,
    )
    backend_policy = (
        parent_a.plan.backend_policy
        if parent_a.plan.backend_policy == parent_b.plan.backend_policy
        else "auto"
    )
    plan = LivingSessionPlan(
        session_id=session_id,
        lineage_id=lineage_id,
        generation=generation,
        parent_session_id=parent_a.plan.session_id,
        mode="confluence",
        title=title,
        motif=motif,
        created_at=timestamp,
        backend_policy=backend_policy,
        recipe_manifest=manifest,
        recipe_sha256=recipe_sha,
        mutations=(),
        rationale=suggestion.rationale,
        experimental=True,
        pre_affect=pre_affect,
    )
    return plan, suggestion


def create_confluence_session(
    archive: LivingSessionArchive,
    parent_a_session_id: str,
    parent_b_session_id: str,
    *,
    from_a: Iterable[str] = (),
    from_b: Iterable[str] = (),
    pre_affect: AffectSnapshot | None = None,
    created_at: str | None = None,
) -> StoredSession:
    """Create and persist a new session that remembers both ancestors."""

    parent_a = archive.get(parent_a_session_id)
    parent_b = archive.get(parent_b_session_id)
    plan, suggestion = create_confluence_plan(
        parent_a,
        parent_b,
        from_a=from_a,
        from_b=from_b,
        pre_affect=pre_affect,
        created_at=created_at,
    )
    archive.create(plan)
    archive.append_event(
        plan.session_id,
        kind="confluence",
        label=f"{parent_a.plan.title} × {parent_b.plan.title}",
        payload={
            "schema": "pysbagen.living-session-confluence.v1",
            "parent_a": _parent_summary(parent_a),
            "parent_b": _parent_summary(parent_b),
            "inheritance": [item.to_dict() for item in suggestion.assignments],
            "tensions": list(suggestion.tensions),
            "rationale": suggestion.rationale,
            "new_seed": plan.recipe_manifest["request"]["seed"],
            "backend_policy_resolution": (
                "parents agree"
                if parent_a.plan.backend_policy == parent_b.plan.backend_policy
                else "parents differ; Confluence uses auto rather than silently preferring either renderer policy"
            ),
        },
    )
    return archive.get(plan.session_id)


def confluence_metadata(stored: StoredSession) -> dict[str, Any] | None:
    """Return the persisted Confluence event payload for a session, if present."""

    events = [event for event in stored.events if event.kind == "confluence"]
    if not events:
        return None
    return max(events, key=lambda event: (event.created_at, event.event_id)).payload


def build_confluence_constellation(
    archive: LivingSessionArchive,
    *,
    focus_session_id: str | None = None,
) -> dict[str, Any]:
    """Enrich the ordinary constellation with second-parent Confluence edges."""

    graph = build_constellation(archive, focus_session_id=focus_session_id)
    visible_ids = {node["session_id"] for node in graph["nodes"]}
    edges = list(graph["edges"])
    existing = {edge["edge_id"] for edge in edges}
    confluence_count = 0
    for stored in archive.list_sessions():
        if stored.plan.session_id not in visible_ids:
            continue
        metadata = confluence_metadata(stored)
        if not metadata:
            continue
        parent_b = str((metadata.get("parent_b") or {}).get("session_id") or "")
        if not parent_b or parent_b not in visible_ids:
            continue
        edge_id = f"{parent_b}->{stored.plan.session_id}:confluence-b"
        if edge_id in existing:
            continue
        edges.append(
            {
                "edge_id": edge_id,
                "source": parent_b,
                "target": stored.plan.session_id,
                "mode": "confluence-b",
                "short_label": "confluence · B",
                "mutations": [],
                "change_count": 0,
                "experimental": True,
                "recipe_identity_preserved": False,
                "causal_interpretability": "multi-ancestor",
            }
        )
        existing.add(edge_id)
        confluence_count += 1

    confluence_ids = {
        stored.plan.session_id
        for stored in archive.list_sessions()
        if confluence_metadata(stored) is not None
    }
    expected_cross_lineage_codes = {"parent-lineage-mismatch", "generation-gap"}
    graph["warnings"] = [
        warning
        for warning in graph["warnings"]
        if not (
            warning["session_id"] in confluence_ids
            and warning["code"] in expected_cross_lineage_codes
        )
    ]
    for node in graph["nodes"]:
        if node["session_id"] in confluence_ids:
            node["warnings"] = [
                warning
                for warning in node["warnings"]
                if warning["code"] not in expected_cross_lineage_codes
            ]

    graph["edges"] = sorted(edges, key=lambda edge: (edge["source"], edge["target"], edge["edge_id"]))
    graph["counts"]["edges"] = len(graph["edges"])
    graph["counts"]["warnings"] = len(graph["warnings"])
    graph["counts"]["confluence_second_parent_edges"] = confluence_count
    graph["schema"] = "pysbagen.living-session-constellation.confluence.v1"
    snapshot_payload = {
        "nodes": graph["nodes"],
        "edges": graph["edges"],
        "warnings": graph["warnings"],
        "lineages": graph["lineages"],
    }
    graph["snapshot_sha256"] = hashlib.sha256(
        json.dumps(snapshot_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return graph


def describe_confluence(stored: StoredSession) -> dict[str, Any]:
    """Return a user-facing hybrid identity and inheritance summary."""

    metadata = confluence_metadata(stored)
    if metadata is None:
        raise ValueError(f"Session {stored.plan.session_id} is not a Confluence session")
    return {
        "session_id": stored.plan.session_id,
        "title": stored.plan.title,
        "memory_phrase": stored.plan.memory_phrase,
        "lineage_id": stored.plan.lineage_id,
        "generation": stored.plan.generation,
        "recipe_sha256": stored.plan.recipe_sha256,
        "status": stored.status,
        "parent_a": metadata["parent_a"],
        "parent_b": metadata["parent_b"],
        "inheritance": metadata["inheritance"],
        "tensions": metadata["tensions"],
        "rationale": metadata["rationale"],
        "backend_policy": stored.plan.backend_policy,
        "outcome": stored.outcome.to_dict() if stored.outcome else None,
    }


def _validate_parents(parent_a: StoredSession, parent_b: StoredSession) -> None:
    if parent_a.plan.session_id == parent_b.plan.session_id:
        raise ValueError("Confluence requires two distinct remembered sessions")
    _request_from_plan(parent_a.plan)
    _request_from_plan(parent_b.plan)


def _request_from_plan(plan: LivingSessionPlan) -> SleepRequest:
    manifest = plan.recipe_manifest
    if manifest.get("format") != "pysbagen-sleep-recipe-v1":
        raise ValueError("Confluence currently supports Living Sessions with sleep recipe manifests")
    payload = dict(manifest.get("request") or {})
    layers = payload.get("layers")
    if layers is not None:
        payload["layers"] = SleepLayers(**layers)
    request = SleepRequest(**payload)
    request.validate()
    return request


def _trait_values(request: SleepRequest) -> dict[str, Any]:
    layers = request.layers or SleepLayers()
    return {
        "problem": request.problem,
        "sound_world": request.sound_world,
        "intensity": request.intensity,
        "duration_minutes": float(request.duration_minutes),
        "layers": asdict(layers),
    }


def _normalize_trait_set(values: Iterable[str]) -> set[str]:
    result = {value.strip() for value in values if value and value.strip()}
    unknown = result - set(TRAIT_KEYS)
    if unknown:
        raise ValueError(
            "Unknown Confluence trait(s): "
            + ", ".join(sorted(unknown))
            + ". Choose from: "
            + ", ".join(TRAIT_KEYS)
        )
    return result


def _bridge_value(trait: str, value_a: Any, value_b: Any) -> Any | None:
    if trait == "intensity":
        order = ["gentle", "balanced", "immersive"]
        left, right = order.index(str(value_a)), order.index(str(value_b))
        midpoint = order[(left + right) // 2]
        if midpoint not in {value_a, value_b}:
            return midpoint
    if trait == "duration_minutes":
        midpoint = round((float(value_a) + float(value_b)) / 2.0 / 5.0) * 5.0
        if 10.0 <= midpoint <= 180.0 and midpoint not in {float(value_a), float(value_b)}:
            return midpoint
    if trait == "layers":
        a = dict(value_a)
        b = dict(value_b)
        union = {key: bool(a.get(key) or b.get(key)) for key in sorted(set(a) | set(b))}
        if union != a and union != b and any(union.values()):
            return union
    return None


def _suggest_source(
    trait: str,
    *,
    preferred: str,
    parent_a: StoredSession,
    parent_b: StoredSession,
    existing: list[ConfluenceInheritance],
) -> Literal["A", "B"]:
    used = {item.source for item in existing}
    if trait == "sound_world":
        echoes_a = sum(event.kind == "echo" for event in parent_a.events)
        echoes_b = sum(event.kind == "echo" for event in parent_b.events)
        if echoes_a != echoes_b:
            return "A" if echoes_a > echoes_b else "B"
    if "A" not in used:
        return "A"
    if "B" not in used:
        return "B"
    if trait in {"problem", "intensity"}:
        return preferred
    return "B" if preferred == "A" else "A"


def _ensure_both_parents_are_present(
    assignments: list[ConfluenceInheritance],
    *,
    values_a: dict[str, Any],
    values_b: dict[str, Any],
    parent_a: StoredSession,
    parent_b: StoredSession,
    protected_a: set[str],
    protected_b: set[str],
) -> list[ConfluenceInheritance]:
    sources = {item.source for item in assignments}
    if "A" in sources and "B" in sources:
        return assignments

    mutable = list(assignments)
    missing = "A" if "A" not in sources else "B"
    protected_other = protected_b if missing == "A" else protected_a
    for index, item in enumerate(mutable):
        if item.source not in {"A", "B"}:
            continue
        if item.trait in protected_other:
            continue
        a, b = values_a[item.trait], values_b[item.trait]
        if a == b:
            continue
        value = a if missing == "A" else b
        mutable[index] = ConfluenceInheritance(
            item.trait,
            missing,
            value,
            _inheritance_reason(item.trait, missing, parent_a, parent_b)
            + " This also keeps the Confluence genuinely dual-parent.",
        )
        return mutable
    return mutable


def _inheritance_reason(
    trait: str,
    source: str,
    parent_a: StoredSession,
    parent_b: StoredSession,
) -> str:
    parent = parent_a if source == "A" else parent_b
    echo_count = sum(event.kind == "echo" for event in parent.events)
    outcome = parent.outcome
    memory_bits = []
    if echo_count:
        memory_bits.append(f"{echo_count} remembered echo{'es' if echo_count != 1 else ''}")
    if outcome is not None:
        memory_bits.append(f"{outcome.rating}/5 outcome")
        if outcome.would_repeat:
            memory_bits.append("marked worth repeating")
    memory = ", ".join(memory_bits) if memory_bits else "its recognizable recipe identity"
    return f"Carry Parent {source}'s {_TRAIT_LABELS[trait]} because this memory contributes {memory}."


def _memory_score(stored: StoredSession) -> tuple[int, int, int, str]:
    outcome = stored.outcome
    rating = outcome.rating if outcome else 0
    repeat = int(bool(outcome and outcome.would_repeat))
    echoes = sum(event.kind == "echo" for event in stored.events)
    return rating, repeat, echoes, stored.plan.session_id


def _selected_user_audio(
    sound_world: Any,
    *,
    request_a: SleepRequest,
    request_b: SleepRequest,
    assignments: tuple[ConfluenceInheritance, ...],
) -> str | None:
    if sound_world != "user_audio":
        return None
    sound_assignment = next(item for item in assignments if item.trait == "sound_world")
    if sound_assignment.source == "A":
        return request_a.user_audio
    if sound_assignment.source == "B":
        return request_b.user_audio
    if request_a.user_audio and request_b.user_audio and request_a.user_audio == request_b.user_audio:
        return request_a.user_audio
    raise ValueError(
        "A new/shared user_audio Confluence requires both parents to reference the same available audio file"
    )


def _layers_from_value(value: Any) -> SleepLayers:
    if isinstance(value, SleepLayers):
        return value
    payload = dict(value)
    layers = SleepLayers(**payload)
    if not layers.enabled_names():
        raise ValueError("Confluence cannot disable every audio layer")
    return layers


def _new_seed(parent_a: LivingSessionPlan, parent_b: LivingSessionPlan, selected: dict[str, Any]) -> int:
    digest = _hash(
        {
            "kind": "confluence-seed",
            "parent_a_recipe": parent_a.recipe_sha256,
            "parent_b_recipe": parent_b.recipe_sha256,
            "selected": selected,
        }
    )
    return int(digest[:8], 16) % (2**31 - 2) + 1


def _hybrid_identity(
    parent_a: LivingSessionPlan,
    parent_b: LivingSessionPlan,
    recipe_sha256: str,
) -> tuple[str, tuple[str, ...]]:
    a_words = parent_a.title.split()
    b_words = parent_b.title.split()
    adjective = a_words[0] if a_words else "Meeting"
    noun = b_words[-1] if b_words else "Current"
    title = f"{adjective} {noun}"
    if title in {parent_a.title, parent_b.title}:
        adjective = b_words[0] if b_words else "Braided"
        noun = a_words[-1] if a_words else "Tide"
        title = f"{adjective} {noun}"
    if title in {parent_a.title, parent_b.title}:
        title = f"Confluence {recipe_sha256[:6]}"

    motifs: list[str] = []
    for motif in (*parent_a.motif, *parent_b.motif):
        if motif not in motifs:
            motifs.append(motif)
        if len(motifs) == 2:
            break
    bridge_motifs = ("meeting", "braid", "crossing", "confluence", "weave", "delta")
    bridge = bridge_motifs[int(recipe_sha256[:8], 16) % len(bridge_motifs)]
    if bridge not in motifs:
        motifs.append(bridge)
    while len(motifs) < 3:
        fallback = bridge_motifs[(int(recipe_sha256[len(motifs):len(motifs)+8], 16) + len(motifs)) % len(bridge_motifs)]
        if fallback not in motifs:
            motifs.append(fallback)
    return title, tuple(motifs[:3])


def _parent_summary(stored: StoredSession) -> dict[str, Any]:
    return {
        "session_id": stored.plan.session_id,
        "lineage_id": stored.plan.lineage_id,
        "title": stored.plan.title,
        "memory_phrase": stored.plan.memory_phrase,
        "recipe_sha256": stored.plan.recipe_sha256,
        "generation": stored.plan.generation,
        "rating": stored.outcome.rating if stored.outcome else None,
        "would_repeat": stored.outcome.would_repeat if stored.outcome else None,
        "echo_count": sum(event.kind == "echo" for event in stored.events),
    }


def _display(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _hash(payload: Any, length: int = 64) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
