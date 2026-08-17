"""Local-first living-session identity, lineage, memory, and outcome tracking."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .sleep import SleepLayers, SleepRequest, build_sleep_recipe, recipe_manifest

SessionMode = Literal["root", "return", "branch", "contrast", "wander"]
SessionComfort = Literal["comfortable", "neutral", "uncomfortable"]
BackendPolicy = Literal["python", "sbagenx", "auto"]

_ADJECTIVES = (
    "Amber",
    "Blue",
    "Distant",
    "Ember",
    "Hidden",
    "Liminal",
    "Moonless",
    "Moss",
    "Quiet",
    "Silver",
    "Soft",
    "Velvet",
)
_NOUNS = (
    "Archive",
    "Bridge",
    "Cairn",
    "Chamber",
    "Constellation",
    "Current",
    "Garden",
    "Harbor",
    "Lantern",
    "Signal",
    "Threshold",
    "Tide",
)
_MOTIFS = (
    "drift",
    "ember",
    "hush",
    "low-glow",
    "rain",
    "return",
    "ripple",
    "shadow",
    "stillness",
    "threshold",
    "tide",
    "warmth",
)


@dataclass(frozen=True)
class AffectSnapshot:
    """Small, non-medical emotional-state snapshot used for personal pattern recall."""

    valence: float
    arousal: float
    agency: float
    note: str | None = None

    def __post_init__(self) -> None:
        if not -1.0 <= float(self.valence) <= 1.0:
            raise ValueError("valence must be between -1 and 1")
        if not 0.0 <= float(self.arousal) <= 1.0:
            raise ValueError("arousal must be between 0 and 1")
        if not 0.0 <= float(self.agency) <= 1.0:
            raise ValueError("agency must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "AffectSnapshot | None":
        return cls(**payload) if payload else None


@dataclass(frozen=True)
class SessionMutation:
    """One disclosed change between a parent session and its child."""

    key: str
    before: Any
    after: Any
    reason: str


@dataclass(frozen=True)
class LivingSessionPlan:
    """Reproducible identity and recipe for one session in a continuing lineage."""

    session_id: str
    lineage_id: str
    generation: int
    parent_session_id: str | None
    mode: SessionMode
    title: str
    motif: tuple[str, ...]
    created_at: str
    backend_policy: BackendPolicy
    recipe_manifest: dict[str, Any]
    recipe_sha256: str
    mutations: tuple[SessionMutation, ...] = ()
    rationale: str = ""
    experimental: bool = False
    pre_affect: AffectSnapshot | None = None
    schema_version: str = "pysbagen.living-session-plan.v1"

    @property
    def memory_phrase(self) -> str:
        return f"{self.title} · {' / '.join(self.motif)}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["motif"] = list(self.motif)
        payload["mutations"] = [asdict(item) for item in self.mutations]
        payload["pre_affect"] = self.pre_affect.to_dict() if self.pre_affect else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LivingSessionPlan":
        data = dict(payload)
        data["motif"] = tuple(data.get("motif") or ())
        data["mutations"] = tuple(SessionMutation(**item) for item in data.get("mutations") or ())
        data["pre_affect"] = AffectSnapshot.from_dict(data.get("pre_affect"))
        return cls(**data)


@dataclass(frozen=True)
class SessionEvent:
    """Append-only event or memorable echo attached to a session."""

    event_id: str
    session_id: str
    kind: str
    created_at: str
    label: str
    position_seconds: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "pysbagen.living-session-event.v1"

    def __post_init__(self) -> None:
        if self.position_seconds is not None and float(self.position_seconds) < 0:
            raise ValueError("position_seconds cannot be negative")
        if not self.kind.strip():
            raise ValueError("event kind cannot be empty")
        if not self.label.strip():
            raise ValueError("event label cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionEvent":
        return cls(**payload)


@dataclass(frozen=True)
class SessionOutcome:
    """Optional local check-in that teaches the archive without claiming efficacy."""

    session_id: str
    completed_at: str
    rating: int
    would_repeat: bool
    comfort: SessionComfort
    post_affect: AffectSnapshot | None = None
    note: str | None = None
    tags: tuple[str, ...] = ()
    schema_version: str = "pysbagen.living-session-outcome.v1"

    def __post_init__(self) -> None:
        if not 1 <= int(self.rating) <= 5:
            raise ValueError("rating must be between 1 and 5")
        if self.comfort not in {"comfortable", "neutral", "uncomfortable"}:
            raise ValueError(f"unknown comfort state: {self.comfort}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tags"] = list(self.tags)
        payload["post_affect"] = self.post_affect.to_dict() if self.post_affect else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionOutcome":
        data = dict(payload)
        data["tags"] = tuple(data.get("tags") or ())
        data["post_affect"] = AffectSnapshot.from_dict(data.get("post_affect"))
        return cls(**data)


@dataclass(frozen=True)
class StoredSession:
    plan: LivingSessionPlan
    events: tuple[SessionEvent, ...] = ()
    outcome: SessionOutcome | None = None

    @property
    def status(self) -> str:
        if self.outcome is not None:
            return "completed"
        if any(event.kind == "render" for event in self.events):
            return "active"
        return "planned"


def default_living_sessions_root() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "PySbagen" / "living-sessions"
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "pysbagen" / "living-sessions"


class LivingSessionArchive:
    """Local content-addressed archive for plans, events, echoes, and outcomes."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).expanduser() if root is not None else default_living_sessions_root()
        self.sessions_root = self.root / "sessions"
        self.sessions_root.mkdir(parents=True, exist_ok=True)

    def create(self, plan: LivingSessionPlan) -> StoredSession:
        path = self.sessions_root / plan.session_id
        if path.exists():
            existing = self.get(plan.session_id)
            if existing.plan.to_dict() != plan.to_dict():
                raise ValueError(f"Session ID collision: {plan.session_id}")
            return existing
        temporary = Path(tempfile.mkdtemp(prefix=f".{plan.session_id}-", dir=self.sessions_root))
        try:
            _write_json_atomic(temporary / "plan.json", plan.to_dict())
            (temporary / "events.jsonl").write_text("", encoding="utf-8")
            os.replace(temporary, path)
        except Exception:
            if temporary.exists():
                import shutil

                shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self.get(plan.session_id)

    def get(self, session_id: str) -> StoredSession:
        path = self.sessions_root / session_id
        plan_path = path / "plan.json"
        if not plan_path.is_file():
            raise KeyError(f"Living session not found: {session_id}")
        plan = LivingSessionPlan.from_dict(json.loads(plan_path.read_text(encoding="utf-8")))
        events: list[SessionEvent] = []
        events_path = path / "events.jsonl"
        if events_path.is_file():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    events.append(SessionEvent.from_dict(json.loads(line)))
        outcome_path = path / "outcome.json"
        outcome = (
            SessionOutcome.from_dict(json.loads(outcome_path.read_text(encoding="utf-8")))
            if outcome_path.is_file()
            else None
        )
        return StoredSession(plan, tuple(events), outcome)

    def list_sessions(self) -> list[StoredSession]:
        sessions: list[StoredSession] = []
        for path in sorted(self.sessions_root.iterdir()):
            if not path.is_dir() or path.name.startswith("."):
                continue
            try:
                sessions.append(self.get(path.name))
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                continue
        return sorted(sessions, key=lambda item: (item.plan.created_at, item.plan.session_id))

    def lineage(self, lineage_id: str) -> list[StoredSession]:
        return [item for item in self.list_sessions() if item.plan.lineage_id == lineage_id]

    def append_event(
        self,
        session_id: str,
        *,
        kind: str,
        label: str,
        position_seconds: float | None = None,
        payload: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> SessionEvent:
        self.get(session_id)
        timestamp = created_at or _utc_now()
        event_payload = payload or {}
        event_id = _short_hash(
            {
                "session_id": session_id,
                "kind": kind,
                "label": label,
                "position_seconds": position_seconds,
                "payload": event_payload,
                "created_at": timestamp,
            },
            24,
        )
        event = SessionEvent(
            event_id=event_id,
            session_id=session_id,
            kind=kind,
            created_at=timestamp,
            label=label,
            position_seconds=position_seconds,
            payload=event_payload,
        )
        path = self.sessions_root / session_id / "events.jsonl"
        existing_ids = {item.event_id for item in self.get(session_id).events}
        if event.event_id not in existing_ids:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(_canonical_json(event.to_dict()) + "\n")
        return event

    def finish(self, outcome: SessionOutcome) -> StoredSession:
        current = self.get(outcome.session_id)
        path = self.sessions_root / outcome.session_id / "outcome.json"
        if current.outcome is not None and current.outcome.to_dict() != outcome.to_dict():
            raise ValueError("Session already has a different outcome; outcomes are immutable")
        _write_json_atomic(path, outcome.to_dict())
        return self.get(outcome.session_id)

    def echoes(self) -> list[dict[str, Any]]:
        echoes: list[dict[str, Any]] = []
        for item in self.list_sessions():
            for event in item.events:
                if event.kind != "echo":
                    continue
                echoes.append(
                    {
                        "session_id": item.plan.session_id,
                        "lineage_id": item.plan.lineage_id,
                        "title": item.plan.title,
                        "memory_phrase": item.plan.memory_phrase,
                        "event": event.to_dict(),
                    }
                )
        return sorted(echoes, key=lambda item: item["event"]["created_at"])

    def atlas(self) -> dict[str, Any]:
        sessions = self.list_sessions()
        completed = [item for item in sessions if item.outcome is not None]
        ratings = [item.outcome.rating for item in completed if item.outcome is not None]
        repeat_votes = [item.outcome.would_repeat for item in completed if item.outcome is not None]
        affect_deltas: list[dict[str, float]] = []
        for item in completed:
            before = item.plan.pre_affect
            after = item.outcome.post_affect if item.outcome else None
            if before and after:
                affect_deltas.append(
                    {
                        "valence": after.valence - before.valence,
                        "arousal": after.arousal - before.arousal,
                        "agency": after.agency - before.agency,
                    }
                )
        lineages: dict[str, list[StoredSession]] = {}
        for item in sessions:
            lineages.setdefault(item.plan.lineage_id, []).append(item)
        return {
            "schema": "pysbagen.living-session-atlas.v1",
            "session_count": len(sessions),
            "planned_count": sum(item.status == "planned" for item in sessions),
            "active_count": sum(item.status == "active" for item in sessions),
            "completed_count": len(completed),
            "lineage_count": len(lineages),
            "echo_count": len(self.echoes()),
            "average_rating": (sum(ratings) / len(ratings)) if ratings else None,
            "would_repeat_rate": (sum(repeat_votes) / len(repeat_votes)) if repeat_votes else None,
            "average_affect_delta": _average_deltas(affect_deltas),
            "lineages": [
                {
                    "lineage_id": lineage_id,
                    "session_count": len(items),
                    "completed_count": sum(item.outcome is not None for item in items),
                    "titles": [item.plan.title for item in items],
                    "latest_session_id": max(items, key=lambda item: item.plan.created_at).plan.session_id,
                }
                for lineage_id, items in sorted(lineages.items())
            ],
            "pattern_candidates": _pattern_candidates(completed),
        }


def create_sleep_plan(
    request: SleepRequest,
    *,
    pre_affect: AffectSnapshot | None = None,
    backend_policy: BackendPolicy = "python",
    created_at: str | None = None,
) -> LivingSessionPlan:
    """Create the root of a reproducible sleep-session lineage."""

    recipe = build_sleep_recipe(request)
    manifest = recipe_manifest(recipe)
    recipe_sha = _short_hash(manifest, 64)
    lineage_id = _short_hash({"kind": "sleep", "recipe_sha256": recipe_sha}, 24)
    timestamp = created_at or _utc_now()
    title, motif = _identity(recipe_sha, generation=0)
    session_id = _session_id(
        lineage_id=lineage_id,
        parent_session_id=None,
        generation=0,
        mode="root",
        recipe_sha256=recipe_sha,
        created_at=timestamp,
    )
    return LivingSessionPlan(
        session_id=session_id,
        lineage_id=lineage_id,
        generation=0,
        parent_session_id=None,
        mode="root",
        title=title,
        motif=motif,
        created_at=timestamp,
        backend_policy=backend_policy,
        recipe_manifest=manifest,
        recipe_sha256=recipe_sha,
        rationale="A stable root identity for a session that can be returned to, branched, contrasted, or wandered from.",
        pre_affect=pre_affect,
    )


def create_child_sleep_plan(
    parent: LivingSessionPlan,
    *,
    mode: Literal["return", "branch", "contrast", "wander"] = "branch",
    archive: LivingSessionArchive | None = None,
    pre_affect: AffectSnapshot | None = None,
    created_at: str | None = None,
) -> LivingSessionPlan:
    """Create a disclosed child variant while preserving lineage identity."""

    if mode not in {"return", "branch", "contrast", "wander"}:
        raise ValueError(f"unknown living-session mode: {mode}")
    request = sleep_request_from_manifest(parent.recipe_manifest)
    used = _mutation_usage(archive.lineage(parent.lineage_id) if archive else [])
    mutations: tuple[SessionMutation, ...]
    if mode == "return":
        child_request = request
        mutations = ()
        rationale = "Return to the exact remembered recipe before introducing more novelty."
        experimental = False
    else:
        candidates = _mutation_candidates(parent, request, mode)
        selected = _select_mutations(candidates, used, mode, parent.session_id)
        child_request = _apply_mutations(request, selected)
        mutations = tuple(selected)
        experimental = mode == "wander"
        rationale = {
            "branch": "Change one disclosed dimension so the archive can learn what mattered.",
            "contrast": "Change one high-salience dimension to test a clearly different route.",
            "wander": "Combine two compatible disclosed changes for exploration; treat the result as less causally interpretable.",
        }[mode]

    manifest = recipe_manifest(build_sleep_recipe(child_request))
    recipe_sha = _short_hash(manifest, 64)
    generation = parent.generation + 1
    timestamp = created_at or _utc_now()
    if mode == "return":
        title, motif = parent.title, parent.motif
    else:
        title, motif = _identity(recipe_sha, generation=generation)
    session_id = _session_id(
        lineage_id=parent.lineage_id,
        parent_session_id=parent.session_id,
        generation=generation,
        mode=mode,
        recipe_sha256=recipe_sha,
        created_at=timestamp,
    )
    return LivingSessionPlan(
        session_id=session_id,
        lineage_id=parent.lineage_id,
        generation=generation,
        parent_session_id=parent.session_id,
        mode=mode,
        title=title,
        motif=motif,
        created_at=timestamp,
        backend_policy=parent.backend_policy,
        recipe_manifest=manifest,
        recipe_sha256=recipe_sha,
        mutations=mutations,
        rationale=rationale,
        experimental=experimental,
        pre_affect=pre_affect,
    )


def recommend_child_mode(parent: StoredSession, archive: LivingSessionArchive) -> str:
    """Choose a transparent next mode from local feedback, never a medical claim."""

    outcome = parent.outcome
    if outcome is None:
        return "branch"
    lineage = archive.lineage(parent.plan.lineage_id)
    exact_returns = sum(
        item.plan.recipe_sha256 == parent.plan.recipe_sha256 and item.plan.mode == "return"
        for item in lineage
    )
    if outcome.comfort == "uncomfortable" or outcome.rating <= 2:
        return "contrast"
    if outcome.rating >= 4 and outcome.would_repeat:
        return "return" if exact_returns == 0 else "branch"
    return "branch"


def sleep_request_from_manifest(manifest: dict[str, Any]) -> SleepRequest:
    """Reconstruct a validated SleepRequest from a stored exact recipe manifest."""

    if manifest.get("format") != "pysbagen-sleep-recipe-v1":
        raise ValueError("Living sleep sessions require a pysbagen-sleep-recipe-v1 manifest")
    payload = dict(manifest.get("request") or {})
    layers_payload = payload.get("layers")
    if layers_payload is not None:
        payload["layers"] = SleepLayers(**layers_payload)
    request = SleepRequest(**payload)
    request.validate()
    return request


def _mutation_candidates(
    parent: LivingSessionPlan,
    request: SleepRequest,
    mode: str,
) -> list[SessionMutation]:
    rng = random.Random(int(_short_hash({"session": parent.session_id, "mode": mode}, 16), 16))
    candidates: list[SessionMutation] = []

    new_seed = rng.randrange(1, 2**31 - 1)
    if new_seed == request.seed:
        new_seed = (new_seed + 1) % (2**31 - 1)
    candidates.append(
        SessionMutation(
            "request.seed",
            request.seed,
            new_seed,
            "Change the generated bed/phases while preserving the selected journey structure.",
        )
    )

    duration_options = [
        value
        for value in (float(request.duration_minutes) - 15.0, float(request.duration_minutes) + 15.0)
        if 10.0 <= value <= 180.0 and value != float(request.duration_minutes)
    ]
    if duration_options:
        after = rng.choice(duration_options)
        candidates.append(
            SessionMutation(
                "request.duration_minutes",
                float(request.duration_minutes),
                after,
                "Test whether a shorter or longer support window fits this lineage better.",
            )
        )

    intensities = ["gentle", "balanced", "immersive"]
    current_index = intensities.index(request.intensity)
    if mode == "contrast":
        intensity_after = intensities[-1 - current_index]
        if intensity_after == request.intensity:
            intensity_after = intensities[0 if current_index > 0 else 2]
    else:
        adjacent = [
            intensities[index]
            for index in (current_index - 1, current_index + 1)
            if 0 <= index < len(intensities)
        ]
        intensity_after = rng.choice(adjacent) if adjacent else request.intensity
    if intensity_after != request.intensity:
        candidates.append(
            SessionMutation(
                "request.intensity",
                request.intensity,
                intensity_after,
                "Change how present the underlying layers feel without changing the stated sleep problem.",
            )
        )

    internal_worlds = ["warm_ambient", "slow_night_music", "rain_room", "deep_night"]
    world_options = [world for world in internal_worlds if world != request.sound_world]
    if world_options:
        world_after = rng.choice(world_options)
        candidates.append(
            SessionMutation(
                "request.sound_world_bundle",
                {"sound_world": request.sound_world, "user_audio": request.user_audio},
                {"sound_world": world_after, "user_audio": None},
                "Move the same intent into a different remembered sound world, with any prior user-audio binding disclosed.",
            )
        )

    layers = request.layers or SleepLayers()
    if mode == "contrast":
        layer_name = "harmonic_box" if layers.harmonic_box else "isochronic"
    else:
        layer_name = rng.choice(["harmonic_box", "isochronic"])
    layer_after = not getattr(layers, layer_name)
    layers_after = replace(layers, **{layer_name: layer_after})
    if layers_after.enabled_names():
        candidates.append(
            SessionMutation(
                f"request.layers.{layer_name}",
                getattr(layers, layer_name),
                layer_after,
                f"{'Enable' if layer_after else 'Remove'} the {layer_name.replace('_', ' ')} layer as one disclosed test.",
            )
        )
    return candidates


def _select_mutations(
    candidates: list[SessionMutation],
    usage: dict[str, int],
    mode: str,
    parent_session_id: str,
) -> list[SessionMutation]:
    if not candidates:
        raise ValueError("No valid living-session mutations are available")
    ranked = sorted(
        candidates,
        key=lambda item: (
            usage.get(item.key, 0),
            _short_hash({"parent": parent_session_id, "mode": mode, "key": item.key}, 16),
        ),
    )
    if mode != "wander":
        return ranked[:1]
    first = ranked[0]
    for candidate in ranked[1:]:
        strong_pair = {first.key, candidate.key}
        if strong_pair == {"request.intensity", "request.layers.harmonic_box"}:
            continue
        if strong_pair == {"request.intensity", "request.layers.isochronic"}:
            continue
        return [first, candidate]
    return [first]


def _apply_mutations(request: SleepRequest, mutations: list[SessionMutation]) -> SleepRequest:
    current = request
    for mutation in mutations:
        if mutation.key == "request.seed":
            current = replace(current, seed=int(mutation.after))
        elif mutation.key == "request.duration_minutes":
            current = replace(current, duration_minutes=float(mutation.after))
        elif mutation.key == "request.intensity":
            current = replace(current, intensity=str(mutation.after))
        elif mutation.key == "request.sound_world_bundle":
            after = dict(mutation.after)
            current = replace(
                current,
                sound_world=str(after["sound_world"]),
                user_audio=after.get("user_audio"),
            )
        elif mutation.key.startswith("request.layers."):
            layer_name = mutation.key.rsplit(".", 1)[-1]
            layers = current.layers or SleepLayers()
            current = replace(current, layers=replace(layers, **{layer_name: bool(mutation.after)}))
        else:
            raise ValueError(f"Unsupported living-session mutation: {mutation.key}")
    current.validate()
    return current


def _mutation_usage(sessions: list[StoredSession]) -> dict[str, int]:
    usage: dict[str, int] = {}
    for item in sessions:
        for mutation in item.plan.mutations:
            usage[mutation.key] = usage.get(mutation.key, 0) + 1
    return usage


def _identity(recipe_sha256: str, *, generation: int) -> tuple[str, tuple[str, ...]]:
    digest = hashlib.sha256(f"{recipe_sha256}:{generation}".encode("utf-8")).digest()
    title = f"{_ADJECTIVES[digest[0] % len(_ADJECTIVES)]} {_NOUNS[digest[1] % len(_NOUNS)]}"
    motifs: list[str] = []
    for value in digest[2:]:
        motif = _MOTIFS[value % len(_MOTIFS)]
        if motif not in motifs:
            motifs.append(motif)
        if len(motifs) == 3:
            break
    return title, tuple(motifs)


def _session_id(
    *,
    lineage_id: str,
    parent_session_id: str | None,
    generation: int,
    mode: str,
    recipe_sha256: str,
    created_at: str,
) -> str:
    return _short_hash(
        {
            "lineage_id": lineage_id,
            "parent_session_id": parent_session_id,
            "generation": generation,
            "mode": mode,
            "recipe_sha256": recipe_sha256,
            "created_at": created_at,
        },
        24,
    )


def _pattern_candidates(completed: list[StoredSession]) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    by_world: dict[str, list[int]] = {}
    by_mode: dict[str, list[int]] = {}
    for item in completed:
        if item.outcome is None:
            continue
        request = item.plan.recipe_manifest.get("request") or {}
        world = str(request.get("sound_world", "unknown"))
        by_world.setdefault(world, []).append(item.outcome.rating)
        by_mode.setdefault(item.plan.mode, []).append(item.outcome.rating)
    for label, values in sorted(by_world.items()):
        if len(values) >= 2:
            patterns.append(
                {
                    "kind": "sound-world",
                    "label": label,
                    "observations": len(values),
                    "average_rating": sum(values) / len(values),
                    "claim": "descriptive local pattern only",
                }
            )
    for label, values in sorted(by_mode.items()):
        if len(values) >= 2:
            patterns.append(
                {
                    "kind": "mode",
                    "label": label,
                    "observations": len(values),
                    "average_rating": sum(values) / len(values),
                    "claim": "descriptive local pattern only",
                }
            )
    return sorted(patterns, key=lambda item: (-item["average_rating"], item["kind"], item["label"]))


def _average_deltas(values: list[dict[str, float]]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        key: sum(item[key] for item in values) / len(values)
        for key in ("valence", "arousal", "agency")
    }


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _short_hash(payload: Any, length: int) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:length]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
