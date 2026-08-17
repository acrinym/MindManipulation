"""Truth-derived graph model for PySbagen Living Session constellations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .living_sessions import LivingSessionArchive, StoredSession


@dataclass(frozen=True)
class ConstellationNode:
    """One preserved Living Session represented as a navigable constellation node."""

    session_id: str
    lineage_id: str
    parent_session_id: str | None
    generation: int
    mode: str
    title: str
    memory_phrase: str
    motif: tuple[str, ...]
    status: str
    created_at: str
    backend_policy: str
    recipe_sha256: str
    experimental: bool
    rationale: str
    mutations: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]
    outcome: dict[str, Any] | None
    pre_affect: dict[str, Any] | None
    request_summary: dict[str, Any]
    x: int = 0
    y: int = 0

    @property
    def echo_count(self) -> int:
        return sum(event.get("kind") == "echo" for event in self.events)

    @property
    def render_count(self) -> int:
        return sum(event.get("kind") == "render" for event in self.events)

    @property
    def rating(self) -> int | None:
        if not self.outcome or self.outcome.get("rating") is None:
            return None
        return int(self.outcome["rating"])

    def to_dict(self, *, redact_notes: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        payload["echo_count"] = self.echo_count
        payload["render_count"] = self.render_count
        payload["rating"] = self.rating
        if redact_notes:
            payload["rationale"] = ""
            payload["events"] = [
                {**event, "label": "[redacted]", "payload": {}}
                for event in payload["events"]
            ]
            payload["pre_affect"] = _redact_affect_note(payload.get("pre_affect"))
            if payload["outcome"]:
                payload["outcome"]["note"] = None
                payload["outcome"]["post_affect"] = _redact_affect_note(
                    payload["outcome"].get("post_affect")
                )
            payload["request_summary"] = {
                key: value
                for key, value in payload["request_summary"].items()
                if key != "user_audio"
            }
        return payload


@dataclass(frozen=True)
class ConstellationEdge:
    """A parent-to-child relationship with disclosed mutation context."""

    source_session_id: str
    target_session_id: str
    mode: str
    label: str
    experimental: bool
    mutations: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConstellationGraph:
    """Deterministic snapshot of one or more local Living Session lineages."""

    nodes: list[ConstellationNode]
    edges: list[ConstellationEdge]
    lineages: list[dict[str, Any]]
    integrity_warnings: list[str] = field(default_factory=list)
    focus_session_id: str | None = None
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = "pysbagen.living-session-constellation.v1"

    @property
    def graph_sha256(self) -> str:
        identity = {
            "schema_version": self.schema_version,
            "nodes": [node.to_dict(redact_notes=False) for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "lineages": self.lineages,
            "integrity_warnings": self.integrity_warnings,
            "focus_session_id": self.focus_session_id,
        }
        encoded = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self, *, redact_notes: bool = False) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "graph_sha256": self.graph_sha256,
            "focus_session_id": self.focus_session_id,
            "summary": {
                "session_count": len(self.nodes),
                "edge_count": len(self.edges),
                "lineage_count": len(self.lineages),
                "completed_count": sum(node.status == "completed" for node in self.nodes),
                "echo_count": sum(node.echo_count for node in self.nodes),
                "warning_count": len(self.integrity_warnings),
            },
            "lineages": self.lineages,
            "nodes": [node.to_dict(redact_notes=redact_notes) for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "integrity_warnings": list(self.integrity_warnings),
            "privacy": {
                "notes_redacted": redact_notes,
                "scope": "local archive snapshot",
            },
        }


def build_constellation(
    archive: LivingSessionArchive,
    *,
    lineage_id: str | None = None,
    focus_session_id: str | None = None,
) -> ConstellationGraph:
    """Build a deterministic graph from the archive without inventing relationships."""

    sessions = archive.list_sessions()
    focus: StoredSession | None = None
    if focus_session_id is not None:
        focus = archive.get(focus_session_id)
        if lineage_id is not None and lineage_id != focus.plan.lineage_id:
            raise ValueError(
                f"focus session belongs to lineage {focus.plan.lineage_id}, not {lineage_id}"
            )
        lineage_id = focus.plan.lineage_id

    if lineage_id is not None:
        sessions = [item for item in sessions if item.plan.lineage_id == lineage_id]
        if not sessions:
            raise KeyError(f"Living-session lineage not found: {lineage_id}")

    nodes = [_node_from_session(item) for item in sessions]
    _apply_layout(nodes)
    by_id = {node.session_id: node for node in nodes}
    edges: list[ConstellationEdge] = []
    warnings: list[str] = []

    for node in nodes:
        parent_id = node.parent_session_id
        if parent_id is None:
            if node.generation != 0:
                warnings.append(
                    f"{node.session_id}: generation {node.generation} has no parent"
                )
            continue
        parent = by_id.get(parent_id)
        if parent is None:
            warnings.append(
                f"{node.session_id}: parent {parent_id} is outside this snapshot or missing"
            )
            continue
        if parent.lineage_id != node.lineage_id:
            warnings.append(f"{node.session_id}: parent belongs to a different lineage")
        if node.generation != parent.generation + 1:
            warnings.append(
                f"{node.session_id}: generation {node.generation} does not follow parent generation {parent.generation}"
            )
        edges.append(
            ConstellationEdge(
                source_session_id=parent.session_id,
                target_session_id=node.session_id,
                mode=node.mode,
                label=_edge_label(node),
                experimental=node.experimental,
                mutations=node.mutations,
            )
        )

    _detect_cycles(nodes, edges, warnings)
    return ConstellationGraph(
        nodes=sorted(
            nodes,
            key=lambda node: (
                node.lineage_id,
                node.generation,
                node.created_at,
                node.session_id,
            ),
        ),
        edges=sorted(
            edges,
            key=lambda edge: (edge.source_session_id, edge.target_session_id),
        ),
        lineages=_lineage_summaries(nodes),
        integrity_warnings=sorted(set(warnings)),
        focus_session_id=focus.plan.session_id if focus else None,
    )


def write_constellation_json(
    graph: ConstellationGraph,
    destination: str | Path,
    *,
    redact_notes: bool = False,
) -> Path:
    """Atomically write the machine-readable graph snapshot."""

    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(
                graph.to_dict(redact_notes=redact_notes),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path.resolve()


def _node_from_session(item: StoredSession) -> ConstellationNode:
    request = dict(item.plan.recipe_manifest.get("request") or {})
    return ConstellationNode(
        session_id=item.plan.session_id,
        lineage_id=item.plan.lineage_id,
        parent_session_id=item.plan.parent_session_id,
        generation=item.plan.generation,
        mode=item.plan.mode,
        title=item.plan.title,
        memory_phrase=item.plan.memory_phrase,
        motif=item.plan.motif,
        status=item.status,
        created_at=item.plan.created_at,
        backend_policy=item.plan.backend_policy,
        recipe_sha256=item.plan.recipe_sha256,
        experimental=item.plan.experimental,
        rationale=item.plan.rationale,
        mutations=tuple(asdict(mutation) for mutation in item.plan.mutations),
        events=tuple(event.to_dict() for event in item.events),
        outcome=item.outcome.to_dict() if item.outcome else None,
        pre_affect=item.plan.pre_affect.to_dict() if item.plan.pre_affect else None,
        request_summary={
            "problem": request.get("problem"),
            "sound_world": request.get("sound_world"),
            "intensity": request.get("intensity"),
            "duration_minutes": request.get("duration_minutes"),
            "seed": request.get("seed"),
            "layers": request.get("layers"),
            "user_audio": request.get("user_audio"),
        },
    )


def _apply_layout(nodes: list[ConstellationNode]) -> None:
    lineages: dict[str, list[ConstellationNode]] = {}
    for node in nodes:
        lineages.setdefault(node.lineage_id, []).append(node)

    lineage_offset = 0
    for lineage_id in sorted(lineages):
        items = sorted(
            lineages[lineage_id],
            key=lambda node: (node.generation, node.created_at, node.session_id),
        )
        by_generation: dict[int, list[ConstellationNode]] = {}
        for node in items:
            by_generation.setdefault(node.generation, []).append(node)
        max_rows = max((len(group) for group in by_generation.values()), default=1)
        for generation, group in sorted(by_generation.items()):
            for row, node in enumerate(group):
                object.__setattr__(node, "x", 70 + generation * 300)
                object.__setattr__(node, "y", 70 + lineage_offset + row * 125)
        lineage_offset += max(180, max_rows * 125 + 90)


def _edge_label(node: ConstellationNode) -> str:
    if not node.mutations:
        return node.mode
    keys = ", ".join(
        mutation["key"].removeprefix("request.") for mutation in node.mutations
    )
    return f"{node.mode}: {keys}"


def _lineage_summaries(nodes: Iterable[ConstellationNode]) -> list[dict[str, Any]]:
    grouped: dict[str, list[ConstellationNode]] = {}
    for node in nodes:
        grouped.setdefault(node.lineage_id, []).append(node)

    summaries: list[dict[str, Any]] = []
    for lineage_id, items in sorted(grouped.items()):
        ordered = sorted(
            items,
            key=lambda node: (node.generation, node.created_at, node.session_id),
        )
        roots = [node for node in ordered if node.parent_session_id is None]
        summaries.append(
            {
                "lineage_id": lineage_id,
                "session_count": len(ordered),
                "completed_count": sum(node.status == "completed" for node in ordered),
                "echo_count": sum(node.echo_count for node in ordered),
                "max_generation": max(
                    (node.generation for node in ordered), default=0
                ),
                "root_session_ids": [node.session_id for node in roots],
                "root_title": roots[0].title if roots else None,
                "latest_session_id": max(
                    ordered,
                    key=lambda node: (node.created_at, node.session_id),
                ).session_id,
            }
        )
    return summaries


def _detect_cycles(
    nodes: list[ConstellationNode],
    edges: list[ConstellationEdge],
    warnings: list[str],
) -> None:
    children: dict[str, list[str]] = {node.session_id: [] for node in nodes}
    for edge in edges:
        children.setdefault(edge.source_session_id, []).append(
            edge.target_session_id
        )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(session_id: str) -> None:
        if session_id in visiting:
            warnings.append(f"{session_id}: cycle detected in parent relationships")
            return
        if session_id in visited:
            return
        visiting.add(session_id)
        for child in children.get(session_id, []):
            visit(child)
        visiting.remove(session_id)
        visited.add(session_id)

    for node in nodes:
        visit(node.session_id)


def _redact_affect_note(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return payload
    return {**payload, "note": None}
