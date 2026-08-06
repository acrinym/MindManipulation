"""Offline lineage navigation for PySbagen Living Sessions."""

from __future__ import annotations

import hashlib
import html
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
        return int(self.outcome["rating"]) if self.outcome and self.outcome.get("rating") is not None else None

    def to_dict(self, *, redact_notes: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        payload["echo_count"] = self.echo_count
        payload["render_count"] = self.render_count
        payload["rating"] = self.rating
        if redact_notes:
            payload["rationale"] = ""
            payload["events"] = [
                {
                    **event,
                    "label": "[redacted]",
                    "payload": {},
                }
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
                if key not in {"user_audio"}
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
            warnings.append(
                f"{node.session_id}: parent belongs to a different lineage"
            )
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
    lineages = _lineage_summaries(nodes)
    return ConstellationGraph(
        nodes=sorted(nodes, key=lambda node: (node.lineage_id, node.generation, node.created_at, node.session_id)),
        edges=sorted(edges, key=lambda edge: (edge.source_session_id, edge.target_session_id)),
        lineages=lineages,
        integrity_warnings=sorted(set(warnings)),
        focus_session_id=focus.plan.session_id if focus else None,
    )


def render_constellation_html(
    graph: ConstellationGraph,
    *,
    redact_notes: bool = False,
) -> str:
    """Render one self-contained offline HTML navigator with no remote dependencies."""

    payload = graph.to_dict(redact_notes=redact_notes)
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    title = "PySbagen Living Session Constellation"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #090b12;
  --panel: #111625;
  --panel2: #171e31;
  --text: #eef2ff;
  --muted: #9ba8c7;
  --line: #435174;
  --focus: #f4d27a;
  --good: #85d7a3;
  --warn: #f6a96b;
  --root: #9cc5ff;
  --return: #b7a5ff;
  --branch: #7ee0d6;
  --contrast: #ff9eab;
  --wander: #f3c56b;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 system-ui, sans-serif; }}
header {{ padding: 18px 22px; border-bottom: 1px solid #232c43; background: #0c101b; }}
header h1 {{ margin: 0 0 4px; font-size: 22px; }}
header p {{ margin: 0; color: var(--muted); }}
.toolbar {{ display: grid; grid-template-columns: minmax(180px, 1fr) repeat(3, minmax(120px, 180px)); gap: 10px; padding: 12px 18px; background: var(--panel); border-bottom: 1px solid #232c43; }}
input, select, button {{ width: 100%; border: 1px solid #35415f; border-radius: 7px; background: #0c1120; color: var(--text); padding: 9px 10px; }}
button {{ cursor: pointer; }}
main {{ display: grid; grid-template-columns: minmax(0, 1fr) 360px; height: calc(100vh - 132px); }}
.stage {{ overflow: auto; position: relative; }}
svg {{ min-width: 100%; min-height: 100%; }}
.edge {{ stroke: var(--line); stroke-width: 2; fill: none; opacity: .8; }}
.edge.experimental {{ stroke-dasharray: 8 6; }}
.node {{ cursor: pointer; }}
.node rect {{ fill: var(--panel2); stroke-width: 2; rx: 10; }}
.node[data-mode="root"] rect {{ stroke: var(--root); }}
.node[data-mode="return"] rect {{ stroke: var(--return); }}
.node[data-mode="branch"] rect {{ stroke: var(--branch); }}
.node[data-mode="contrast"] rect {{ stroke: var(--contrast); }}
.node[data-mode="wander"] rect {{ stroke: var(--wander); }}
.node.focus rect, .node.selected rect {{ stroke: var(--focus); stroke-width: 4; }}
.node.dim {{ opacity: .15; }}
.node text {{ fill: var(--text); pointer-events: none; }}
.node .sub {{ fill: var(--muted); font-size: 11px; }}
aside {{ border-left: 1px solid #232c43; background: var(--panel); overflow: auto; padding: 16px; }}
aside h2 {{ margin-top: 0; }}
.meta {{ display: grid; grid-template-columns: 110px 1fr; gap: 6px 10px; }}
.meta dt {{ color: var(--muted); }}
.meta dd {{ margin: 0; overflow-wrap: anywhere; }}
.card {{ background: var(--panel2); border: 1px solid #2c3651; border-radius: 8px; padding: 10px; margin: 10px 0; }}
.badge {{ display: inline-block; border: 1px solid #4b5a7d; border-radius: 999px; padding: 2px 7px; margin: 2px; color: var(--muted); }}
.warnings {{ color: var(--warn); }}
.empty {{ color: var(--muted); padding: 24px; }}
footer {{ color: var(--muted); font-size: 12px; margin-top: 18px; }}
@media (max-width: 900px) {{
  .toolbar {{ grid-template-columns: 1fr 1fr; }}
  main {{ grid-template-columns: 1fr; height: auto; }}
  .stage {{ height: 68vh; }}
  aside {{ border-left: 0; border-top: 1px solid #232c43; }}
}}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <p><span id="summary"></span> · snapshot <code>{payload["graph_sha256"][:16]}</code> · local/offline</p>
</header>
<section class="toolbar" aria-label="Constellation filters">
  <input id="search" type="search" placeholder="Search title, motif, session, mutation, event">
  <select id="lineage"><option value="">All lineages</option></select>
  <select id="mode"><option value="">All modes</option><option>root</option><option>return</option><option>branch</option><option>contrast</option><option>wander</option></select>
  <select id="status"><option value="">All states</option><option>planned</option><option>active</option><option>completed</option></select>
</section>
<main>
  <section class="stage" aria-label="Session lineage graph">
    <svg id="graph" role="img" aria-label="Living Session constellation"></svg>
  </section>
  <aside id="detail">
    <h2>Select a session</h2>
    <p>Click a node to inspect its exact recipe identity, ancestry, disclosed mutations, echoes, outcome, backend policy, and provenance.</p>
    <div class="warnings" id="warnings"></div>
    <footer>This file contains a selected local archive snapshot. Notes redacted: {str(redact_notes).lower()}.</footer>
  </aside>
</main>
<script id="constellation-data" type="application/json">{data_json}</script>
<script>
const data = JSON.parse(document.getElementById("constellation-data").textContent);
const svg = document.getElementById("graph");
const detail = document.getElementById("detail");
const search = document.getElementById("search");
const lineage = document.getElementById("lineage");
const mode = document.getElementById("mode");
const status = document.getElementById("status");
const byId = new Map(data.nodes.map(n => [n.session_id, n]));
const focusId = data.focus_session_id || "";
const nodeEls = new Map();
const edgeEls = [];
const ns = "http://www.w3.org/2000/svg";

document.getElementById("summary").textContent =
  `${data.summary.session_count} sessions · ${data.summary.lineage_count} lineages · ${data.summary.echo_count} echoes`;

for (const item of data.lineages) {{
  const option = document.createElement("option");
  option.value = item.lineage_id;
  option.textContent = `${item.root_title || item.lineage_id} (${item.session_count})`;
  lineage.appendChild(option);
}}

const maxX = Math.max(900, ...data.nodes.map(n => n.x + 300));
const maxY = Math.max(600, ...data.nodes.map(n => n.y + 180));
svg.setAttribute("viewBox", `0 0 ${maxX} ${maxY}`);
svg.setAttribute("width", maxX);
svg.setAttribute("height", maxY);

for (const edge of data.edges) {{
  const source = byId.get(edge.source_session_id);
  const target = byId.get(edge.target_session_id);
  if (!source || !target) continue;
  const path = document.createElementNS(ns, "path");
  const sx = source.x + 220, sy = source.y + 44;
  const tx = target.x, ty = target.y + 44;
  const mid = (sx + tx) / 2;
  path.setAttribute("d", `M ${sx} ${sy} C ${mid} ${sy}, ${mid} ${ty}, ${tx} ${ty}`);
  path.setAttribute("class", `edge ${edge.experimental ? "experimental" : ""}`);
  path.dataset.source = edge.source_session_id;
  path.dataset.target = edge.target_session_id;
  path.dataset.label = edge.label;
  svg.appendChild(path);
  edgeEls.push(path);
}}

for (const node of data.nodes) {{
  const group = document.createElementNS(ns, "g");
  group.setAttribute("class", `node ${node.session_id === focusId ? "focus" : ""}`);
  group.setAttribute("transform", `translate(${node.x},${node.y})`);
  group.dataset.id = node.session_id;
  group.dataset.mode = node.mode;
  const rect = document.createElementNS(ns, "rect");
  rect.setAttribute("width", "220");
  rect.setAttribute("height", "88");
  const titleText = document.createElementNS(ns, "text");
  titleText.setAttribute("x", "12");
  titleText.setAttribute("y", "25");
  titleText.textContent = node.title;
  const metaText = document.createElementNS(ns, "text");
  metaText.setAttribute("x", "12");
  metaText.setAttribute("y", "47");
  metaText.setAttribute("class", "sub");
  metaText.textContent = `g${node.generation} · ${node.mode} · ${node.status}`;
  const countText = document.createElementNS(ns, "text");
  countText.setAttribute("x", "12");
  countText.setAttribute("y", "68");
  countText.setAttribute("class", "sub");
  countText.textContent = `${node.echo_count} echoes · ${node.render_count} renders${node.rating ? ` · ${node.rating}/5` : ""}`;
  group.append(rect, titleText, metaText, countText);
  group.addEventListener("click", () => selectNode(node.session_id));
  svg.appendChild(group);
  nodeEls.set(node.session_id, group);
}}

function searchable(node) {{
  return JSON.stringify({{
    title: node.title,
    motif: node.motif,
    id: node.session_id,
    lineage: node.lineage_id,
    mutations: node.mutations,
    events: node.events,
    tags: node.outcome?.tags || []
  }}).toLowerCase();
}}

function applyFilters() {{
  const q = search.value.trim().toLowerCase();
  for (const node of data.nodes) {{
    const visible =
      (!lineage.value || node.lineage_id === lineage.value) &&
      (!mode.value || node.mode === mode.value) &&
      (!status.value || node.status === status.value) &&
      (!q || searchable(node).includes(q));
    nodeEls.get(node.session_id).style.display = visible ? "" : "none";
    node._visible = visible;
  }}
  for (const edgeEl of edgeEls) {{
    edgeEl.style.display =
      byId.get(edgeEl.dataset.source)._visible && byId.get(edgeEl.dataset.target)._visible ? "" : "none";
  }}
}}
search.addEventListener("input", applyFilters);
lineage.addEventListener("change", applyFilters);
mode.addEventListener("change", applyFilters);
status.addEventListener("change", applyFilters);

function esc(value) {{
  return String(value ?? "").replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
}}
function pretty(value) {{ return esc(typeof value === "string" ? value : JSON.stringify(value)); }}

function selectNode(id) {{
  const node = byId.get(id);
  for (const [nodeId, el] of nodeEls) {{
    el.classList.toggle("selected", nodeId === id);
    el.classList.toggle("dim", nodeId !== id && nodeId !== node.parent_session_id && !data.edges.some(e => e.source_session_id === id && e.target_session_id === nodeId));
  }}
  const mutations = node.mutations.length
    ? node.mutations.map(m => `<div class="card"><strong>${esc(m.key)}</strong><br>${pretty(m.before)} → ${pretty(m.after)}<br><small>${esc(m.reason)}</small></div>`).join("")
    : "<p>No recipe mutation; this is an exact root/return relationship.</p>";
  const events = node.events.length
    ? node.events.map(e => `<div class="card"><strong>${esc(e.kind)}</strong>${e.position_seconds == null ? "" : ` @ ${e.position_seconds}s`}<br>${esc(e.label)}${Object.keys(e.payload || {{}}).length ? `<pre>${esc(JSON.stringify(e.payload, null, 2))}</pre>` : ""}</div>`).join("")
    : "<p>No events recorded.</p>";
  const outcome = node.outcome
    ? `<div class="card"><strong>${node.outcome.rating}/5</strong> · ${esc(node.outcome.comfort)} · repeat ${node.outcome.would_repeat ? "yes" : "no"}<br>${esc(node.outcome.note || "")}<br>${(node.outcome.tags || []).map(t => `<span class="badge">${esc(t)}</span>`).join("")}</div>`
    : "<p>No outcome recorded.</p>";
  detail.innerHTML = `
    <h2>${esc(node.memory_phrase)}</h2>
    <dl class="meta">
      <dt>Session</dt><dd><code>${esc(node.session_id)}</code></dd>
      <dt>Lineage</dt><dd><code>${esc(node.lineage_id)}</code></dd>
      <dt>Parent</dt><dd>${node.parent_session_id ? `<code>${esc(node.parent_session_id)}</code>` : "root"}</dd>
      <dt>Generation</dt><dd>${node.generation}</dd>
      <dt>Mode</dt><dd>${esc(node.mode)}${node.experimental ? " · experimental" : ""}</dd>
      <dt>Status</dt><dd>${esc(node.status)}</dd>
      <dt>Backend</dt><dd>${esc(node.backend_policy)}</dd>
      <dt>Recipe</dt><dd><code>${esc(node.recipe_sha256)}</code></dd>
      <dt>Created</dt><dd>${esc(node.created_at)}</dd>
    </dl>
    <h3>Journey</h3>
    <div class="card">${Object.entries(node.request_summary).map(([k,v]) => `<span class="badge">${esc(k)}=${pretty(v)}</span>`).join("")}</div>
    <p>${esc(node.rationale)}</p>
    <h3>Disclosed changes</h3>${mutations}
    <h3>Echoes and events</h3>${events}
    <h3>Outcome</h3>${outcome}
    <button id="copy-id">Copy session ID</button>
    ${data.integrity_warnings.length ? `<div class="warnings"><h3>Snapshot warnings</h3><ul>${data.integrity_warnings.map(w => `<li>${esc(w)}</li>`).join("")}</ul></div>` : ""}
    <footer>Snapshot hash: <code>${data.graph_sha256}</code><br>Notes redacted: ${data.privacy.notes_redacted}.</footer>`;
  document.getElementById("copy-id").addEventListener("click", async () => {{
    try {{ await navigator.clipboard.writeText(node.session_id); }}
    catch (_) {{ /* clipboard may be unavailable for file://; ID remains visible */ }}
  }});
}}

applyFilters();
if (focusId && byId.has(focusId)) selectNode(focusId);
else if (data.nodes.length) selectNode(data.nodes[0].session_id);

if (data.integrity_warnings.length) {{
  document.getElementById("warnings").innerHTML =
    `<h3>Snapshot warnings</h3><ul>${data.integrity_warnings.map(w => `<li>${esc(w)}</li>`).join("")}</ul>`;
}}
</script>
</body>
</html>
"""


def write_constellation_html(
    graph: ConstellationGraph,
    destination: str | Path,
    *,
    redact_notes: bool = False,
) -> Path:
    """Atomically write a self-contained HTML constellation snapshot."""

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
            render_constellation_html(graph, redact_notes=redact_notes),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path.resolve()


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
    layers = request.get("layers")
    request_summary = {
        "problem": request.get("problem"),
        "sound_world": request.get("sound_world"),
        "intensity": request.get("intensity"),
        "duration_minutes": request.get("duration_minutes"),
        "seed": request.get("seed"),
        "layers": layers,
        "user_audio": request.get("user_audio"),
    }
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
        request_summary=request_summary,
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
    keys = ", ".join(mutation["key"].removeprefix("request.") for mutation in node.mutations)
    return f"{node.mode}: {keys}"


def _lineage_summaries(nodes: Iterable[ConstellationNode]) -> list[dict[str, Any]]:
    grouped: dict[str, list[ConstellationNode]] = {}
    for node in nodes:
        grouped.setdefault(node.lineage_id, []).append(node)
    summaries: list[dict[str, Any]] = []
    for lineage_id, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda node: (node.generation, node.created_at, node.session_id))
        roots = [node for node in ordered if node.parent_session_id is None]
        summaries.append(
            {
                "lineage_id": lineage_id,
                "session_count": len(ordered),
                "completed_count": sum(node.status == "completed" for node in ordered),
                "echo_count": sum(node.echo_count for node in ordered),
                "max_generation": max((node.generation for node in ordered), default=0),
                "root_session_ids": [node.session_id for node in roots],
                "root_title": roots[0].title if roots else None,
                "latest_session_id": max(ordered, key=lambda node: (node.created_at, node.session_id)).session_id,
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
        children.setdefault(edge.source_session_id, []).append(edge.target_session_id)

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
