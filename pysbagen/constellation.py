"""Offline Living Sessions constellation graph and self-contained navigator."""

from __future__ import annotations

import hashlib
import html
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .living_sessions import LivingSessionArchive, StoredSession


@dataclass(frozen=True)
class ConstellationWarning:
    """One structural fact that should remain visible in the navigator."""

    code: str
    session_id: str
    detail: str


_NODE_WIDTH = 236
_NODE_HEIGHT = 112
_X_GAP = 300
_Y_GAP = 144
_LINEAGE_GAP = 88
_CANVAS_MARGIN = 72


def build_constellation(
    archive: LivingSessionArchive,
    *,
    lineage_id: str | None = None,
    focus_session_id: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic graph from the archive without changing its records."""

    sessions = archive.list_sessions()
    if lineage_id is not None:
        sessions = [item for item in sessions if item.plan.lineage_id == lineage_id]
        if not sessions:
            raise KeyError(f"Living-session lineage not found: {lineage_id}")

    by_id = {item.plan.session_id: item for item in sessions}
    if focus_session_id is not None and focus_session_id not in by_id:
        raise KeyError(f"Living session is not present in this constellation: {focus_session_id}")

    warnings = _structural_warnings(sessions, by_id)
    positions, canvas = _layout(sessions)
    nodes = [
        _node_payload(item, positions[item.plan.session_id], warnings)
        for item in sorted(sessions, key=_session_sort_key)
    ]
    edges = [
        _edge_payload(item, by_id[item.plan.parent_session_id])
        for item in sorted(sessions, key=_session_sort_key)
        if item.plan.parent_session_id in by_id
    ]
    lineages = _lineage_payloads(sessions)
    snapshot_source = {
        "nodes": nodes,
        "edges": edges,
        "warnings": [asdict(item) for item in warnings],
        "lineages": lineages,
    }
    snapshot_sha256 = hashlib.sha256(_canonical_json(snapshot_source).encode("utf-8")).hexdigest()
    return {
        "schema": "pysbagen.living-session-constellation.v1",
        "snapshot_sha256": snapshot_sha256,
        "scope": {
            "lineage_id": lineage_id,
            "focus_session_id": focus_session_id,
            "archive_root": str(archive.root.resolve()),
        },
        "counts": {
            "sessions": len(nodes),
            "edges": len(edges),
            "lineages": len(lineages),
            "completed": sum(node["status"] == "completed" for node in nodes),
            "echoes": sum(node["echo_count"] for node in nodes),
            "warnings": len(warnings),
        },
        "canvas": canvas,
        "lineages": lineages,
        "nodes": nodes,
        "edges": edges,
        "warnings": [asdict(item) for item in warnings],
        "interpretation_note": (
            "The constellation is a local navigation and provenance surface. "
            "Outcome patterns are descriptive personal records, not medical-efficacy conclusions."
        ),
    }


def constellation_to_text(graph: dict[str, Any]) -> str:
    """Render a compact terminal map that remains useful without a browser."""

    counts = graph["counts"]
    lines = [
        "PySbagen Living Sessions Constellation",
        f"Snapshot: {graph['snapshot_sha256']}",
        (
            f"Sessions: {counts['sessions']} · lineages: {counts['lineages']} · "
            f"connections: {counts['edges']} · echoes: {counts['echoes']}"
        ),
    ]
    if counts["warnings"]:
        lines.append(f"Structural warnings: {counts['warnings']}")
    nodes_by_id = {node["session_id"]: node for node in graph["nodes"]}
    for lineage in graph["lineages"]:
        lines.append("")
        lines.append(
            f"Lineage {lineage['lineage_id']} · {lineage['session_count']} session(s) · "
            f"latest {lineage['latest_session_id']}"
        )
        lineage_nodes = [
            node for node in graph["nodes"] if node["lineage_id"] == lineage["lineage_id"]
        ]
        for node in sorted(lineage_nodes, key=lambda item: (item["generation"], item["created_at"], item["session_id"])):
            parent = node["parent_session_id"]
            prefix = "ROOT" if parent is None else f"↳ {parent[:8]}"
            rating = f" · {node['outcome']['rating']}/5" if node["outcome"] is not None else ""
            echoes = f" · {node['echo_count']} echo(es)" if node["echo_count"] else ""
            lines.append(
                f"  g{node['generation']} {prefix:10} {node['mode']:8} "
                f"{node['memory_phrase']} [{node['status']}]{rating}{echoes}"
            )
            if parent is not None and parent in nodes_by_id:
                for mutation in node["mutations"]:
                    lines.append(
                        f"      {mutation['key']}: {_display_value(mutation['before'])} → "
                        f"{_display_value(mutation['after'])}"
                    )
    lines.append("")
    lines.append(graph["interpretation_note"])
    return "\n".join(lines)


def render_constellation_html(
    graph: dict[str, Any],
    *,
    title: str = "PySbagen Living Sessions Constellation",
) -> str:
    """Return a self-contained offline navigator with no remote assets or scripts."""

    safe_title = html.escape(title, quote=True)
    embedded = _json_for_html(graph)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>
:root {{ color-scheme: dark; --bg:#101216; --panel:#181c22; --panel2:#202630; --text:#edf2f7; --muted:#aab5c3; --line:#64748b; --accent:#8dd3ff; --good:#8ce99a; --warn:#ffd166; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif; }}
header {{ padding:18px 22px; border-bottom:1px solid #2b3440; background:#12161c; display:flex; gap:20px; align-items:flex-start; justify-content:space-between; }}
h1 {{ margin:0 0 4px; font-size:22px; }}
.meta {{ color:var(--muted); font-size:12px; word-break:break-all; }}
.controls {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; }}
.controls input,.controls select {{ background:var(--panel2); color:var(--text); border:1px solid #3a4655; border-radius:7px; padding:7px 9px; }}
main {{ display:grid; grid-template-columns:minmax(0,1fr) 360px; height:calc(100vh - 92px); }}
#viewport {{ position:relative; overflow:auto; background:radial-gradient(circle at 1px 1px,#28313c 1px,transparent 0); background-size:24px 24px; }}
#canvas {{ position:relative; min-width:100%; min-height:100%; }}
#edges {{ position:absolute; inset:0; overflow:visible; pointer-events:none; }}
.edge {{ fill:none; stroke:var(--line); stroke-width:2; opacity:.76; }}
.edge.return {{ stroke-dasharray:7 6; }}
.edge.wander {{ stroke:var(--warn); }}
.edge-label {{ fill:var(--muted); font-size:11px; text-anchor:middle; paint-order:stroke; stroke:var(--bg); stroke-width:5px; stroke-linejoin:round; }}
.node {{ position:absolute; width:{_NODE_WIDTH}px; min-height:{_NODE_HEIGHT}px; text-align:left; padding:11px 12px; border:1px solid #445164; border-radius:12px; background:linear-gradient(145deg,#202731,#171c23); color:var(--text); cursor:pointer; box-shadow:0 9px 24px #0007; }}
.node:hover,.node:focus-visible,.node.selected {{ border-color:var(--accent); outline:none; box-shadow:0 0 0 2px #8dd3ff33,0 12px 28px #0009; }}
.node.completed {{ border-left:5px solid var(--good); }}
.node.active {{ border-left:5px solid var(--accent); }}
.node.planned {{ border-left:5px solid #9aa6b2; }}
.node.wander {{ background:linear-gradient(145deg,#302b20,#1d1b17); }}
.node.warning::after {{ content:'!'; position:absolute; right:8px; top:7px; color:var(--warn); font-weight:800; }}
.node-title {{ font-weight:750; font-size:15px; margin-right:14px; }}
.node-sub {{ color:var(--muted); font-size:11px; margin-top:2px; }}
.chips {{ display:flex; flex-wrap:wrap; gap:4px; margin-top:8px; }}
.chip {{ border:1px solid #3e4a59; border-radius:999px; padding:1px 6px; color:#cbd5e1; font-size:10px; }}
#details {{ overflow:auto; border-left:1px solid #2b3440; background:var(--panel); padding:18px; }}
#details h2 {{ margin:0 0 3px; font-size:20px; }}
#details h3 {{ margin:20px 0 7px; font-size:13px; color:var(--accent); text-transform:uppercase; letter-spacing:.08em; }}
.kv {{ display:grid; grid-template-columns:120px minmax(0,1fr); gap:5px 10px; }}
.kv dt {{ color:var(--muted); }} .kv dd {{ margin:0; overflow-wrap:anywhere; }}
ul {{ padding-left:18px; }}
.note {{ color:var(--muted); }}
.warning-text {{ color:var(--warn); }}
.empty {{ color:var(--muted); padding:28px; }}
@media (max-width:900px) {{ main {{ grid-template-columns:1fr; grid-template-rows:minmax(55vh,1fr) auto; height:auto; }} #viewport {{ height:62vh; }} #details {{ border-left:0; border-top:1px solid #2b3440; max-height:none; }} }}
</style>
</head>
<body>
<header>
<div><h1>{safe_title}</h1><div class="meta" id="summary"></div></div>
<div class="controls">
<input id="search" type="search" placeholder="Search title, motif, echo, hash…" aria-label="Search sessions">
<select id="lineage" aria-label="Filter lineage"></select>
<select id="status" aria-label="Filter status"><option value="">All statuses</option><option>planned</option><option>active</option><option>completed</option></select>
<select id="mode" aria-label="Filter mode"><option value="">All modes</option><option>root</option><option>return</option><option>branch</option><option>contrast</option><option>wander</option></select>
</div>
</header>
<main>
<section id="viewport" aria-label="Session constellation"><div id="canvas"><svg id="edges" aria-hidden="true"></svg><div id="nodes"></div></div></section>
<aside id="details"><div class="empty">Select a session to inspect its recipe identity, changes, echoes, outcome, backend, and provenance.</div></aside>
</main>
<script id="constellation-data" type="application/json">{embedded}</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('constellation-data').textContent);
const canvas=document.getElementById('canvas'), nodesLayer=document.getElementById('nodes'), edgesSvg=document.getElementById('edges'), details=document.getElementById('details');
const search=document.getElementById('search'), lineage=document.getElementById('lineage'), statusFilter=document.getElementById('status'), modeFilter=document.getElementById('mode');
const nodeMap=new Map(data.nodes.map(n=>[n.session_id,n])); let selected=null;
canvas.style.width=data.canvas.width+'px'; canvas.style.height=data.canvas.height+'px'; edgesSvg.setAttribute('width',data.canvas.width); edgesSvg.setAttribute('height',data.canvas.height);
document.getElementById('summary').textContent=`${data.counts.sessions} sessions · ${data.counts.lineages} lineages · ${data.counts.echoes} echoes · snapshot ${data.snapshot_sha256}`;
lineage.innerHTML='<option value="">All lineages</option>'+data.lineages.map(x=>`<option value="${escapeAttr(x.lineage_id)}">${escapeText(x.lineage_id)} · ${x.session_count} sessions</option>`).join('');
function escapeText(v){return String(v).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function escapeAttr(v){return escapeText(v).replace(/"/g,'&quot;');}
function visible(n){const q=search.value.trim().toLowerCase(); const hay=JSON.stringify(n).toLowerCase(); return (!q||hay.includes(q))&&(!lineage.value||n.lineage_id===lineage.value)&&(!statusFilter.value||n.status===statusFilter.value)&&(!modeFilter.value||n.mode===modeFilter.value);}
function pathFor(a,b){const x1=a.layout.x+236, y1=a.layout.y+56, x2=b.layout.x, y2=b.layout.y+56, bend=Math.max(70,(x2-x1)*.48); return `M ${x1} ${y1} C ${x1+bend} ${y1}, ${x2-bend} ${y2}, ${x2} ${y2}`;}
function render(){nodesLayer.innerHTML=''; edgesSvg.innerHTML=''; const shown=new Set(data.nodes.filter(visible).map(n=>n.session_id));
for(const e of data.edges){if(!shown.has(e.source)||!shown.has(e.target))continue; const a=nodeMap.get(e.source),b=nodeMap.get(e.target); const p=document.createElementNS('http://www.w3.org/2000/svg','path'); p.setAttribute('d',pathFor(a,b)); p.setAttribute('class','edge '+e.mode); edgesSvg.appendChild(p); const t=document.createElementNS('http://www.w3.org/2000/svg','text'); t.setAttribute('x',(a.layout.x+236+b.layout.x)/2); t.setAttribute('y',(a.layout.y+b.layout.y)/2+48); t.setAttribute('class','edge-label'); t.textContent=e.short_label; edgesSvg.appendChild(t);}
for(const n of data.nodes){if(!shown.has(n.session_id))continue; const b=document.createElement('button'); b.type='button'; b.className=`node ${n.status} ${n.mode} ${n.warnings.length?'warning':''} ${selected===n.session_id?'selected':''}`; b.style.left=n.layout.x+'px'; b.style.top=n.layout.y+'px'; b.dataset.id=n.session_id; b.innerHTML=`<div class="node-title">${escapeText(n.title)}</div><div class="node-sub">g${n.generation} · ${escapeText(n.mode)} · ${escapeText(n.session_id.slice(0,8))}</div><div class="chips"><span class="chip">${escapeText(n.status)}</span><span class="chip">${n.echo_count} echo</span>${n.outcome?`<span class="chip">${n.outcome.rating}/5</span>`:''}<span class="chip">${escapeText(n.backend.policy)}</span></div>`; b.onclick=()=>selectNode(n.session_id,true); nodesLayer.appendChild(b);}
if(!shown.size)nodesLayer.innerHTML='<div class="empty">No sessions match the current filters.</div>';}
function row(label,value){return `<dt>${escapeText(label)}</dt><dd>${escapeText(value??'—')}</dd>`;}
function list(items,renderItem){return items.length?`<ul>${items.map(x=>`<li>${renderItem(x)}</li>`).join('')}</ul>`:'<div class="note">None recorded.</div>';}
function selectNode(id,scroll){selected=id; const n=nodeMap.get(id); render(); details.innerHTML=`<h2>${escapeText(n.title)}</h2><div class="note">${escapeText(n.motif.join(' / '))}</div><h3>Identity</h3><dl class="kv">${row('Session',n.session_id)}${row('Lineage',n.lineage_id)}${row('Parent',n.parent_session_id)}${row('Generation',n.generation)}${row('Mode',n.mode)}${row('Status',n.status)}${row('Created',n.created_at)}</dl><h3>Recipe</h3><dl class="kv">${row('SHA-256',n.recipe_sha256)}${row('Problem',n.recipe.problem)}${row('Sound world',n.recipe.sound_world)}${row('Intensity',n.recipe.intensity)}${row('Duration',n.recipe.duration_minutes+' min')}${row('Layers',n.recipe.layers.join(', ')||'default')}</dl><h3>Changes</h3>${list(n.mutations,m=>`${escapeText(m.key)}: <b>${escapeText(JSON.stringify(m.before))}</b> → <b>${escapeText(JSON.stringify(m.after))}</b><br><span class="note">${escapeText(m.reason)}</span>`)}<h3>Echoes and events</h3>${list(n.echoes,e=>`${escapeText(e.label)}${e.position_seconds==null?'':` @ ${e.position_seconds}s`}`)}<h3>Outcome</h3>${n.outcome?`<dl class="kv">${row('Rating',n.outcome.rating+'/5')}${row('Comfort',n.outcome.comfort)}${row('Would repeat',n.outcome.would_repeat?'yes':'no')}${row('Tags',n.outcome.tags.join(', '))}${row('Affect delta',n.outcome.affect_delta?JSON.stringify(n.outcome.affect_delta):'not recorded')}</dl>`:'<div class="note">Not recorded.</div>'}<h3>Backend and provenance</h3><dl class="kv">${row('Policy',n.backend.policy)}${row('Actual',n.backend.actual_backends.join(', ')||'not rendered')}${row('Output SHA',n.backend.latest_output_sha256)}${row('Interpretability',n.causal_interpretability)}</dl>${n.warnings.length?`<h3>Structural warnings</h3>${list(n.warnings,w=>`<span class="warning-text">${escapeText(w.detail)}</span>`)}`:''}<p class="note">${escapeText(data.interpretation_note)}</p>`; if(scroll)document.querySelector(`[data-id="${CSS.escape(id)}"]`)?.scrollIntoView({behavior:'smooth',block:'center',inline:'center'});}
for(const el of [search,lineage,statusFilter,modeFilter])el.addEventListener('input',render);
render(); if(data.scope.focus_session_id)selectNode(data.scope.focus_session_id,true); else if(data.nodes.length)selectNode(data.nodes[0].session_id,false);
</script>
</body>
</html>
"""


def write_constellation_html(
    archive: LivingSessionArchive,
    path: str | Path,
    *,
    lineage_id: str | None = None,
    focus_session_id: str | None = None,
    title: str = "PySbagen Living Sessions Constellation",
) -> tuple[Path, dict[str, Any]]:
    """Write one offline HTML snapshot atomically and return its graph receipt."""

    graph = build_constellation(archive, lineage_id=lineage_id, focus_session_id=focus_session_id)
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(render_constellation_html(graph, title=title), encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve(), graph


def _session_sort_key(item: StoredSession) -> tuple[Any, ...]:
    return (item.plan.lineage_id, item.plan.generation, item.plan.created_at, item.plan.session_id)


def _layout(sessions: list[StoredSession]) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    by_lineage: dict[str, list[StoredSession]] = {}
    for item in sessions:
        by_lineage.setdefault(item.plan.lineage_id, []).append(item)
    positions: dict[str, dict[str, int]] = {}
    top = _CANVAS_MARGIN
    max_generation = 0
    for lineage_id in sorted(by_lineage):
        items = sorted(by_lineage[lineage_id], key=_session_sort_key)
        by_generation: dict[int, list[StoredSession]] = {}
        for item in items:
            by_generation.setdefault(item.plan.generation, []).append(item)
            max_generation = max(max_generation, item.plan.generation)
        rows = max((len(group) for group in by_generation.values()), default=1)
        block_height = max(_NODE_HEIGHT + 64, rows * _Y_GAP)
        for generation, group in sorted(by_generation.items()):
            ordered = sorted(group, key=lambda item: (item.plan.created_at, item.plan.session_id))
            offset = (rows - len(ordered)) * _Y_GAP // 2
            for index, item in enumerate(ordered):
                positions[item.plan.session_id] = {"x": _CANVAS_MARGIN + generation * _X_GAP, "y": top + offset + index * _Y_GAP}
        top += block_height + _LINEAGE_GAP
    width = max(760, _CANVAS_MARGIN * 2 + (max_generation + 1) * _X_GAP + _NODE_WIDTH)
    height = max(460, top + _CANVAS_MARGIN - _LINEAGE_GAP)
    return positions, {"width": width, "height": height}


def _node_payload(item: StoredSession, layout: dict[str, int], warnings: list[ConstellationWarning]) -> dict[str, Any]:
    plan = item.plan
    request = plan.recipe_manifest.get("request") or {}
    layers = request.get("layers") or {}
    enabled_layers = sorted(key for key, enabled in layers.items() if enabled)
    echoes = [
        {"event_id": event.event_id, "kind": event.kind, "label": event.label, "position_seconds": event.position_seconds, "created_at": event.created_at, "payload": event.payload}
        for event in item.events
        if event.kind in {"echo", "shift", "insight", "discomfort"}
    ]
    render_events = [event for event in item.events if event.kind == "render"]
    actual_backends = sorted({str(event.payload.get("actual_backend")) for event in render_events if event.payload.get("actual_backend")})
    latest_render = max(render_events, key=lambda event: event.created_at) if render_events else None
    node_warnings = [asdict(warning) for warning in warnings if warning.session_id == plan.session_id]
    return {
        "session_id": plan.session_id,
        "lineage_id": plan.lineage_id,
        "parent_session_id": plan.parent_session_id,
        "generation": plan.generation,
        "mode": plan.mode,
        "title": plan.title,
        "motif": list(plan.motif),
        "memory_phrase": plan.memory_phrase,
        "created_at": plan.created_at,
        "status": item.status,
        "experimental": plan.experimental,
        "rationale": plan.rationale,
        "recipe_sha256": plan.recipe_sha256,
        "recipe": {
            "problem": request.get("problem"),
            "sound_world": request.get("sound_world"),
            "intensity": request.get("intensity"),
            "duration_minutes": request.get("duration_minutes"),
            "seed": request.get("seed"),
            "layers": enabled_layers,
            "user_audio_bound": bool(request.get("user_audio")),
        },
        "mutations": [asdict(mutation) for mutation in plan.mutations],
        "echo_count": sum(event.kind == "echo" for event in item.events),
        "echoes": echoes,
        "event_count": len(item.events),
        "outcome": _outcome_payload(item),
        "backend": {
            "policy": plan.backend_policy,
            "actual_backends": actual_backends,
            "latest_output_sha256": latest_render.payload.get("output_sha256") if latest_render else None,
            "latest_output_path": latest_render.payload.get("output_path") if latest_render else None,
            "latest_backend_reason": latest_render.payload.get("backend_reason") if latest_render else None,
        },
        "causal_interpretability": _causal_interpretability(plan.mode, len(plan.mutations)),
        "warnings": node_warnings,
        "layout": layout,
    }


def _outcome_payload(item: StoredSession) -> dict[str, Any] | None:
    outcome = item.outcome
    if outcome is None:
        return None
    before = item.plan.pre_affect
    after = outcome.post_affect
    affect_delta = None
    if before is not None and after is not None:
        affect_delta = {"valence": after.valence - before.valence, "arousal": after.arousal - before.arousal, "agency": after.agency - before.agency}
    return {"completed_at": outcome.completed_at, "rating": outcome.rating, "would_repeat": outcome.would_repeat, "comfort": outcome.comfort, "note": outcome.note, "tags": list(outcome.tags), "affect_delta": affect_delta}


def _edge_payload(child: StoredSession, parent: StoredSession) -> dict[str, Any]:
    mutations = [asdict(mutation) for mutation in child.plan.mutations]
    return {
        "edge_id": f"{parent.plan.session_id}->{child.plan.session_id}",
        "source": parent.plan.session_id,
        "target": child.plan.session_id,
        "mode": child.plan.mode,
        "short_label": _edge_label(child),
        "mutations": mutations,
        "change_count": len(mutations),
        "experimental": child.plan.experimental,
        "recipe_identity_preserved": parent.plan.recipe_sha256 == child.plan.recipe_sha256,
        "causal_interpretability": _causal_interpretability(child.plan.mode, len(mutations)),
    }


def _edge_label(child: StoredSession) -> str:
    if child.plan.mode == "return":
        return "exact return"
    keys = [mutation.key.removeprefix("request.") for mutation in child.plan.mutations]
    return child.plan.mode if not keys else f"{child.plan.mode} · {' + '.join(keys)}"


def _lineage_payloads(sessions: list[StoredSession]) -> list[dict[str, Any]]:
    by_lineage: dict[str, list[StoredSession]] = {}
    for item in sessions:
        by_lineage.setdefault(item.plan.lineage_id, []).append(item)
    payloads: list[dict[str, Any]] = []
    for lineage_id, items in sorted(by_lineage.items()):
        ordered = sorted(items, key=_session_sort_key)
        roots = [item.plan.session_id for item in ordered if item.plan.parent_session_id is None]
        latest = max(ordered, key=lambda item: (item.plan.created_at, item.plan.session_id))
        payloads.append({
            "lineage_id": lineage_id,
            "session_count": len(ordered),
            "completed_count": sum(item.outcome is not None for item in ordered),
            "echo_count": sum(event.kind == "echo" for item in ordered for event in item.events),
            "root_session_ids": roots,
            "latest_session_id": latest.plan.session_id,
            "max_generation": max(item.plan.generation for item in ordered),
            "titles": [item.plan.title for item in ordered],
        })
    return payloads


def _structural_warnings(sessions: list[StoredSession], by_id: dict[str, StoredSession]) -> list[ConstellationWarning]:
    warnings: list[ConstellationWarning] = []
    for item in sessions:
        plan = item.plan
        parent_id = plan.parent_session_id
        if parent_id is None:
            if plan.generation != 0:
                warnings.append(ConstellationWarning("root-generation-mismatch", plan.session_id, f"Session has no parent but reports generation {plan.generation}."))
            continue
        parent = by_id.get(parent_id)
        if parent is None:
            warnings.append(ConstellationWarning("parent-not-in-snapshot", plan.session_id, f"Parent {parent_id} is absent from the selected constellation snapshot."))
            continue
        if parent.plan.lineage_id != plan.lineage_id:
            warnings.append(ConstellationWarning("parent-lineage-mismatch", plan.session_id, "Parent and child report different lineage identities."))
        if plan.generation != parent.plan.generation + 1:
            warnings.append(ConstellationWarning("generation-gap", plan.session_id, f"Child generation {plan.generation} does not directly follow parent generation {parent.plan.generation}."))
        if plan.mode == "return" and plan.recipe_sha256 != parent.plan.recipe_sha256:
            warnings.append(ConstellationWarning("return-recipe-mismatch", plan.session_id, "A return edge does not preserve the parent recipe SHA-256."))
    return sorted(warnings, key=lambda item: (item.session_id, item.code, item.detail))


def _causal_interpretability(mode: str, change_count: int) -> str:
    if mode == "return" and change_count == 0:
        return "exact-repeat"
    if mode in {"branch", "contrast"} and change_count == 1:
        return "high"
    if mode == "wander" and 1 <= change_count <= 2:
        return "bounded-exploration"
    if mode == "root":
        return "baseline"
    return "uncertain"


def _display_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_for_html(payload: Any) -> str:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
