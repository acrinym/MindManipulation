"""Read-only Living Sessions constellation snapshots and offline navigation."""

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

NODE_W, NODE_H, X_GAP, Y_GAP, MARGIN = 232, 108, 292, 138, 64


@dataclass(frozen=True)
class ConstellationWarning:
    code: str
    session_id: str
    detail: str


def build_constellation(
    archive: LivingSessionArchive,
    *,
    lineage_id: str | None = None,
    focus_session_id: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic graph without modifying the archive."""
    sessions = archive.list_sessions()
    if lineage_id is not None:
        sessions = [s for s in sessions if s.plan.lineage_id == lineage_id]
        if not sessions:
            raise KeyError(f"Living-session lineage not found: {lineage_id}")
    by_id = {s.plan.session_id: s for s in sessions}
    if focus_session_id is not None and focus_session_id not in by_id:
        raise KeyError(f"Living session is not present in this constellation: {focus_session_id}")

    warnings = _warnings(sessions, by_id)
    layout, canvas = _layout(sessions)
    ordered = sorted(sessions, key=_sort_key)
    nodes = [_node(s, layout[s.plan.session_id], warnings) for s in ordered]
    edges = [
        _edge(s, by_id[s.plan.parent_session_id])
        for s in ordered
        if s.plan.parent_session_id in by_id
    ]
    lineages = _lineages(sessions)
    snapshot_payload = {
        "nodes": nodes,
        "edges": edges,
        "warnings": [asdict(w) for w in warnings],
        "lineages": lineages,
    }
    snapshot = hashlib.sha256(_canonical(snapshot_payload).encode()).hexdigest()
    return {
        "schema": "pysbagen.living-session-constellation.v1",
        "snapshot_sha256": snapshot,
        "scope": {
            "archive_root": str(archive.root.resolve()),
            "lineage_id": lineage_id,
            "focus_session_id": focus_session_id,
        },
        "counts": {
            "sessions": len(nodes),
            "edges": len(edges),
            "lineages": len(lineages),
            "completed": sum(n["status"] == "completed" for n in nodes),
            "echoes": sum(n["echo_count"] for n in nodes),
            "warnings": len(warnings),
        },
        "canvas": canvas,
        "lineages": lineages,
        "nodes": nodes,
        "edges": edges,
        "warnings": [asdict(w) for w in warnings],
        "interpretation_note": (
            "The constellation is a local navigation and provenance surface. "
            "Outcome patterns are descriptive personal records, not medical-efficacy conclusions."
        ),
    }


def constellation_to_text(graph: dict[str, Any]) -> str:
    counts = graph["counts"]
    lines = [
        "PySbagen Living Sessions Constellation",
        f"Snapshot: {graph['snapshot_sha256']}",
        f"Sessions: {counts['sessions']} · lineages: {counts['lineages']} · connections: {counts['edges']} · echoes: {counts['echoes']}",
    ]
    if counts["warnings"]:
        lines.append(f"Structural warnings: {counts['warnings']}")
    for lineage in graph["lineages"]:
        lines.extend(["", f"Lineage {lineage['lineage_id']} · {lineage['session_count']} session(s) · latest {lineage['latest_session_id']}"])
        items = [n for n in graph["nodes"] if n["lineage_id"] == lineage["lineage_id"]]
        for n in sorted(items, key=lambda x: (x["generation"], x["created_at"], x["session_id"])):
            parent = "ROOT" if n["parent_session_id"] is None else f"↳ {n['parent_session_id'][:8]}"
            rating = f" · {n['outcome']['rating']}/5" if n["outcome"] else ""
            echoes = f" · {n['echo_count']} echo(es)" if n["echo_count"] else ""
            lines.append(f"  g{n['generation']} {parent:10} {n['mode']:8} {n['memory_phrase']} [{n['status']}]{rating}{echoes}")
            for mutation in n["mutations"]:
                lines.append(f"      {mutation['key']}: {_display(mutation['before'])} → {_display(mutation['after'])}")
    lines.extend(["", graph["interpretation_note"]])
    return "\n".join(lines)


def render_constellation_html(
    graph: dict[str, Any],
    *,
    title: str = "PySbagen Living Sessions Constellation",
) -> str:
    """Render a self-contained HTML navigator with no remote resources."""
    data = _html_json(graph)
    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{color-scheme:dark;--bg:#101318;--panel:#1a2028;--text:#eef3f8;--muted:#aab6c4;--line:#637287;--accent:#8dd3ff;--good:#8ce99a;--warn:#ffd166}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}header{padding:14px 18px;border-bottom:1px solid #303947;display:flex;gap:16px;justify-content:space-between;align-items:flex-start}h1{font-size:20px;margin:0}.meta,.note{color:var(--muted)}.controls{display:flex;gap:8px;flex-wrap:wrap}.controls input,.controls select{background:#202733;color:var(--text);border:1px solid #435064;border-radius:7px;padding:7px}main{display:grid;grid-template-columns:minmax(0,1fr) 350px;height:calc(100vh - 76px)}#view{overflow:auto;position:relative;background:radial-gradient(circle at 1px 1px,#29323e 1px,transparent 0);background-size:24px 24px}#canvas{position:relative}svg{position:absolute;inset:0;pointer-events:none}.edge{fill:none;stroke:var(--line);stroke-width:2}.edge.return{stroke-dasharray:7 6}.edge.wander{stroke:var(--warn)}.edge-label{fill:var(--muted);font-size:11px;text-anchor:middle;paint-order:stroke;stroke:var(--bg);stroke-width:5px}.node{position:absolute;width:232px;min-height:108px;text-align:left;padding:10px;border:1px solid #47556a;border-left:5px solid #95a2b2;border-radius:11px;background:#1b222c;color:var(--text);cursor:pointer;box-shadow:0 8px 22px #0007}.node.active{border-left-color:var(--accent)}.node.completed{border-left-color:var(--good)}.node.wander{background:#29251d}.node.warning:after{content:'!';position:absolute;right:8px;top:6px;color:var(--warn);font-weight:800}.node:hover,.node:focus-visible,.node.selected{outline:none;border-color:var(--accent);box-shadow:0 0 0 2px #8dd3ff44}.title{font-weight:750;font-size:15px}.sub{font-size:11px;color:var(--muted)}.chips{display:flex;gap:4px;flex-wrap:wrap;margin-top:8px}.chip{border:1px solid #435064;border-radius:999px;padding:1px 6px;font-size:10px}#details{overflow:auto;border-left:1px solid #303947;background:var(--panel);padding:16px}#details h2{margin:0}#details h3{font-size:12px;color:var(--accent);letter-spacing:.08em;text-transform:uppercase;margin:18px 0 6px}.kv{display:grid;grid-template-columns:112px minmax(0,1fr);gap:4px 8px}.kv dt{color:var(--muted)}.kv dd{margin:0;overflow-wrap:anywhere}.empty{padding:24px;color:var(--muted)}.warning-text{color:var(--warn)}@media(max-width:850px){main{grid-template-columns:1fr;grid-template-rows:60vh auto;height:auto}#details{border-left:0;border-top:1px solid #303947}}
</style></head><body><header><div><h1>__TITLE__</h1><div id="summary" class="meta"></div></div><div class="controls"><input id="search" type="search" placeholder="Search title, motif, echo, hash…"><select id="lineage" aria-label="Filter lineage"></select><select id="status"><option value="">All statuses</option><option>planned</option><option>active</option><option>completed</option></select><select id="mode"><option value="">All modes</option><option>root</option><option>return</option><option>branch</option><option>contrast</option><option>wander</option></select></div></header>
<main><section id="view"><div id="canvas"><svg id="edges"></svg><div id="nodes"></div></div></section><aside id="details"><div class="empty">Select a session to inspect changes, memory, outcome, backend, and provenance.</div></aside></main>
<script id="constellation-data" type="application/json">__DATA__</script><script>
'use strict';const d=JSON.parse(document.getElementById('constellation-data').textContent),map=new Map(d.nodes.map(n=>[n.session_id,n]));let selected=null;const canvas=document.getElementById('canvas'),svg=document.getElementById('edges'),layer=document.getElementById('nodes'),details=document.getElementById('details'),search=document.getElementById('search'),lineage=document.getElementById('lineage'),status=document.getElementById('status'),mode=document.getElementById('mode');canvas.style.width=d.canvas.width+'px';canvas.style.height=d.canvas.height+'px';svg.setAttribute('width',d.canvas.width);svg.setAttribute('height',d.canvas.height);document.getElementById('summary').textContent=`${d.counts.sessions} sessions · ${d.counts.lineages} lineages · ${d.counts.echoes} echoes · ${d.snapshot_sha256}`;
const esc=v=>String(v??'—').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));lineage.innerHTML='<option value="">All lineages</option>'+d.lineages.map(x=>`<option value="${esc(x.lineage_id)}">${esc(x.lineage_id)} · ${x.session_count}</option>`).join('');const shown=n=>{const q=search.value.trim().toLowerCase();return(!q||JSON.stringify(n).toLowerCase().includes(q))&&(!lineage.value||n.lineage_id===lineage.value)&&(!status.value||n.status===status.value)&&(!mode.value||n.mode===mode.value)};const path=(a,b)=>{const x1=a.layout.x+232,y1=a.layout.y+54,x2=b.layout.x,y2=b.layout.y+54,k=Math.max(70,(x2-x1)*.48);return`M ${x1} ${y1} C ${x1+k} ${y1}, ${x2-k} ${y2}, ${x2} ${y2}`};
function render(){layer.innerHTML='';svg.innerHTML='';const ids=new Set(d.nodes.filter(shown).map(n=>n.session_id));for(const e of d.edges){if(!ids.has(e.source)||!ids.has(e.target))continue;const a=map.get(e.source),b=map.get(e.target),p=document.createElementNS('http://www.w3.org/2000/svg','path'),t=document.createElementNS('http://www.w3.org/2000/svg','text');p.setAttribute('d',path(a,b));p.setAttribute('class','edge '+e.mode);t.setAttribute('x',(a.layout.x+232+b.layout.x)/2);t.setAttribute('y',(a.layout.y+b.layout.y)/2+46);t.setAttribute('class','edge-label');t.textContent=e.short_label;svg.append(p,t)}for(const n of d.nodes){if(!ids.has(n.session_id))continue;const b=document.createElement('button');b.className=`node ${n.status} ${n.mode} ${n.warnings.length?'warning':''} ${selected===n.session_id?'selected':''}`;b.style.left=n.layout.x+'px';b.style.top=n.layout.y+'px';b.dataset.id=n.session_id;b.innerHTML=`<div class="title">${esc(n.title)}</div><div class="sub">g${n.generation} · ${esc(n.mode)} · ${esc(n.session_id.slice(0,8))}</div><div class="chips"><span class="chip">${esc(n.status)}</span><span class="chip">${n.echo_count} echo</span>${n.outcome?`<span class="chip">${n.outcome.rating}/5</span>`:''}<span class="chip">${esc(n.backend.policy)}</span></div>`;b.onclick=()=>select(n.session_id,true);layer.appendChild(b)}if(!ids.size)layer.innerHTML='<div class="empty">No sessions match the filters.</div>'}
const row=(k,v)=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`,list=(xs,f)=>xs.length?`<ul>${xs.map(x=>`<li>${f(x)}</li>`).join('')}</ul>`:'<div class="note">None recorded.</div>';function select(id,scroll){selected=id;const n=map.get(id);render();details.innerHTML=`<h2>${esc(n.title)}</h2><div class="note">${esc(n.motif.join(' / '))}</div><h3>Identity</h3><dl class="kv">${row('Session',n.session_id)}${row('Lineage',n.lineage_id)}${row('Parent',n.parent_session_id)}${row('Generation',n.generation)}${row('Mode',n.mode)}${row('Status',n.status)}</dl><h3>Recipe</h3><dl class="kv">${row('SHA-256',n.recipe_sha256)}${row('Problem',n.recipe.problem)}${row('Sound world',n.recipe.sound_world)}${row('Intensity',n.recipe.intensity)}${row('Duration',n.recipe.duration_minutes+' min')}${row('Layers',n.recipe.layers.join(', ')||'default')}</dl><h3>Changes</h3>${list(n.mutations,m=>`${esc(m.key)}: <b>${esc(JSON.stringify(m.before))}</b> → <b>${esc(JSON.stringify(m.after))}</b><br><span class="note">${esc(m.reason)}</span>`)}<h3>Echoes and events</h3>${list(n.echoes,e=>`${esc(e.label)}${e.position_seconds==null?'':` @ ${e.position_seconds}s`}`)}<h3>Outcome</h3>${n.outcome?`<dl class="kv">${row('Rating',n.outcome.rating+'/5')}${row('Comfort',n.outcome.comfort)}${row('Would repeat',n.outcome.would_repeat?'yes':'no')}${row('Tags',n.outcome.tags.join(', '))}${row('Affect delta',n.outcome.affect_delta?JSON.stringify(n.outcome.affect_delta):'not recorded')}</dl>`:'<div class="note">Not recorded.</div>'}<h3>Backend and provenance</h3><dl class="kv">${row('Policy',n.backend.policy)}${row('Actual',n.backend.actual_backends.join(', ')||'not rendered')}${row('Output SHA',n.backend.latest_output_sha256)}${row('Interpretability',n.causal_interpretability)}</dl>${n.warnings.length?`<h3>Structural warnings</h3>${list(n.warnings,w=>`<span class="warning-text">${esc(w.detail)}</span>`)}`:''}<p class="note">${esc(d.interpretation_note)}</p>`;if(scroll)[...document.querySelectorAll('.node')].find(x=>x.dataset.id===id)?.scrollIntoView({behavior:'smooth',block:'center',inline:'center'})}for(const x of[search,lineage,status,mode])x.addEventListener('input',render);render();if(d.scope.focus_session_id)select(d.scope.focus_session_id,true);else if(d.nodes.length)select(d.nodes[0].session_id,false);
</script></body></html>"""
    return template.replace("__TITLE__", html.escape(title, quote=True)).replace("__DATA__", data)


def write_constellation_html(
    archive: LivingSessionArchive,
    path: str | Path,
    *,
    lineage_id: str | None = None,
    focus_session_id: str | None = None,
    title: str = "PySbagen Living Sessions Constellation",
) -> tuple[Path, dict[str, Any]]:
    graph = build_constellation(archive, lineage_id=lineage_id, focus_session_id=focus_session_id)
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(render_constellation_html(graph, title=title), encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve(), graph


def _sort_key(s: StoredSession) -> tuple[Any, ...]:
    return s.plan.lineage_id, s.plan.generation, s.plan.created_at, s.plan.session_id


def _layout(sessions: list[StoredSession]) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    grouped: dict[str, list[StoredSession]] = {}
    for s in sessions:
        grouped.setdefault(s.plan.lineage_id, []).append(s)
    positions: dict[str, dict[str, int]] = {}
    top, max_generation = MARGIN, 0
    for lineage_id in sorted(grouped):
        generations: dict[int, list[StoredSession]] = {}
        for s in grouped[lineage_id]:
            generations.setdefault(s.plan.generation, []).append(s)
            max_generation = max(max_generation, s.plan.generation)
        rows = max((len(v) for v in generations.values()), default=1)
        for generation, items in sorted(generations.items()):
            ordered = sorted(items, key=lambda s: (s.plan.created_at, s.plan.session_id))
            offset = (rows - len(ordered)) * Y_GAP // 2
            for index, s in enumerate(ordered):
                positions[s.plan.session_id] = {"x": MARGIN + generation * X_GAP, "y": top + offset + index * Y_GAP}
        top += max(NODE_H + 58, rows * Y_GAP) + 82
    width = max(760, MARGIN * 2 + (max_generation + 1) * X_GAP + NODE_W)
    return positions, {"width": width, "height": max(450, top + MARGIN - 82)}


def _node(s: StoredSession, layout: dict[str, int], warnings: list[ConstellationWarning]) -> dict[str, Any]:
    plan, request = s.plan, s.plan.recipe_manifest.get("request") or {}
    layer_data = request.get("layers") or {}
    echoes = [
        {"event_id": e.event_id, "kind": e.kind, "label": e.label, "position_seconds": e.position_seconds, "created_at": e.created_at, "payload": e.payload}
        for e in s.events if e.kind in {"echo", "shift", "insight", "discomfort"}
    ]
    renders = [e for e in s.events if e.kind == "render"]
    latest = max(renders, key=lambda e: e.created_at) if renders else None
    return {
        "session_id": plan.session_id, "lineage_id": plan.lineage_id,
        "parent_session_id": plan.parent_session_id, "generation": plan.generation,
        "mode": plan.mode, "title": plan.title, "motif": list(plan.motif),
        "memory_phrase": plan.memory_phrase, "created_at": plan.created_at,
        "status": s.status, "experimental": plan.experimental, "rationale": plan.rationale,
        "recipe_sha256": plan.recipe_sha256,
        "recipe": {
            "problem": request.get("problem"), "sound_world": request.get("sound_world"),
            "intensity": request.get("intensity"), "duration_minutes": request.get("duration_minutes"),
            "seed": request.get("seed"), "layers": sorted(k for k, v in layer_data.items() if v),
            "user_audio_bound": bool(request.get("user_audio")),
        },
        "mutations": [asdict(m) for m in plan.mutations],
        "echo_count": sum(e.kind == "echo" for e in s.events), "echoes": echoes,
        "event_count": len(s.events), "outcome": _outcome(s),
        "backend": {
            "policy": plan.backend_policy,
            "actual_backends": sorted({str(e.payload.get("actual_backend")) for e in renders if e.payload.get("actual_backend")}),
            "latest_output_sha256": latest.payload.get("output_sha256") if latest else None,
            "latest_output_path": latest.payload.get("output_path") if latest else None,
            "latest_backend_reason": latest.payload.get("backend_reason") if latest else None,
        },
        "causal_interpretability": _interpretability(plan.mode, len(plan.mutations)),
        "warnings": [asdict(w) for w in warnings if w.session_id == plan.session_id],
        "layout": layout,
    }


def _outcome(s: StoredSession) -> dict[str, Any] | None:
    if s.outcome is None:
        return None
    before, after = s.plan.pre_affect, s.outcome.post_affect
    delta = None if before is None or after is None else {
        "valence": after.valence - before.valence,
        "arousal": after.arousal - before.arousal,
        "agency": after.agency - before.agency,
    }
    return {
        "completed_at": s.outcome.completed_at, "rating": s.outcome.rating,
        "would_repeat": s.outcome.would_repeat, "comfort": s.outcome.comfort,
        "note": s.outcome.note, "tags": list(s.outcome.tags), "affect_delta": delta,
    }


def _edge(child: StoredSession, parent: StoredSession) -> dict[str, Any]:
    mutations = [asdict(m) for m in child.plan.mutations]
    keys = [m.key.removeprefix("request.") for m in child.plan.mutations]
    label = "exact return" if child.plan.mode == "return" else child.plan.mode if not keys else f"{child.plan.mode} · {' + '.join(keys)}"
    return {
        "edge_id": f"{parent.plan.session_id}->{child.plan.session_id}",
        "source": parent.plan.session_id, "target": child.plan.session_id,
        "mode": child.plan.mode, "short_label": label, "mutations": mutations,
        "change_count": len(mutations), "experimental": child.plan.experimental,
        "recipe_identity_preserved": parent.plan.recipe_sha256 == child.plan.recipe_sha256,
        "causal_interpretability": _interpretability(child.plan.mode, len(mutations)),
    }


def _lineages(sessions: list[StoredSession]) -> list[dict[str, Any]]:
    grouped: dict[str, list[StoredSession]] = {}
    for s in sessions:
        grouped.setdefault(s.plan.lineage_id, []).append(s)
    result = []
    for lineage_id, items in sorted(grouped.items()):
        ordered = sorted(items, key=_sort_key)
        latest = max(ordered, key=lambda s: (s.plan.created_at, s.plan.session_id))
        result.append({
            "lineage_id": lineage_id, "session_count": len(ordered),
            "completed_count": sum(s.outcome is not None for s in ordered),
            "echo_count": sum(e.kind == "echo" for s in ordered for e in s.events),
            "root_session_ids": [s.plan.session_id for s in ordered if s.plan.parent_session_id is None],
            "latest_session_id": latest.plan.session_id,
            "max_generation": max(s.plan.generation for s in ordered),
            "titles": [s.plan.title for s in ordered],
        })
    return result


def _warnings(sessions: list[StoredSession], by_id: dict[str, StoredSession]) -> list[ConstellationWarning]:
    result: list[ConstellationWarning] = []
    for s in sessions:
        p, parent_id = s.plan, s.plan.parent_session_id
        if parent_id is None:
            if p.generation != 0:
                result.append(ConstellationWarning("root-generation-mismatch", p.session_id, f"Session has no parent but reports generation {p.generation}."))
            continue
        parent = by_id.get(parent_id)
        if parent is None:
            result.append(ConstellationWarning("parent-not-in-snapshot", p.session_id, f"Parent {parent_id} is absent from the selected constellation snapshot."))
            continue
        if parent.plan.lineage_id != p.lineage_id:
            result.append(ConstellationWarning("parent-lineage-mismatch", p.session_id, "Parent and child report different lineage identities."))
        if p.generation != parent.plan.generation + 1:
            result.append(ConstellationWarning("generation-gap", p.session_id, f"Child generation {p.generation} does not directly follow parent generation {parent.plan.generation}."))
        if p.mode == "return" and p.recipe_sha256 != parent.plan.recipe_sha256:
            result.append(ConstellationWarning("return-recipe-mismatch", p.session_id, "A return edge does not preserve the parent recipe SHA-256."))
    return sorted(result, key=lambda w: (w.session_id, w.code, w.detail))


def _interpretability(mode: str, changes: int) -> str:
    if mode == "root": return "baseline"
    if mode == "return" and changes == 0: return "exact-repeat"
    if mode in {"branch", "contrast"} and changes == 1: return "high"
    if mode == "wander" and 1 <= changes <= 2: return "bounded-exploration"
    return "uncertain"


def _display(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else str(value)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _html_json(value: Any) -> str:
    return _canonical(value).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
