# PySbagen Living Sessions — Constellation Guide

**Status:** Wave 2 product surface  
**Command:** `sbgpy-session constellation`  
**Network requirement:** none  
**Archive mutation:** none

## Why this exists

A lineage should not become a pile of session IDs and WAV filenames. The Constellation turns the local Living Sessions archive into a navigable map of:

- where a session came from;
- what changed;
- what stayed identical;
- which moments were remembered;
- what outcome was recorded;
- which backend actually rendered it;
- which recipe and output hashes prove its identity.

This is not a decorative diagram. Selecting a node exposes the evidence behind it.

## Open the full local constellation

```bash
sbgpy-session constellation --html
```

This writes `living-session-constellation.html`. The file is self-contained. It loads no CDN, analytics script, font, image, remote stylesheet, or network API.

Choose another destination:

```bash
sbgpy-session constellation --html ./atlas/tonight.html
```

## Restrict the view

One lineage:

```bash
sbgpy-session constellation --lineage LINEAGE_ID --html lineage.html
```

Open with one session selected and centered:

```bash
sbgpy-session constellation --focus SESSION_ID --html focused.html
```

A focus session must be inside the selected snapshot. Missing lineages and out-of-scope focus IDs fail closed.

## Terminal and JSON views

```bash
sbgpy-session constellation
sbgpy-session constellation --json
sbgpy-session constellation --html map.html --json
```

The export receipt includes the exact graph snapshot SHA-256, HTML path, and HTML SHA-256. Generating a constellation never appends an archive event or modifies a session. It is a read-only snapshot.

## What a node shows

Each node contains or exposes:

- title and motif;
- session, lineage, parent, and generation IDs;
- mode and status;
- exact recipe SHA-256;
- stated problem, sound world, intensity, duration, seed, and layers;
- disclosed mutations;
- echoes, shifts, insights, and discomfort markers;
- optional rating, comfort, repeat intent, tags, and affect delta;
- backend policy;
- actual renderer identities found in render receipts;
- latest output SHA-256;
- causal-interpretability label;
- structural warnings.

## What an edge means

- `exact return` — parent and child preserve recipe identity;
- `branch` — one disclosed dimension changed;
- `contrast` — one audible high-salience dimension changed;
- `wander` — one or two disclosed changes, marked bounded exploration.

The edge label names the changed recipe dimensions. The details panel preserves before/after values and the recorded reason.

## Search and filters

The offline navigator supports full-text search across titles, motifs, echoes, IDs, and hashes; lineage, status, and mode filters; click and keyboard selection; and scrolling to a focused session.

## Structural truth

The map does not invent missing ancestry. It reports:

- a parent absent from the selected snapshot;
- parent/child lineage mismatch;
- generation gaps;
- a return whose recipe hash changed;
- a parentless session claiming a non-zero generation.

These warnings describe archive structure. They do not rewrite or repair history.

## HTE / InvisiSynth design use

The design applies the project-owner supplied HTE-Newest reasoning corpus:

- **Memory:** show episodes and continuity, but treat patterns as guidance rather than destiny;
- **Learning:** keep prediction, change, outcome, and mismatch connected;
- **Affect:** expose optional emotional tags beside the episode they belong to;
- **Dependency Grapher:** make ancestry and dependency direction legible;
- **Parallel Oracle / Synthesis:** keep technical provenance and human memory visible at the same time;
- **InvisiSynth:** surface absences and broken relationships rather than drawing a falsely complete graph.

No HTE runtime code or private configuration is copied into the product.

## Product boundary

Constellation is a PySbagen session-intelligence surface. It adds no synthesis engine, curve language, mix effect, native editor, or SBaGenX DSP duplicate.

SBaGenX remains the optional advanced native SBG/SBGF engine. PySbagen maps the continuing human and provenance history above whichever qualified renderer was actually used.

The possible Cycloside cross-project path remains permission-gated and is not part of this train.

## Interpretation boundary

The navigator shows personal descriptive history. A high rating, affect change, or repeated preference is not proof of medical efficacy or causation.
