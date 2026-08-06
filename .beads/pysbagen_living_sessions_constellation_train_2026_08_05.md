# PySbagen Living Sessions — Constellation Train

**Date:** August 5, 2026  
**Branch:** `agent/living-sessions-constellation-20260805`  
**Stack base:** `agent/living-sessions-train-20260731` / PR #12  
**Status:** implementation in qualification

## Product intent

Turn Living Sessions ancestry into an offline navigation surface that helps someone return, compare, understand, and continue—not a graph exported once and forgotten.

## Hard boundaries

- Do not duplicate SBaGenX DSP, authoring, plotting, or curve features.
- Do not mutate the archive while viewing it.
- Do not hide recipe hashes or backend receipts behind human-facing names.
- Do not infer missing ancestry.
- Do not introduce remote assets, analytics, cloud accounts, badges, streaks, or leaderboards.
- Do not build or explore the permission-gated Cycloside path.

## HTE / InvisiSynth synthesis

The train uses:

- Memory Engine: episodes + continuity + uncertainty;
- Learning Engine: change → outcome → mismatch visibility;
- Affect Engine: emotional tags stay attached to their episode;
- Dependency Grapher: directed ancestry and critical relationships;
- Parallel Oracle and Synthesis: human memory and technical provenance coexist;
- InvisiSynth: missing parents, lineage breaks, and suspicious absences remain visible.

## Beads

### LIV-011A — Deterministic constellation graph

- nodes for stored sessions;
- directed parent/child edges;
- deterministic layout;
- snapshot SHA-256;
- lineage filtering and focus validation;
- read-only operation.

### LIV-011B — Evidence-rich node and edge model

- recipe summaries and full recipe hashes;
- before/after mutation values;
- causal-interpretability labels;
- echoes and event anchors;
- outcomes and optional affect delta;
- backend policy, actual backend, and output hash;
- structural warnings without auto-repair.

### LIV-011C — Self-contained offline navigator

- no remote assets;
- searchable nodes;
- lineage, status, and mode filters;
- selectable session details;
- centered focus session;
- responsive layout;
- safe JSON embedding for arbitrary local labels.

### LIV-011D — Terminal and JSON surfaces

- useful terminal map without a browser;
- complete graph JSON;
- HTML path, HTML hash, and snapshot-hash receipts;
- explicit errors for missing scope and focus.

### LIV-011E — Product integration

- `sbgpy-session constellation`;
- public Python API exports;
- package inclusion;
- focused tests;
- operator guide and completion receipt.

## Acceptance

This train is complete only when:

- the view reveals why two sessions differ;
- an exact return is visibly exact;
- echoes and outcomes are inspectable from their session;
- actual backend/output identity remains visible;
- incomplete ancestry stays incomplete rather than invented;
- hostile local labels cannot break the HTML document;
- the generated file works offline;
- the archive remains unmodified by graph generation;
- Python 3.10–3.13 tests and the distributable package pass;
- the next product wave is handed off without reopening Wave 1 or creating audit machinery.
