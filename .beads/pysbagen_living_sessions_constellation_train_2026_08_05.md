# PySbagen Living Sessions — Constellation Train

**Date:** August 5, 2026  
**Branch:** `agent/living-sessions-constellation-20260805`  
**Pull request:** `#14` — Build offline Living Sessions constellation navigator  
**Stack base:** `agent/living-sessions-train-20260731` / PR #12  
**Status:** complete and qualified

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

## Delivered beads

### LIV-011A — Deterministic constellation graph — complete

- nodes for stored sessions;
- directed parent/child edges;
- deterministic layout;
- snapshot SHA-256;
- lineage filtering and focus validation;
- read-only graph generation.

### LIV-011B — Evidence-rich node and edge model — complete

- recipe summaries and full recipe hashes;
- before/after mutation values;
- causal-interpretability labels;
- echoes and event anchors;
- outcomes and optional affect delta;
- backend policy, actual backend, and output hash;
- structural warnings without auto-repair.

### LIV-011C — Self-contained offline navigator — complete

- no remote assets;
- searchable nodes;
- lineage, status, and mode filters;
- selectable session details;
- centered focus session;
- responsive layout;
- safe JSON embedding for arbitrary local labels.

### LIV-011D — Terminal and JSON surfaces — complete

- useful terminal map without a browser;
- complete graph JSON;
- HTML path, HTML hash, and snapshot-hash receipts;
- explicit errors for missing scope and focus.

### LIV-011E — Product integration — complete

- `sbgpy-session constellation`;
- public Python API exports;
- package inclusion;
- focused tests;
- operator guide and completion receipt.

## Acceptance result

The train satisfies its product acceptance:

- the view reveals why two sessions differ;
- an exact return is visibly exact;
- echoes and outcomes are inspectable from their session;
- actual backend/output identity remains visible;
- incomplete ancestry stays incomplete rather than invented;
- hostile local labels cannot break the HTML document;
- the generated file works offline without remote assets;
- graph generation appends no session or archive event;
- Python 3.10–3.13 tests and the distributable package pass;
- the next product wave is handed off without reopening Wave 1 or creating audit machinery.

## Qualification

GitHub Actions Python qualification run `#67` passed implementation head `8ff3a2892f3b57f38cfe7cc7a25dbfe171ffb2c8`:

- Python 3.10 — passed;
- Python 3.11 — passed;
- Python 3.12 — passed;
- Python 3.13 — passed;
- complete repository result — **73 tests passed**;
- source distribution and wheel build — passed;
- wheel includes `pysbagen/constellation.py` and the updated session CLI.

The first qualification run exposed one wording-order mismatch in a test after **72 tests passed**. The regression was corrected to lock the actual user-facing interpretation language; no product or safety assertion was weakened.

## Next product wave

**LIV-012 — Confluence Sessions** is the next original product train: explicitly combine selected dimensions from two known lineages while preserving both parent identities, inherited-dimension receipts, conflicts, and causal uncertainty.

This is queued only inside PySbagen. The Cycloside cross-project possibility remains parked behind its explicit permission gate.
