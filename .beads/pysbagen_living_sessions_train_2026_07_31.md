# PySbagen Living Sessions Product Train

**Date:** July 31, 2026  
**Status:** Active — first complete return loop implemented  
**Branch:** `agent/living-sessions-train-20260731`  
**Stack base:** `agent/sbagenx-interoperability-train-20260731` / PR `#11`  
**Research input:** `docs/research/HTE_LIVING_SESSIONS_GAP_SYNTHESIS_2026_07_31.md`

## Train goal

Make PySbagen a repeatedly useful, memorable personal session system without duplicating SBaGenX DSP and without relying on manipulative engagement mechanics.

A completed session must create future value through exact identity, lineage, remembered moments, transparent variation, and optional local learning.

## Product boundary

### SBaGenX remains responsible for

- advanced native SBG/SBGF execution;
- curves and expression programs;
- native DSP, export, plotting, authoring, and frontends;
- future qualified native rendering selected by explicit policy.

### PySbagen owns

- session identity and exact recipe provenance;
- human-guided products;
- backend-independent lineage and event records;
- echoes and remembered moments;
- transparent variation planning;
- optional outcome history and descriptive local pattern detection;
- local-first archives and exportable receipts.

## Anti-patterns

- no points, badges, loot mechanics, leaderboards, or coercive streaks;
- no hidden parameter changes;
- no random frequency generation outside validated product choices;
- no cloud account requirement;
- no opaque recommendation model;
- no medical-efficacy inference;
- no copying HTE runtime code into PySbagen;
- no changing the stated user problem merely to manufacture novelty.

---

## LIV-001 — HTE and InvisiSynth product-gap synthesis

**Status:** complete

Applied the supplied HTE-Newest pack's InvisiSynth, Learning, Affect, Parallel Oracle, Synthesis, Intersection, and Lateral concepts.

**Delivered:**

- `docs/research/HTE_LIVING_SESSIONS_GAP_SYNTHESIS_2026_07_31.md`;
- explicit domain-gap findings;
- cross-domain mechanism table;
- anti-drawer acceptance test;
- rejected manipulative or disposable concepts.

## LIV-002 — Memorable identity bound to exact recipes

**Status:** complete

Delivered deterministic:

- two-word session title;
- three-word motif;
- recipe SHA-256;
- stable lineage ID;
- unique session occurrence ID;
- human memory phrase that never replaces machine provenance.

## LIV-003 — Return, branch, contrast, and wander modes

**Status:** complete

Delivered four explicit routes:

- `return` — exact recipe and remembered identity;
- `branch` — exactly one disclosed change;
- `contrast` — one high-salience disclosed change;
- `wander` — at most two compatible disclosed changes, marked experimental and less causally interpretable.

Current mutation dimensions are restricted to existing product-level controls:

- generated-bed seed;
- duration;
- intensity;
- sound world and its user-audio binding;
- one underlying layer toggle.

The stated sleep problem remains unchanged.

## LIV-004 — Local append-only archive

**Status:** complete

Delivered a platform-local archive containing:

- immutable plan JSON;
- append-only event JSONL;
- immutable outcome JSON;
- content-derived identifiers;
- idempotent plan creation;
- lineage queries;
- chronological session listing.

## LIV-005 — Echo memory markers

**Status:** complete

Delivered named `echo`, `shift`, `insight`, `discomfort`, and custom events with:

- session/lineage identity;
- optional transport position;
- timestamp;
- human label;
- optional structured payload.

Echoes remain metadata anchors. Audio extraction/recomposition is deferred to a later orchestration bead.

## LIV-006 — Affect-tagged optional outcomes

**Status:** complete

Delivered optional:

- pre-session valence, arousal, and agency snapshot;
- post-session snapshot;
- rating;
- comfort state;
- would-repeat signal;
- notes and tags;
- immutable outcome records.

These records are personal description, not clinical measurement.

## LIV-007 — Transparent next-mode recommendation

**Status:** complete

Delivered simple explainable rules:

- no outcome → branch;
- uncomfortable or low-rated outcome → contrast;
- strong first outcome with repeat intent → exact return;
- repeated strong exact return → branch;
- otherwise → branch.

No hidden score or remote model is involved.

## LIV-008 — Personal atlas

**Status:** complete

Delivered:

- planned, active, completed, lineage, and echo counts;
- average optional rating;
- would-repeat rate;
- average recorded affect delta;
- lineage title histories;
- descriptive sound-world and mode pattern candidates after repeated observations.

## LIV-009 — Living Sessions CLI

**Status:** complete

Delivered `sbgpy-session` commands:

- `new-sleep`;
- `next`;
- `show`;
- `list`;
- `mark`;
- `finish`;
- `render`;
- `atlas`.

The render command reconstructs and validates the exact stored SleepRequest, renders through the current Python backend, hashes the output, and appends a backend/recipe/output event receipt.

## LIV-010 — Qualification and package integration

**Status:** in progress

Required:

- Python 3.10–3.13 test matrix;
- complete repository tests;
- source distribution and wheel build;
- wheel contains `living_sessions.py` and `session_cli.py`;
- installed `sbgpy-session --help` works;
- no regression in PR #11 interoperability, SBGF, DRG, library, rendering, or Sleep Guide paths;
- review comments and nits resolved where compatible with product philosophy.

## LIV-011 — Constellation view

**Status:** queued

Build a visual or TUI graph showing:

- root sessions;
- returns and branches;
- mutations on edges;
- echoes on nodes;
- outcome context;
- backend identity;
- source/recipe hashes on demand.

The graph must remain useful with no network connection.

## LIV-012 — Confluence sessions

**Status:** queued after lineage qualification

Combine two understood lineages while preserving both parent identities.

Requirements:

- two-parent provenance;
- compatible-dimension checks;
- every inherited and changed dimension disclosed;
- no silent blending of contradictory intents;
- mark lower causal interpretability;
- exact reproducibility.

## LIV-013 — Echo weaving

**Status:** queued after one-shot cue verification

Turn selected echoes into backend-independent orchestration anchors:

- structural reprise;
- cue trigger;
- gentle contrast point;
- marker-relative event;
- no blind audio copying;
- complete receipt.

Coordinate with SBX-011 so SBaGenX capabilities are not duplicated.

## LIV-014 — Shareable seed capsule

**Status:** queued

Export a self-contained identity and recipe capsule that includes:

- exact recipe/source hashes;
- title and motif;
- lineage summary;
- backend requirements;
- optional referenced media manifest;
- no personal affect/outcome history unless explicitly selected.

## LIV-015 — Cross-product Living Sessions

**Status:** queued

Generalize the archive and lineage model beyond Sleep Guide to:

- imported SBG sessions;
- native-required SBGF programs;
- research protocols;
- future guided creativity, pain, cessation, or focus products;
- Python and SBaGenX backends through the same experience layer.

## Current implementation files

- `pysbagen/living_sessions.py`
- `pysbagen/session_cli.py`
- `pysbagen/tests/test_living_sessions.py`
- `docs/research/HTE_LIVING_SESSIONS_GAP_SYNTHESIS_2026_07_31.md`
- `docs/product/LIVING_SESSIONS_GUIDE.md`

## Definition of done

This train is complete only when:

- continued use creates honest new value;
- exact return and disclosed variation both work;
- user memory never weakens recipe provenance;
- local history remains inspectable and reversible;
- backend selection remains explicit;
- native DSP is not duplicated;
- no manipulative retention mechanism is introduced;
- full tests and package builds are green;
- queued visual, confluence, echo-weaving, capsule, and cross-product work is either delivered or handed off explicitly.
