# PySbagen Living Sessions Product Train

**Date:** July 31, 2026  
**Status:** Wave 1 complete and qualified; Wave 2 product queue active  
**Branch:** `agent/living-sessions-train-20260731`  
**Pull request:** `#12` — Build memorable Living Sessions above the audio engines  
**Stack base:** `agent/sbagenx-interoperability-train-20260731` / PR `#11`  
**Research:** `docs/research/HTE_LIVING_SESSIONS_GAP_SYNTHESIS_2026_07_31.md`  
**Wave 1 receipt:** `.beads/pysbagen_living_sessions_train_2026_07_31_WAVE1_COMPLETION.md`

## Goal

Make PySbagen a repeatedly useful, memorable personal session system without duplicating SBaGenX DSP and without using manipulative retention mechanics.

Repeated use must create honest new value through exact identity, lineage, remembered moments, transparent variation, and optional local learning.

## Boundary

### SBaGenX owns

- advanced native SBG/SBGF execution;
- curves, native DSP, export, plotting, authoring, and frontend work;
- future qualified native rendering.

### PySbagen owns

- session identity and exact recipe provenance;
- guided human products;
- backend-independent lineage and event records;
- echoes and remembered moments;
- transparent variation planning;
- optional outcome history and descriptive local patterns;
- local-first archives and exportable receipts.

## Anti-patterns

- no points, badges, loot mechanics, leaderboards, or coercive streaks;
- no hidden parameter changes;
- no random frequency invention outside validated product choices;
- no cloud account requirement;
- no opaque recommendation model;
- no medical-efficacy inference;
- no copied HTE runtime code;
- no changing the user's stated problem merely to manufacture novelty.

---

# Wave 1 — Complete

## LIV-001 — HTE and InvisiSynth gap synthesis

**Status:** complete

Applied the supplied HTE pack's InvisiSynth, Learning, Affect, Parallel Oracle, Synthesis, Intersection, and Lateral mechanisms. Recorded gaps, combinations, rejected ideas, and anti-drawer acceptance criteria.

## LIV-002 — Memorable identity bound to exact recipes

**Status:** complete

Delivered deterministic two-word titles, three unique motifs, exact recipe SHA-256, stable lineage IDs, unique occurrence IDs, parents, and generations.

Human identity never replaces machine provenance.

## LIV-003 — Return, branch, contrast, and wander

**Status:** complete

- `return` — exact recipe and remembered identity;
- `branch` — exactly one disclosed change;
- `contrast` — exactly one high-salience audible change; seed-only novelty is excluded when an audible choice exists;
- `wander` — at most two compatible disclosed changes, marked less causally interpretable.

The stated sleep problem remains unchanged.

## LIV-004 — Local append-only archive

**Status:** complete

Delivered immutable plan JSON, append-only event JSONL, immutable outcome JSON, idempotent creation, lineage queries, and chronological listing.

## LIV-005 — Echo memory markers

**Status:** complete

Delivered named echo, shift, insight, discomfort, and custom events with optional transport positions and structured local payloads.

Echoes are metadata anchors; audio extraction remains deferred.

## LIV-006 — Affect-tagged optional outcomes

**Status:** complete

Delivered optional pre/post valence, arousal, and agency; rating; comfort; would-repeat; notes; tags; and immutable outcome history.

These are personal descriptive records, not clinical measurements.

## LIV-007 — Transparent next-mode recommendation

**Status:** complete

Rules are explicit:

- no outcome → branch;
- uncomfortable or rating 1–2 → contrast;
- strong first result with repeat intent → return;
- repeated strong exact return → branch;
- otherwise → branch.

## LIV-008 — Personal atlas

**Status:** complete

Delivered session/lineage/echo counts, optional average rating and repeat rate, affect deltas, lineage title histories, and descriptive local pattern candidates after repeated observations.

## LIV-009 — Product CLI

**Status:** complete

Delivered `sbgpy-session`:

- `new-sleep`;
- `next`;
- `show`;
- `list`;
- `mark`;
- `finish`;
- `render`;
- `atlas`.

Rendering reconstructs the exact stored request and records recipe, actual backend, backend reason, output properties, and output SHA-256.

Backend policy is fail-closed:

- `python` → Python;
- `auto` → currently Python with recorded reason;
- `sbagenx` → refusal until native rendering is qualified.

## LIV-010 — Qualification and package integration

**Status:** complete

GitHub Actions Python qualification run `#57` passed:

- Python 3.10 — passed;
- Python 3.11 — passed;
- Python 3.12 — passed;
- Python 3.13 — passed;
- **68 tests passed**;
- source distribution and wheel built;
- wheel includes Living Sessions runtime, policy, and CLI modules;
- SPDX license metadata removed the previous setuptools license-table warning.

Review truth:

- self-review found and fixed seed-only contrast and silent native-policy fallback;
- Bugbot was unavailable;
- CodeRabbit auto-review skipped the stacked base;
- manual CodeRabbit request produced success status but no submitted review or inline thread at receipt time.

---

# Wave 2 — Product Queue

## LIV-011 — Constellation view

**Status:** next

Build an offline visual/TUI graph showing roots, returns, branches, edge mutations, echoes, outcomes, backend identities, and hashes on demand.

This must be a usable navigation surface, not a decorative graph.

## LIV-012 — Confluence sessions

**Status:** queued after constellation foundation

Combine two understood lineages while preserving both parents, checking compatible dimensions, disclosing all inheritance/changes, marking reduced causal interpretability, and remaining exactly reproducible.

## LIV-013 — Echo Weaving

**Status:** queued after upstream cue verification

Turn selected echoes into structural or cue anchors above either renderer. Coordinate with SBX-011; do not duplicate native DSP or blindly copy rendered audio.

## LIV-014 — Shareable seed capsule

**Status:** queued

Export exact identity, recipe/source hashes, lineage summary, backend requirements, and referenced-media manifest. Exclude personal affect/outcome history by default.

## LIV-015 — Cross-product Living Sessions

**Status:** queued

Generalize the model to imported SBG, native-required SBGF, research protocols, future guided products, and both Python/SBaGenX backends.

## Current implementation files

- `pysbagen/living_sessions.py`
- `pysbagen/living_session_policy.py`
- `pysbagen/session_cli.py`
- `pysbagen/tests/test_living_sessions.py`
- `docs/research/HTE_LIVING_SESSIONS_GAP_SYNTHESIS_2026_07_31.md`
- `docs/product/LIVING_SESSIONS_GUIDE.md`
- `.beads/pysbagen_living_sessions_train_2026_07_31_WAVE1_COMPLETION.md`

## Next execution target

Run **LIV-011 — Constellation view** as the next product train. Do not reopen Wave 1 as another audit layer unless a real regression appears.
