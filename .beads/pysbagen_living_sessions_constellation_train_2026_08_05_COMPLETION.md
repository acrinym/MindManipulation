# PySbagen Living Sessions — Constellation Completion Receipt

**Date:** August 5, 2026  
**Status:** complete and qualified  
**Branch:** `agent/living-sessions-constellation-20260805`  
**Pull request:** `#14` — Build offline Living Sessions constellation navigator  
**Stack base:** PR `#12` / `agent/living-sessions-train-20260731`

## Product delivered

Constellation turns the local Living Sessions archive into a continuing navigation surface with:

- deterministic session nodes and directed ancestry;
- exact recipe, lineage, parent, generation, and occurrence identity;
- visible return, branch, contrast, and wander relationships;
- before/after mutation values and recorded reasons;
- causal-interpretability labels;
- echoes, shifts, insights, and discomfort anchors;
- optional outcomes and affect deltas;
- backend policy, actual renderer, output path, and output SHA-256 receipts;
- structural warnings for incomplete or contradictory ancestry;
- terminal, JSON, and self-contained offline HTML views;
- search plus lineage, status, and mode filters;
- graph snapshot and HTML-file SHA-256 receipts.

## Product boundary retained

Constellation adds no duplicate native DSP.

- SBaGenX remains the optional advanced SBG/SBGF engine and authoring/runtime layer.
- PySbagen owns the continuing human/session memory and provenance surface above qualified renderers.
- The navigator reads existing session records and does not append viewing events.
- The Cycloside cross-project possibility remains permission-gated and untouched.

## HTE / InvisiSynth use

The project-owner supplied `HTE-Newest.zip` was used as a reasoning corpus:

- Memory Engine for episode continuity and uncertainty;
- Learning Engine for change/outcome/mismatch visibility;
- Affect Engine for state-tagged episodes;
- Dependency Grapher for directed ancestry;
- Parallel Oracle and Synthesis for simultaneous human and technical truth;
- InvisiSynth for visible missing parents and broken relationships.

No HTE runtime code, private configuration, or dependency was copied into PySbagen.

## User paths

```bash
sbgpy-session constellation
sbgpy-session constellation --json
sbgpy-session constellation --html
sbgpy-session constellation --lineage LINEAGE_ID --html lineage.html
sbgpy-session constellation --focus SESSION_ID --html focused.html
```

## Files delivered

Runtime:

- `pysbagen/constellation.py`
- constellation integration in `pysbagen/session_cli.py`
- public exports in `pysbagen/__init__.py`

Tests:

- `pysbagen/tests/test_constellation.py`

Documentation:

- `docs/product/LIVING_SESSIONS_CONSTELLATION_GUIDE.md`
- `.beads/pysbagen_living_sessions_constellation_train_2026_08_05.md`
- this completion receipt

## Qualification

GitHub Actions Python qualification run `#67` passed implementation head `8ff3a2892f3b57f38cfe7cc7a25dbfe171ffb2c8`:

- Python 3.10 product-path tests — passed;
- Python 3.11 product-path tests — passed;
- Python 3.12 product-path tests — passed;
- Python 3.13 product-path tests — passed;
- complete repository result — **73 tests passed**;
- Python 3.12 source distribution — passed;
- Python 3.12 wheel build — passed;
- the wheel includes `pysbagen/constellation.py` and updated public/CLI surfaces.

Focused coverage includes:

- deterministic graph and snapshot identity;
- parent/child ancestry and exact-return labeling;
- mutation and causal-interpretability receipts;
- echoes, outcomes, affect deltas, backend identities, and output hashes;
- lineage and focus fail-closed behavior;
- missing-parent warnings without invented edges;
- hostile `</script>` labels remaining inert;
- no external script or stylesheet dependency;
- atomic offline HTML writing;
- CLI export and hash receipts.

The first run failed only because the regression expected the phrase `personal descriptive records` while the product emitted `descriptive personal records`; **72 tests had passed**. The test was aligned with the actual user-facing language, and all 73 tests then passed. No functional, provenance, safety, or scope assertion was removed.

## Review state

- PR is open and mergeable.
- Bugbot is not enabled and performed no review.
- CodeRabbit automatic review skipped the non-default stacked base.
- A manual `@coderabbitai review` request was posted against the qualified head.
- Any later submitted findings must be resolved before merge; a success status without a review body is not represented as a completed review.

## Anti-drawer acceptance

Constellation creates repeat value because the user can:

- navigate a growing personal history rather than search filenames;
- return to an exact known session;
- understand one-change and bounded-exploration descendants;
- revisit remembered moments in context;
- compare outcomes without discarding provenance;
- see which renderer and exact output produced an experience;
- notice missing or contradictory history rather than being shown a falsely complete map.

It uses no points, badges, streak pressure, social ranking, analytics, cloud account, or hidden recommendation model.

## Next queued wave

**LIV-012 — Confluence Sessions**: combine explicitly selected dimensions from two known lineages while retaining both parents, inherited-dimension receipts, conflict disclosures, and causal uncertainty.

The Cycloside integration idea remains parked and may not be explored or built without Justin's explicit permission.
