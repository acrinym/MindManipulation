# PySbagen × SBaGenX Differentiation and Interoperability Beadtrain

**Date:** July 31, 2026  
**Status:** Active — discovery, API-47 validation, dual-engine truth, and SBGF preservation implemented and qualified  
**Branch:** `agent/sbagenx-interoperability-train-20260731`  
**Pull request:** `#11` — Differentiate PySbagen and begin SBaGenX interoperability  
**Source matrix:** `docs/planning/SBAGENX_DIFFERENTIATION_AND_INTEROP_MATRIX.md`  
**SBaGenX reference:** `lm7137/SBaGenX@be7c74d378774a759f1e4149dc9df3617b4d0d3b`

## Train goal

Make PySbagen the provenance-first, human-facing protocol and session operating layer for the SBaGen ecosystem while treating SBaGenX as the optional advanced native SBaGen engine.

PySbagen will not rebuild SBaGenX's `.sbg`/`.sbgf` editor, curve engine, mix effects, live controls, multivoice DSP, native export stack, plotting, packaging, or mobile frontend.

SBaGenX remains optional. PySbagen's Python renderer remains the portable fallback and current guided-product engine.

## Boundaries

- Do not copy SBaGenX source wholesale into PySbagen.
- Do not vendor or redistribute SBaGenX binaries without license/dependency review.
- Do not make SBaGenX mandatory for DRG preservation, inspection, the local library, or Sleep Guide use.
- Do not claim bit-identical cross-engine output unless fixture-proven.
- Do not invent another curve language before first-class `.sbgf` preservation.
- Do not independently recreate confirmed SBaGenX DSP or editor capabilities.
- Every native operation must record backend version, API version, source identity, and operation result.
- Unknown API revisions and missing symbols fail closed.

---

## SBX-001 — Source-pinned differentiation matrix

**Status:** complete

Inspected the actual SBaGenX source and classified work as delegate, interoperate, shared/fallback, PySbagen-owned, or verify-later.

**Delivered:** `docs/planning/SBAGENX_DIFFERENTIATION_AND_INTEROP_MATRIX.md`.

## SBX-002 — Optional backend discovery and capability probe

**Status:** complete

Delivered:

- `pysbagen/sbagenx_backend.py`;
- `sbgpy-inspect backend`;
- JSON and human reports;
- `SBAGENX_BIN`, PATH, `SBAGENXLIB_PATH`, and system-library discovery;
- executable identity from the real `sbagenx -h` banner;
- `sbx_version()` and `sbx_api_version()`;
- symbol-backed capability reporting;
- discovery-only mode;
- truthful distinction between candidate found and usable backend;
- explicit Python-fallback reporting when absent.

Discovery does not authorize native rendering.

## SBX-003 — Version-gated native binding contract

**Status:** complete for validation operations

Delivered a narrow API-47 ctypes contract in `pysbagen/sbagenx_native.py`:

- exact function signatures for version, API, SBG validation, SBGF validation, and diagnostic cleanup;
- exact API-47 `SbxDiagnostic` layout;
- fail-closed API revision policy;
- required-symbol checks;
- native-memory cleanup;
- empty-version and malformed/null-diagnostic defenses;
- structured errors instead of fallback guesses.

Render/context/writer bindings remain intentionally unimplemented until SBX-006.

## SBX-004 — Native SBG and SBGF validation adapter

**Status:** complete

Delivered:

- `validate_sbagenx_source()` for exact API-47 native validation;
- `inspect_with_sbagenx()` for one combined PySbagen/SBaGenX truth report;
- `sbgpy-inspect backend --validate SOURCE`;
- immutable source byte count and SHA-256 checked by both paths;
- UTF-8 BOM, UTF-8, and Latin-1 source handling;
- native severity/code/line/column/range/message diagnostics;
- native library path, version, and API identity;
- PySbagen compatibility disposition preserved independently;
- explicit discrepancy codes when PySbagen accepts and SBaGenX rejects, SBaGenX accepts and PySbagen limits, native errors exist, source identities differ, or native semantics may exceed a PySbagen approximation;
- exit `0` only for an identity-matched, native-valid, PySbagen-safe source without discrepancies;
- exit `1` for disclosed cross-engine limitations/discrepancies;
- exit `2` for identity mismatch, native rejection, or operational failure.

Native success never erases PySbagen blockers, unsupported states, approximations, missing sources, or provenance warnings.

## SBX-005 — SBGF preservation and inspectable protocol identity

**Status:** complete

Delivered `pysbagen/sbgf.py` and first-class `.sbgf` inspection:

- immutable original bytes, byte count, SHA-256, and encoding;
- parameter declarations and source expressions;
- solve directives with source lines;
- output/expression assignments;
- referenced function names;
- quoted media dependencies, resolved paths, source lines, and missing-source findings;
- unknown lines preserved rather than dropped or speculatively interpreted;
- explicit `inspection-only` disposition and qualified-native-runtime requirement;
- no invented SBG timeline for a function-driven program;
- ordinary `sbgpy-inspect inspect SOURCE.sbgf` support;
- local content-addressed library storage and offline source-hash verification;
- combined native validation through the interoperability report.

PySbagen does not replace `.sbgf` with a competing project language and does not claim to render its expression semantics independently.

## SBX-006 — Optional native render backend with receipts

**Status:** queued — next implementation bead

Add explicit `python`, `sbagenx`, and capability-gated `auto` backend selection. Prefer the native library; never silently shell out. Receipts must record backend/version/API, source identity, format, duration, configuration, output hash, and discrepancies.

No native render path may land before typed context/writer cleanup, representative parity fixtures, and output receipts are all present.

## SBX-007 — Cross-engine parity and discrepancy laboratory

**Status:** queued

Use legal/synthetic fixtures for shared binaural, monaural, isochronic, timing, transition, silence, noise, and mix semantics. Compare duration, frames, RMS/peak, channel/frequency relationships, transitions, and deterministic repeat behavior without demanding bit identity.

## SBX-008 — Guided-product backend policy

**Status:** queued

Allow products to request capabilities rather than hard-code engines. Sleep Guide remains portable by default; SBGF or advanced native recipes may be native-preferred or native-required. Save the visible selection reason in every exact recipe/session record.

## SBX-009 — Backend-independent session markers and event ledger

**Status:** queued

Build append-only named/quick markers with transport/wall-clock time, backend identity, protocol hash, telemetry snapshot where available, post-session notes, and privacy-safe JSON/CSV export.

## SBX-010 — Outcome history and local preference learning

**Status:** queued

Record exact played protocol/backend/path, intended purpose, immediate and delayed response, comfort/adverse effects, preference signals, and version differences. Learning remains local, explainable, reversible, and non-medical.

## SBX-011 — One-shot cue and orchestration layer

**Status:** queued after upstream verification

First verify whether current SBaGenX already provides timed one-shot cues. If absent, implement cues above the selected renderer with exact trigger policy, source identity, gain/pan/ducking/overlap, continuous-renderer preservation, ledger entry, and receipt.

## SBX-012 — Installation, discovery, and license boundary

**Status:** partial

The initial guide documents separate installation, environment overrides, candidate/qualification semantics, API-47 scope, and non-vendoring policy. Still required: platform-specific runtime locations, dependency failures, supported future API table, and complete attribution/license packaging guidance.

## SBX-013 — End-to-end qualification and completion receipt

**Status:** queued

Required journeys:

1. no SBaGenX installed → existing Python/inspection paths remain usable;
2. native candidate → truthful identity/capability report;
3. incompatible API → safe refusal;
4. SBG validation → combined dual-engine findings;
5. SBGF preservation → local library → native validation → exact identity;
6. explicit native render → provenance sidecar;
7. visible automatic backend decision;
8. portable guided Sleep journey;
9. marker → outcome → preference history;
10. existing DRG, compatibility, library, and sleep suites remain green.

## Current qualification receipt

GitHub Actions **Python qualification run #49** passed the complete current implementation:

- Python 3.10 product-path tests — passed;
- Python 3.11 product-path tests — passed;
- Python 3.12 product-path tests — passed;
- Python 3.13 product-path tests — passed;
- complete repository result — **59 tests passed**;
- Python 3.12 source distribution and wheel build — passed;
- wheel inspection includes:
  - `pysbagen/sbagenx_backend.py`;
  - `pysbagen/sbagenx_native.py`;
  - `pysbagen/sbgf.py`;
  - `pysbagen/interoperability.py`.

The build emits an existing setuptools deprecation warning for the TOML-table form of `project.license`; it does not fail this train but should be repaired before February 18, 2027. GitHub's runner also warns that Node 20 actions are being forced onto Node 24; the workflow remains green.

CodeRabbit status is green. Two correctness threads were addressed and resolved:

1. candidate discovery was separated from successful backend qualification, with normal CLI failure for unusable candidates;
2. unrecognized help banners no longer become fake version strings.

Regression coverage includes:

- executable help-banner identity and unrecognized-banner refusal;
- native library version/API/symbol discovery;
- discovered-versus-qualified state;
- non-executable and unloadable candidates;
- missing backend/configured path reporting;
- exact API-47 diagnostic decoding;
- rejection of unknown API revisions;
- rejection of malformed null diagnostic pointers;
- Latin-1 source preservation and SHA-256 identity;
- SBGF structure, solve directives, function inventory, missing media, local-library verification, and no invented timeline;
- native-valid/PySbagen-limited discrepancies;
- PySbagen-accepted/native-rejected discrepancies.

## Definition of done

This train is complete only when:

- confirmed SBaGenX features are no longer scheduled for duplicate implementation;
- optional native integration is version/symbol gated;
- native validation/rendering cannot bypass compatibility truth or provenance;
- SBGF and DRG remain first-class preserved artifacts;
- backend choice is visible and reproducible;
- session intelligence works above either renderer;
- Python fallback remains supported;
- full tests and package builds pass before merge;
- remaining verify-later rows are resolved honestly.
