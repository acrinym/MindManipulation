# PySbagen × SBaGenX Differentiation and Interoperability Beadtrain

**Date:** July 31, 2026  
**Status:** Active — implementation started  
**Branch:** `agent/sbagenx-interoperability-train-20260731`  
**Source matrix:** `docs/planning/SBAGENX_DIFFERENTIATION_AND_INTEROP_MATRIX.md`  
**SBaGenX reference:** `lm7137/SBaGenX@be7c74d378774a759f1e4149dc9df3617b4d0d3b`

## Train goal

Make PySbagen the provenance-first, human-facing protocol and session operating layer for the SBaGen ecosystem while treating SBaGenX as the optional advanced native SBaGen engine.

This train deliberately avoids rebuilding SBaGenX's `.sbg`/`.sbgf` editor, curve engine, mix effects, live parameter controls, native export stack, plotting system, packaging, or mobile frontend.

At completion, PySbagen must be able to:

- detect and qualify an installed SBaGenX executable or native library;
- preserve exact SBaGenX version/API/capability identity;
- validate and inspect SBG/SBGF through a version-gated native adapter;
- optionally render through SBaGenX with complete provenance and discrepancy receipts;
- retain the Python renderer as a portable fallback;
- record session markers, user events, outcomes, and local preferences independently of the rendering backend;
- run existing guided experiences through an explicit backend-selection policy;
- preserve DRG and SBGF artifacts without pretending conversion is lossless;
- never silently route work through an unqualified native backend.

Creativity implementation remains deferred.

## Boundaries

- Do not copy SBaGenX source wholesale into PySbagen.
- Do not vendor or redistribute SBaGenX binaries without a separate license/dependency review.
- Do not make SBaGenX mandatory for DRG preservation, inspection, the local library, or existing Sleep Guide use.
- Do not claim bit-identical cross-engine output unless fixture-proven.
- Do not invent another curve language before supporting `.sbgf` preservation and native validation.
- Do not independently recreate `mixspin`, `mixbeat`, `mixpulse`, `mixam`, native live ramps, or the SBaGenX editor.
- Every native operation must record backend version, API version, capabilities, source identity, invocation policy, and output identity.
- Unknown API revisions or missing symbols fail closed to inspection/fallback rather than guessing.

---

# Beads

## SBX-001 — Source-pinned differentiation matrix

**Status:** complete

Inspect the actual SBaGenX repository and classify proposed work as delegate, interoperate, shared/fallback, PySbagen-owned, or verify-later.

**Delivered:** `docs/planning/SBAGENX_DIFFERENTIATION_AND_INTEROP_MATRIX.md`.

## SBX-002 — Optional backend discovery and capability probe

**Status:** complete

Add a standard-library-only discovery surface for:

- `SBAGENX_BIN` and PATH executable lookup;
- `SBAGENXLIB_PATH` and system shared-library lookup;
- executable version probing;
- `sbx_version()` and `sbx_api_version()`;
- symbol-backed capabilities for native rendering, SBG/SBGF validation, container writing, live controls, and mix-stream processing;
- text and deterministic JSON reports;
- explicit fallback when SBaGenX is absent.

**Delivered:**

- `pysbagen/sbagenx_backend.py`
- `sbgpy-inspect backend`
- unit tests using a fake native library

Discovery does not yet authorize native rendering.

## SBX-003 — Version-gated native binding contract

**Status:** queued  
**Depends on:** SBX-002

Build a narrow ctypes binding around only the symbols PySbagen actually uses.

Requirements:

- exact argument/restype declarations;
- API-version policy table;
- symbol requirements per operation;
- owned-memory cleanup for diagnostics/contexts/writers;
- platform-safe loading and actionable errors;
- no broad, untyped dynamic calls;
- backend identity object reusable by manifests.

**Acceptance:** Unknown or incomplete libraries produce a structured unavailable/unsupported result without crashing or changing fallback behavior.

## SBX-004 — Native SBG and SBGF validation adapter

**Status:** queued  
**Depends on:** SBX-003

Expose native validation while preserving PySbagen's own compatibility report.

Requirements:

- validate source text without mutating it;
- map native diagnostics to source line/column/severity/code;
- record native engine/API identity;
- retain both PySbagen and SBaGenX findings when they disagree;
- never let native success erase PySbagen loss/provenance warnings.

**Acceptance:** One report can show Python-parser findings, native findings, and discrepancies side by side.

## SBX-005 — SBGF preservation and inspectable protocol identity

**Status:** queued  
**Depends on:** SBX-004

Treat `.sbgf` as a first-class preserved artifact:

- immutable source bytes and hash;
- encoding and syntax diagnostics;
- declared parameters, solve directive, expression families, and source locations where available;
- referenced media and source dependencies;
- native version/API used for validation;
- explicit rendered-only/equivalent/partial states for conversions.

Do not attempt to replace `.sbgf` with a new PySbagen project language.

## SBX-006 — Optional native render backend with receipts

**Status:** queued  
**Depends on:** SBX-003 through SBX-005

Add explicit backend selection:

- `python` — current PySbagen renderer;
- `sbagenx` — qualified native library;
- `auto` — select native only for fixture-qualified semantics, otherwise Python or blocked according to policy.

Every render receipt must include:

- selected backend and reason;
- SBaGenX version/API and capability set when native;
- source hash and import report;
- sample rate, format, duration, and output hash;
- unsupported/approximated fields;
- cross-engine comparison status when available.

No silent subprocess rendering. Prefer the native library; CLI invocation is a separately disclosed fallback only if required.

## SBX-007 — Cross-engine parity and discrepancy laboratory

**Status:** queued  
**Depends on:** SBX-006

Build representative legal/synthetic fixtures for shared semantics:

- binaural, monaural, isochronic;
- finite and open-ended sequences;
- transitions and silence;
- white/pink noise;
- background mix where both engines support it.

Measure:

- duration/frame count;
- RMS/peak and channel relationships;
- dominant frequencies and beat relationships;
- transition boundaries;
- deterministic repeat behavior;
- documented expected differences.

The output is a product discrepancy report, not a demand for bit identity.

## SBX-008 — Guided-product backend policy

**Status:** queued  
**Depends on:** SBX-006, SBX-007

Allow guided products to request capabilities rather than hard-code engines.

Examples:

- Sleep Guide can remain on the Python journey engine by default;
- an advanced imported SBGF session may require SBaGenX;
- a recipe may be portable, native-preferred, or native-required;
- backend choice must be visible before playback and saved in the exact recipe/session manifest.

Existing guided journeys must remain usable without SBaGenX installed.

## SBX-009 — Backend-independent session markers and event ledger

**Status:** queued  
**Depends on:** SBX-006

Build a PySbagen-owned append-only session ledger:

- named and quick markers;
- transport-relative and wall-clock time;
- backend identity and position;
- active recipe/protocol hash;
- native telemetry snapshot when available;
- optional post-session notes;
- privacy-safe JSON/CSV export.

This feature belongs above either renderer and must not be buried in one GUI.

## SBX-010 — Outcome history and local preference learning

**Status:** queued  
**Depends on:** SBX-009

Add local records for:

- what was played and through which backend/path;
- exact parameters and source identities;
- intended purpose;
- immediate and delayed subjective response;
- comfort/adverse effects;
- preference signals;
- repeatability and protocol-version differences.

Learning remains local, explainable, reversible, and recipe-specific. Do not infer medical efficacy.

## SBX-011 — One-shot cue and orchestration layer

**Status:** queued after verification  
**Depends on:** SBX-006, SBX-009

First verify whether current SBaGenX provides a completed timed one-shot cue path.

If absent, implement cues as PySbagen session orchestration above the selected renderer:

- exact or event-relative trigger time;
- source hash, gain, pan, ducking, overlap, and retrigger policy;
- no reset of continuous native/Python synthesis;
- event-ledger entry and output/session receipt.

Do not fork SBaGenX DSP merely to add orchestration.

## SBX-012 — Installation, discovery, and license boundary

**Status:** queued  
**Depends on:** SBX-003, SBX-006

Document and qualify:

- separate SBaGenX installation;
- environment overrides and system discovery;
- supported API ranges;
- Windows, Linux, and macOS library names/locations;
- optional-dependency behavior;
- license and attribution responsibilities;
- diagnostics for incompatible/missing runtime dependencies.

PySbagen packages must not silently bundle third-party native artifacts.

## SBX-013 — End-to-end qualification and completion receipt

**Status:** queued  
**Depends on:** SBX-001 through SBX-012

Prove complete journeys:

1. no SBaGenX installed → Python/inspection paths continue normally;
2. native library installed → version/capability report;
3. SBG validation → dual-engine findings and discrepancy record;
4. SBGF preservation → native validation → exact identity;
5. explicit native render → provenance sidecar;
6. auto backend selection → visible reason;
7. guided Sleep journey remains portable;
8. session marker → outcome record → local preference history;
9. incompatible API → safe refusal/fallback;
10. existing DRG, compatibility, library, and sleep tests remain green.

## Definition of done

This train is complete only when:

- PySbagen no longer plans to duplicate confirmed SBaGenX features;
- optional native integration is version/symbol gated;
- native validation/rendering cannot bypass compatibility truth or provenance;
- SBGF and DRG remain preserved first-class artifacts;
- backend choice is visible and reproducible;
- session intelligence works above either renderer;
- the Python fallback remains supported;
- tests and package builds pass across supported Python versions;
- the completion receipt lists any remaining verify-later rows honestly.
