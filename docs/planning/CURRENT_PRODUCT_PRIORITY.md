# PySbagen Current Product Priority

**Date:** July 31, 2026  
**Status:** Compatibility delivered; SBaGenX interoperability and PySbagen differentiation are the active priority

## Delivered compatibility foundation

The I-Doser/SBaGen compatibility, preservation, inspection, qualification, and local-library train was merged through PR `#9` at `0c95a67ca65db22d6441b123a5709bcaf929a064`.

Delivered capabilities remain:

1. honest SBG/DRG import reports and render dispositions;
2. complete DRG package preservation;
3. original-SBaGen semantic compatibility matrix;
4. timeline/source inspection before playback;
5. audio-source and listening-path qualification;
6. local-first provenance library.

## Product-direction correction

A source-level review of `lm7137/SBaGenX` showed that the broad Python workstation plan would duplicate the active modern SBaGen lineage. SBaGenX already provides substantial native engine, `.sbg`/`.sbgf`, curve, mix-effect, live-control, multivoice, export, plotting, packaging, desktop, and Android work.

Therefore:

> PySbagen will not rebuild SBaGenX in Python.

The broad workstation recreation train is retained only as a superseded record:

- `.beads/pysbagen_experimenter_workstation_train_2026_07_31.md`

## Active priority

Authoritative boundary:

- `docs/planning/SBAGENX_DIFFERENTIATION_AND_INTEROP_MATRIX.md`

Active implementation train:

- `.beads/pysbagen_sbagenx_interoperability_train_2026_07_31.md`

Active branch and PR:

- `agent/sbagenx-interoperability-train-20260731`
- PR `#11` — **Differentiate PySbagen and begin SBaGenX interoperability**

## Product responsibilities

### SBaGenX

Treat SBaGenX as the optional advanced native SBaGen engine for:

- advanced SBG sequencing and DSP;
- `.sbgf` curves and built-in programs;
- native validation and future optional rendering;
- mixspin/mixbeat/mixpulse/mixam;
- native live parameter controls;
- multiple voices, auxiliary tones, plotting, export, and frontend work.

### PySbagen

Keep PySbagen focused on:

- SBG/SBGF/DRG inspection and preservation;
- explicit compatibility-loss reports;
- immutable protocol/package/session/output provenance;
- local-first protocol library;
- audio-source and listening-path qualification;
- guided human-question products;
- backend-independent session markers and event ledger;
- outcome history and local preference learning;
- evidence-position and claim provenance;
- consent-aware research workflows;
- one-shot cue/session orchestration only after checking upstream.

SBaGenX remains optional. The Python renderer remains the portable fallback and current guided-product engine.

## Implementation delivered on the active branch

### Backend discovery

- `pysbagen/sbagenx_backend.py`
- `sbgpy-inspect backend`
- executable/library discovery, candidate-versus-usable status, version/API identity, and symbol-backed capabilities

### Typed native validation

- `pysbagen/sbagenx_native.py`
- `sbgpy-inspect backend --validate SOURCE`
- exact API-47 ctypes signatures and diagnostic layout
- fail-closed unknown-API and missing-symbol behavior
- immutable source byte count and SHA-256
- SBG/SBGF diagnostics with severity, code, location, range, and message
- native library/version/API identity in deterministic reports

Native rendering remains deliberately disabled. It waits for render/context/writer bindings, dual-engine discrepancy reports, parity fixtures, explicit backend policy, and complete provenance sidecars.

## Creativity status

The creativity research remains preserved in:

- `docs/research/CREATIVITY_AUDIO_RESEARCH_FOUNDATIONS.md`
- `docs/planning/CREATIVITY_PRODUCT_GAP_CHECK.md`

**Creativity implementation remains deferred.** Interoperability and the differentiated session-intelligence foundation take precedence.

## Completion rule

This priority is complete only when:

- native integration is version- and symbol-gated;
- native validation and rendering cannot bypass compatibility truth or provenance;
- SBGF and DRG remain first-class preserved artifacts;
- backend choice is visible and reproducible;
- Python fallback remains supported;
- PySbagen-owned session/event/outcome features work above either backend;
- full tests and package builds pass before merge;
- verify-later rows are resolved rather than guessed.
