# PySbagen Current Product Priority

**Date:** July 31, 2026  
**Status:** Compatibility delivered; SBaGenX interoperability and PySbagen differentiation are the active priority

## Delivered foundation

The I-Doser/SBaGen compatibility, preservation, inspection, qualification, and local-library train was implemented and merged through:

- `.beads/pysbagen_compatibility_preservation_train_2026_07_31.md`
- PR `#9` — **Build the SBaGen and DRG compatibility product**
- merge commit `0c95a67ca65db22d6441b123a5709bcaf929a064`
- completion receipt `.beads/pysbagen_compatibility_preservation_train_2026_07_31_COMPLETION.md`

Delivered capabilities remain:

1. honest SBG/DRG import reports and render dispositions;
2. complete DRG package preservation;
3. original-SBaGen semantic compatibility matrix;
4. timeline/source inspection before playback;
5. audio-source and listening-path qualification;
6. local-first provenance library.

## Product-direction correction

The original SBaGen TODO identified many missing workstation and DSP features. A first follow-up plan assumed PySbagen should implement them all.

A source-level review of `lm7137/SBaGenX` showed that this would duplicate the active modern SBaGen lineage. SBaGenX already provides substantial native engine, `.sbg`/`.sbgf`, curve, mix-effect, live-control, multivoice, export, plotting, packaging, desktop, and Android work.

Therefore:

> PySbagen will not rebuild SBaGenX in Python.

The broad workstation recreation train is superseded:

- `.beads/pysbagen_experimenter_workstation_train_2026_07_31.md`

## Active priority

The authoritative boundary is:

- `docs/planning/SBAGENX_DIFFERENTIATION_AND_INTEROP_MATRIX.md`

The active implementation train is:

- `.beads/pysbagen_sbagenx_interoperability_train_2026_07_31.md`

Active branch:

- `agent/sbagenx-interoperability-train-20260731`

## Product responsibilities

### SBaGenX

Treat SBaGenX as the optional advanced native SBaGen engine for:

- advanced SBG sequencing and DSP;
- `.sbgf` curves and built-in programs;
- native validation and rendering;
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
- one-shot cue/session orchestration only after verifying it is not already available upstream.

SBaGenX remains optional. The Python renderer remains the portable fallback and current guided-product engine.

## Implementation started

The first interoperability foundation is now implemented:

- `pysbagen/sbagenx_backend.py`
- `sbgpy-inspect backend`

It discovers `SBAGENX_BIN`, `SBAGENXLIB_PATH`, system candidates, native version/API, and symbol-backed capabilities. It does not yet route rendering through SBaGenX; native rendering waits for typed bindings, parity fixtures, and complete provenance receipts.

## Creativity status

The creativity research and product gap remain valid and preserved in:

- `docs/research/CREATIVITY_AUDIO_RESEARCH_FOUNDATIONS.md`
- `docs/planning/CREATIVITY_PRODUCT_GAP_CHECK.md`

However, **creativity implementation remains deferred**. Interoperability and the differentiated session-intelligence foundation take precedence.

## Completion rule

This priority is complete only when:

- native integration is version- and symbol-gated;
- SBG/SBGF validation and optional rendering retain compatibility truth and provenance;
- backend choice is visible and reproducible;
- Python fallback remains supported;
- PySbagen-owned session/event/outcome features work above either backend;
- remaining verify-later rows are explicitly resolved rather than guessed.
