# Original SBaGen TODO — Ecosystem Reconciliation

**Date:** July 31, 2026  
**Status:** Superseded as a Python-only implementation plan  
**Primary sources:**

- `sbagen-1.4.5/sbagen-1.4.5/TODO.txt`
- `https://uazu.net/sbagen/`
- `docs/research/IDOSER_SBAGEN_PAIN_POINT_SCOUT.md`
- `lm7137/SBaGenX@be7c74d378774a759f1e4149dc9df3617b4d0d3b`

## Scope correction

The previous version of this matrix compared the original SBaGen TODO only against PySbagen. That correctly exposed missing capabilities inside PySbagen, but it led to the wrong product conclusion: rebuild the entire experimenter workstation and advanced DSP stack in Python.

Inspection of the actual SBaGenX repository shows that many original TODO requests have already been implemented or substantially advanced in the modern SBaGen lineage, including:

- reusable `sbagenxlib` engine architecture;
- `.sbg` and `.sbgf` editing and validation;
- built-in drop/slide/sigmoid/curve programs;
- live carrier, beat, amplitude, and mix controls;
- multiple voices and auxiliary tones;
- `mixspin`, `mixbeat`, `mixpulse`, and `mixam`;
- isochronic, monaural, binaural, spin, bell, orbitbeat, and noise families;
- raw/WAV/OGG/FLAC/MP3 export;
- plotting, graph video, desktop packaging, and an Android frontend.

Those capabilities should not be independently recreated in PySbagen merely because PySbagen itself lacks them.

## Active source-to-product matrix

The authoritative reconciliation is now:

- `docs/planning/SBAGENX_DIFFERENTIATION_AND_INTEROP_MATRIX.md`

It classifies each capability as:

- **delegate** to SBaGenX;
- **interoperate** through a version-gated optional adapter;
- **shared/fallback** between engines;
- **PySbagen-owned** product work;
- **verify-later** before building anything.

## PySbagen-owned gaps that remain valid

The following are still distinct PySbagen product responsibilities:

- complete DRG package preservation;
- immutable source/package/recipe/session/output provenance;
- explicit compatibility-loss and render-disposition reports;
- local content-addressed protocol library;
- listening-path and source suitability qualification;
- guided human-question products;
- backend-independent session markers and event ledgers;
- outcome history and local preference learning;
- consent-aware research workflows;
- evidence-position and claim provenance;
- one-shot cue/session orchestration only if SBaGenX does not already provide it.

## Active implementation plan

- `.beads/pysbagen_sbagenx_interoperability_train_2026_07_31.md`

The earlier broad Python workstation recreation plan is archived as superseded:

- `.beads/pysbagen_experimenter_workstation_train_2026_07_31.md`

Creativity remains parked. The immediate work is ecosystem interoperability and PySbagen's genuinely differentiated session-intelligence layer.
