# PySbagen Experimenter Workstation and Original-TODO Completion Beadtrain

**Date:** July 31, 2026  
**Status:** Superseded before implementation  
**Original branch:** `agent/experimenter-workstation-train-20260731` — not created or authorized  
**Replacement:** `.beads/pysbagen_sbagenx_interoperability_train_2026_07_31.md`

## Why this train was superseded

This plan was created after comparing the original SBaGen TODO only against PySbagen. It correctly identified capabilities absent from PySbagen, but it did not first compare them against the active modern SBaGen continuation.

A source-level review of `lm7137/SBaGenX` showed that the proposed train would duplicate substantial existing work:

- the reusable native engine and sequence runtime;
- `.sbg`/`.sbgf` editing and validation;
- curves and built-in programs;
- native mix effects including `mixspin` and `mixbeat`;
- live parameter controls;
- multiple voices and auxiliary tones;
- expanded noise/tone families;
- native export, plotting, packaging, and frontends.

Running WST-001 through WST-015 as written would turn PySbagen into a competing, less mature Python reimplementation of SBaGenX rather than a differentiated product.

## Preserved ideas

The following ideas remain valid but move into the interoperability train as PySbagen-owned session/product layers:

- backend-independent session markers and event ledger;
- outcome history and local preference learning;
- one-shot cue orchestration if still absent upstream;
- exact backend/version/protocol/output receipts;
- guided-product capability selection;
- DRG and SBGF preservation;
- loss-aware interchange and compatibility reporting;
- research consent and protocol-assignment workflows.

## Removed duplication

The replacement train does not independently recreate:

- a competing `.sbg`/`.sbgf` editor;
- a new curve/project language before `.sbgf` support;
- `mixspin`, `mixbeat`, `mixpulse`, or `mixam` DSP;
- native live carrier/beat/amplitude/mix ramps;
- the SBaGenX export stack, plot system, desktop packaging, or mobile frontend.

## Authoritative product boundary

- `docs/planning/SBAGENX_DIFFERENTIATION_AND_INTEROP_MATRIX.md`
- `.beads/pysbagen_sbagenx_interoperability_train_2026_07_31.md`

This archived file is retained so the scope correction remains traceable. No bead in this superseded train should be executed unless it is explicitly reintroduced after a fresh SBaGenX capability check.
