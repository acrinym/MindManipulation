# PySbagen Compatibility, Preservation, and Inspection Train — Completion Receipt

**Date:** July 31, 2026  
**Branch:** `agent/compatibility-preservation-train-20260731`  
**Status:** Implementation complete; PR qualification pending  
**Source plan:** `.beads/pysbagen_compatibility_preservation_train_2026_07_31.md`

## Train result

The 13-bead compatibility train was implemented as one coherent product path. SBG and DRG artifacts now enter through a canonical import contract before rendering, playback, extraction, or library storage.

## Completed beads

### CP-001 — Canonical import contract

Added explicit compatibility states, render dispositions, deterministic JSON, human text generated from structured data, source hashes, metadata, package elements, warnings, blockers, missing sources, duration, timing, loop behavior, and schema/importer versions.

### CP-002 — Honest SBG import report

Added non-destructive SBG inspection through API, terminal, and desktop paths. Reports retain unsupported and approximated constructs, missing audio, source locations, start mode, end behavior, and render safety.

### CP-003 — Complete DRG preservation

Replaced schedule/image-only handling with a structured package model that retains original bytes, every decoded element in order, opaque elements, encrypted/decrypted identities, metadata, image bytes, nested SBG source, warnings, and a deterministic preservation manifest.

### CP-004 — Honest DRG import report

Composed package inspection with nested SBG semantic inspection. Package recovery can succeed partially without erasing other recovered elements, and an unsupported nested schedule can never make the DRG fully compatible.

### CP-005 — Semantic compatibility matrix

Added a machine-readable, versioned matrix and matching human document covering core definitions, tones, mixing, noise, files, timing, event operations, transitions, motion syntax, finite/open schedules, directives, historical backend controls, and PySbagen extensions.

### CP-006 — Fixture-proven semantic repairs

Hardened UTF-8/Latin-1 schedule loading and rejected duplicate tone-set labels instead of silently overwriting them. Existing silence, streaming, duration, and crossfade paths remain the renderer foundation.

### CP-007 — Unsupported-feature policy

Added one render policy shared by API, CLI, and GUI:

- supported/equivalent proceed;
- partial/approximated require acknowledgement;
- unsupported/unknown/missing-source/intentionally-excluded remain inspection-only;
- unsafe or malformed imports are blocked.

Schedule renders receive a `.pysbagen.json` sidecar with the exact import report, output identity, and disclosed changes.

### CP-008 — Timeline/source inspector model

Added a toolkit-independent serializable timeline with chronological events, active tone sets, component parameters, silence, transitions, file layers, duration, and open-ended spans.

### CP-009 — Inspection before playback interfaces

Added `sbgpy-inspect` and `sbgpy-inspect-gui`. The desktop inspector shows the report, timeline, DRG elements, and source qualification without requiring PyAudio. Render controls are driven by the canonical policy.

### CP-010 — Imported-source qualification

Added bounded-memory source analysis for container, codec, channels, sample rate, duration, peak, clipping, stereo correlation, near-mono state, resampling, and suitability. ffprobe provides metadata fallback where available.

### CP-011 — Listening/rendering-path qualification

Added explicit route qualification for method, channel count, sample rate, headphones/speakers, Bluetooth, spatial processing, and normalization. Directly observed information is separated from user-declared external processing.

### CP-012 — Local-first provenance library

Added content-hash identity, immutable source copies, import reports, timelines, DRG preservation bundles, explicit lifecycle states, duplicate provenance retention, supersession links, offline verification, and self-describing export manifests.

### CP-013 — Complete journey qualification

Added synthetic and repository-authored tests for supported SBG inspection, open-ended schedules, acknowledged approximations, duplicate labels, missing sources, complete synthetic DRG preservation, output sidecars, offline library verification, source/path qualification, and matrix/document drift.

## Product entry points

- `sbgpy-inspect inspect SOURCE`
- `sbgpy-inspect source AUDIO`
- `sbgpy-inspect path ...`
- `sbgpy-inspect library ...`
- `sbgpy-inspect-gui`
- existing `sbgpy` rendering now enforces compatibility policy for SBG and DRG imports

## Legal and product boundary

No proprietary I-Doser content was added. DRG tests construct synthetic packages. Compatibility operates on lawfully possessed or legally distributable user files. Creativity implementation remains deferred.

## Pre-PR local qualification

A reconstructed local product path completed:

- compatibility journey tests — **10 passed**;
- package compile check — **passed**;
- desktop inspector compile check — **passed**;
- machine matrix/document drift check — **passed**.

The GitHub PR remains responsible for full repository qualification before merge. This receipt must be updated with the final PR and merge result after that gate completes.
