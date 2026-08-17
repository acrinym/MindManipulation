# SBaGenX Differentiation and Interoperability Matrix

**Date:** July 31, 2026  
**Status:** Active product boundary  
**SBaGenX source reviewed:** `lm7137/SBaGenX` at `be7c74d378774a759f1e4149dc9df3617b4d0d3b`  
**PySbagen source:** `acrinym/MindManipulation`

## Decision

PySbagen will not rebuild SBaGenX in Python.

SBaGenX is the modern native continuation of the SBaGen engine and already owns substantial authoring, sequencing, DSP, curve, mix-processing, live-control, export, plotting, packaging, and frontend work. PySbagen's strongest product is a different layer: compatibility truth, DRG preservation, provenance, guided experiences, local session intelligence, personalization, and research workflows.

The intended relationship is:

> **SBaGenX performs advanced SBaGen-native synthesis and authoring.**  
> **PySbagen inspects, preserves, guides, records, personalizes, and qualifies protocols and sessions.**

SBaGenX remains optional. PySbagen's existing Python renderer stays available as a portable fallback and for current guided products.

## State vocabulary

- **delegate** — SBaGenX already owns the stronger implementation; do not recreate it.
- **interoperate** — PySbagen should inspect, invoke, wrap, or preserve the SBaGenX capability with versioned receipts.
- **shared/fallback** — PySbagen retains its current implementation, but SBaGenX may become an optional backend.
- **PySbagen-owned** — a distinct product capability not found as a central SBaGenX feature.
- **verify-later** — source evidence is incomplete; do not claim absence or duplicate it yet.

## Capability matrix

| Capability | SBaGenX observed state | PySbagen decision |
|---|---|---|
| Native SBaGen parsing, timing, sequencing, and DSP | Library-first `sbagenxlib` owns core behavior. | **delegate/interoperate** — bind to the native API rather than reproduce advanced semantics. |
| `.sbg` editor and diagnostics | Tauri/Svelte/Monaco GUI with multi-document editing and native validation. | **delegate** — PySbagen may link/open in SBaGenX, not build a competing editor. |
| `.sbgf` function and curve language | Native curve program loading, solving, preparation, sampling, and GUI editing. | **interoperate** — preserve and inspect `.sbgf`; do not invent a competing curve/project language first. |
| Built-in `drop`, `slide`, `sigmoid`, and `curve` programs | Implemented through the native runtime. | **delegate/interoperate**. |
| Live carrier, beat, amplitude, and mix ramps | Native live-control API and GUI controls exist. | **delegate/interoperate** — attach session receipts and event history around them. |
| `mixspin`, `mixbeat`, `mixpulse`, and `mixam` | Native mix-effect parsing and processing exist. | **delegate** — remove independent reimplementation beads. |
| Multiple voices and auxiliary tones | Native context API exposes multiple voice lanes and auxiliary overlays. | **delegate/interoperate**; qualify exact behavior before claiming parity with every historical construct. |
| Isochronic, monaural, binaural, spin, bell, orbitbeat | Native engine support exists. | **shared/fallback** — retain Python support where present; prefer native backend only after parity gates. |
| White, pink, brown, custom-spectrum noise | Native support exists; PySbagen currently has white/pink. | **delegate/interoperate** for advanced native noise; retain Python fallback. |
| Raw/WAV/OGG/FLAC/MP3 export and PCM conversion | Native writer API exists. | **interoperate** — record selected backend, API version, format, and output hash. |
| Plot, beat, envelope, and graph-video inspection | SBaGenX exposes sampling and frontend plotting paths. | **interoperate** where useful; PySbagen retains its compatibility timeline and source inspector. |
| Desktop and Android frontends | Desktop GUI and separate Android frontend use the same engine. | **delegate** — do not make frontend duplication a product priority. |
| DRG package decoding and complete element preservation | No central SBaGenX DRG product path found in the reviewed repository. | **PySbagen-owned**. |
| Immutable source/package hashes and provenance chain | Not observed as a central SBaGenX product contract. | **PySbagen-owned**. |
| Explicit supported/equivalent/partial/approximated/blocked import states | Not observed as the same product-wide compatibility contract. | **PySbagen-owned**. |
| Local content-addressed protocol library | Not observed as a central SBaGenX feature. | **PySbagen-owned**. |
| Listening-route and source suitability qualification | Not observed as a central SBaGenX workflow. | **PySbagen-owned**. |
| Guided human-question products such as Sleep Guide | SBaGenX ships general-purpose programs/examples, not PySbagen's guided request model. | **PySbagen-owned**. |
| Session markers and append-only user-event ledger | No completed equivalent found during the reviewed source pass. | **PySbagen-owned**, while remaining open to native telemetry integration. |
| Outcome history and local preference learning | No central equivalent found. | **PySbagen-owned**. |
| One-shot voice/sample cues tied to protocol events | Not confirmed as a completed native capability. | **verify-later**, then build as PySbagen session orchestration only if still absent. |
| Scene banks and complete-sequence crossfades | Native live parameter ramps exist; complete scene-bank behavior was not confirmed. | **verify-later**; prefer orchestration above native contexts over another DSP engine. |
| Research consent, sham/control assignment, adverse-event capture | Not a central SBaGenX product responsibility. | **PySbagen-owned** future Research Dose Environment. |
| Evidence-position labels and claim provenance | Not observed as a central SBaGenX feature. | **PySbagen-owned**. |
| Gnaural interchange | Not confirmed in the reviewed source pass. | **verify-later**; any converter must be loss-aware and protocol-manifest driven. |
| JACK/device backend evolution | SBaGenX's library intentionally leaves live device I/O to hosts. | **do not duplicate prematurely**; backend integration follows real user/platform need. |

## Architecture boundary

### SBaGenX adapter layer

The optional adapter may:

- discover the CLI and native library;
- read `sbx_version()` and `sbx_api_version()`;
- capability-gate on actual exported symbols;
- validate `.sbg` and `.sbgf` through the native library;
- render through the native library only after fixture parity and receipt support;
- expose native telemetry/live controls to PySbagen session records;
- fail closed when the installed API is unknown or lacks required symbols.

The adapter must not:

- make SBaGenX mandatory for inspection, DRG preservation, or existing guided products;
- shell out silently when a native library path was expected;
- call an installed backend without recording exact backend identity;
- describe cross-engine output as identical unless fixture-proven;
- redistribute SBaGenX binaries without satisfying its license and dependency obligations.

### PySbagen product layer

PySbagen continues to own:

- SBG/DRG compatibility reports and loss disclosure;
- complete DRG preservation;
- source, package, recipe, session, and output provenance;
- local-first library and lifecycle states;
- listening-path and source qualification;
- guided experiences and exact recipe capture;
- session markers, event ledgers, outcome history, and local preference learning;
- consent-aware research workflows;
- evidence and claim-position records.

## Immediate implementation result

The first interoperability foundation is `pysbagen.sbagenx_backend` and:

```bash
sbgpy-inspect backend
sbgpy-inspect backend --json
sbgpy-inspect backend --discover-only
```

It discovers `SBAGENX_BIN`, `SBAGENXLIB_PATH`, system candidates, native version/API, and symbol-backed capabilities. Discovery does not yet authorize native rendering.

## Superseded plan

The broad workstation recreation plan is archived as superseded:

- `.beads/pysbagen_experimenter_workstation_train_2026_07_31.md`

The active plan is:

- `.beads/pysbagen_sbagenx_interoperability_train_2026_07_31.md`
