# PySbagen Experimenter Workstation and Original-TODO Completion Beadtrain

**Date:** July 31, 2026  
**Status:** Queued — active next product train  
**Base:** `main` after the compatibility/preservation merge and scope correction  
**Planned implementation branch:** `agent/experimenter-workstation-train-20260731`  
**Source matrix:** `docs/planning/SBAGEN_ORIGINAL_TODO_GAP_MATRIX.md`

## Train goal

Complete the product layer that the original SBaGen website and bundled TODO described but never delivered: a reusable, editable, live-controllable experimenter workstation around the audio engine.

This train is not another compatibility audit. Its fixtures and matrices exist only to make real authoring, playback, live control, capture, conversion, and rendering dependable.

At completion, a user must be able to:

- open an SBG/DRG/project document in one workstation;
- understand and edit the sequence as text or visually without losing source provenance;
- run independent channels and parameter slides concurrently;
- schedule one-shot samples or voice cues;
- use reproducible random ranges, sweeps, fades, modulation, and colored noise;
- control master volume and crossfade among scenes while playing;
- mark interesting moments and export session-event timing records;
- convert supported semantics to and from Gnaural XML and explicit audio formats;
- use a backend-neutral transport that supports export, ordinary playback, and optional JACK;
- see exact receipts for every live or rendered alteration.

Creativity implementation remains deferred. The workstation must become dependable before adding a Creativity Cycle or other outcome-specific product.

## Boundaries

- Do not copy or redistribute proprietary I-Doser content.
- Do not silently reinterpret unsupported legacy constructs.
- Do not build fake hardware adapters or endpoints.
- Flashing/light-glasses features remain isolated experimental lanes with explicit safety gates.
- LPT1 control is intentionally excluded from core.
- Existing compatibility, sleep, streaming, atomic-write, and provenance behavior must remain intact.
- No stubs, dead controls, or menu items that claim unavailable functionality.
- No recursive audit train: every bead must deliver a user-visible capability or a necessary shared engine component.

---

# Beads

## WST-001 — Canonical editable project document

**Status:** queued  
**Depends on:** existing import/timeline models

Create a versioned project schema that can contain:

- immutable imported SBG/DRG source identity;
- editable tone sets, channels, events, automation curves, samples, scenes, and metadata;
- raw-source preservation plus semantic edits;
- source-to-project and project-to-render provenance;
- deterministic JSON serialization;
- migration hooks for future schema versions.

**Acceptance:** Opening and saving an untouched supported SBG preserves semantic identity; edits produce a new project identity without overwriting the original source.

## WST-002 — Full schedule editor workstation shell

**Status:** queued  
**Depends on:** WST-001

Turn the advanced GUI into one coherent workstation with:

- Open, Save, Save As, Import, Export, and recent-local-file actions;
- raw SBG editor with syntax highlighting and source-located diagnostics;
- visual timeline/channel editor generated from the same project model;
- inspector, validation, transport, and render panels;
- reversible text/visual edits where semantics are representable;
- explicit conflict handling when raw text and visual edits diverge.

**Acceptance:** A user can open, edit, validate, preview, save, reopen, and render a project without switching among disconnected applications.

## WST-003 — Backend-neutral transport and master bus

**Status:** queued  
**Depends on:** WST-001

Build one callback-oriented transport used by GUI and API for:

- play, pause, resume, stop, seek, loop-region, and current position;
- master gain independent of source recipes;
- clipping/headroom forecast and live meter data;
- clean cancellation and device cleanup;
- offline render using the same event clock;
- optional backend adapters, beginning with existing playback and then JACK where available.

**Acceptance:** Live playback and offline rendering share event timing and receipt semantics; master gain is captured in manifests and never requires editing the source schedule.

## WST-004 — Independent channels and concurrent automation

**Status:** queued  
**Depends on:** WST-001, WST-003

Add explicit channel buses with:

- independent sources and tone sets;
- gain, mute, solo, pan/motion, and routing;
- automation that continues while unrelated channel events occur;
- independent slides and overlapping transitions;
- stable generator state across event boundaries.

Implement original `slide:` and `spin:` semantics from fixtures where understood; retain visible approximation states where exact historical behavior remains uncertain.

**Acceptance:** At least three channels can run concurrent, independently changing parameters without one channel resetting another.

## WST-005 — General automation curves and organic variation

**Status:** queued  
**Depends on:** WST-004

Add serializable automation for:

- linear, equal-power, logarithmic, sinusoidal, Gaussian, and user-defined curves;
- carrier, beat, amplitude, pan/motion, filter, and supported generator parameters;
- bounded low-frequency drift/LFO;
- seeded random ranges with exact replay;
- preview of min/max and rate before playback.

**Acceptance:** Random and organic variation is reproducible from the saved seed and cannot exceed declared bounds.

## WST-006 — One-shot sample and voice-cue triggers

**Status:** queued  
**Depends on:** WST-003, WST-004

Add timeline-triggered WAV/AIFF/FLAC/OGG/MP3 cues with:

- exact trigger time or event-relative placement;
- gain, pan, optional ducking, overlap, and retrigger policy;
- missing-source and decode diagnostics;
- one-shot behavior distinct from looping background beds;
- source hashes and licensing/provenance fields.

**Acceptance:** A voice cue can fire once at a declared sequence point without truncating or resetting continuous tone channels.

## WST-007 — Live scenes and keyboard crossfades

**Status:** queued  
**Depends on:** WST-003, WST-005

Add scene banks and user-configurable keyboard controls for:

- selecting or launching scenes;
- real-time crossfading between complete sequences or scene groups;
- configurable fade curve and duration;
- panic/stop and master mute;
- visible current and target scene;
- deterministic event logging.

**Acceptance:** Scene changes do not block the UI, click, leak generators, or lose the event record.

## WST-008 — Session markers and user-event ledger

**Status:** queued  
**Depends on:** WST-003

Add live marker capture from GUI buttons and keyboard shortcuts:

- named and quick unnamed markers;
- transport-relative and wall-clock timestamps;
- active scene/channel/parameter snapshot;
- optional note entered after the session rather than interrupting playback;
- append-only local event ledger tied to exact recipe and output identities;
- privacy-safe JSON/CSV export.

**Acceptance:** A user can mark an interesting moment without stopping playback and later recover the exact active configuration.

## WST-009 — Colored noise and generalized modulation

**Status:** queued  
**Depends on:** WST-005

Expand noise and modulation support:

- white, pink, brown/red, blue, violet, gray, and configurable spectral slope;
- deterministic seed and continuous filter state;
- amplitude and supported filter/pitch modulation;
- spectrum/headroom qualification tests;
- no chunk-boundary discontinuities.

**Acceptance:** Noise color and modulation survive streaming, seeking where supported, and deterministic rendering.

## WST-010 — Mix-stream processors: `mixspin` and `mixbeat`

**Status:** queued  
**Depends on:** WST-004, WST-005

Implement advanced source-bus processors:

- `mixspin` spatial/motion processing applied to an imported or mixed stream;
- `mixbeat` analytic-signal/Hilbert-based frequency shifting only after signal-level verification;
- bypass and wet/dry controls;
- latency and edge-effect reporting;
- headphone/speaker suitability guidance;
- explicit experimental status until reference fixtures demonstrate intended behavior.

**Acceptance:** Processor tests verify channel separation, target shift/motion, bounded amplitude, and deterministic output; unsupported source conditions fail visibly.

## WST-011 — Loss-aware interchange and audio formats

**Status:** queued  
**Depends on:** WST-001, WST-005, WST-006

Add:

- Gnaural XML import and export;
- field-by-field reversible/equivalent/approximated/lost conversion reports;
- explicit WAV and AIFF output selection with runtime capability detection;
- project-manifest export independent of rendered audio;
- no claim of round-trip losslessness unless fixture-proven.

**Acceptance:** A supported Gnaural fixture round-trips semantically; unsupported fields remain disclosed rather than dropped.

## WST-012 — Desktop launch and file association

**Status:** queued  
**Depends on:** WST-002

Provide platform launch behavior for supported packaging paths:

- open `.sbg`, `.drg`, and project documents in the workstation;
- safe argument handling and import inspection before playback;
- Windows and macOS launcher/file-association documentation and build metadata;
- no playback merely from double-click without inspection/explicit action.

**Acceptance:** Packaged launchers open the document in a stable editor state and never silently render or play it.

## WST-013 — Experimental visual/hardware synchronization boundary

**Status:** queued  
**Depends on:** WST-003

Create the boundary—not fake hardware implementations—for synchronized outputs:

- plugin interface receiving bounded session clock/beat events;
- disabled-by-default screen-pulse experiment with photosensitive-seizure warning, conservative rate/contrast limits, and immediate stop;
- documented future AudioStrobe/light-glasses adapter contract;
- explicit exclusion of direct LPT1 control from core.

**Acceptance:** Ordinary installations expose no flashing default; experimental activation is explicit and independently stoppable.

## WST-014 — Workstation journey tests and performance qualification

**Status:** queued  
**Depends on:** WST-001 through WST-013

Prove complete journeys:

1. import SBG → edit text → validate → visual timeline → save project → reopen → render;
2. three independent channels with overlapping slides and one-shot voice cue;
3. seeded random automation → repeat render → identical hash;
4. live scene crossfade → marker capture → event-ledger export;
5. colored-noise modulation without chunk discontinuities;
6. Gnaural import/export report;
7. master gain and AIFF/WAV receipts;
8. safe cancellation and device cleanup;
9. existing sleep and compatibility journeys remain green.

Measure bounded memory and UI responsiveness on long sessions rather than buffering full output.

## WST-015 — Documentation, migration, and completion receipt

**Status:** queued  
**Depends on:** WST-014

Update README, workstation guide, CLI help, project schema, compatibility matrix, and original-TODO gap matrix. Record every row as delivered, partial with reason, experimental-lane, or intentionally excluded.

The completion receipt must not say "original TODO complete" while any unrecorded missing row remains.

---

# Required execution order

1. WST-001 → WST-003 establish project and transport foundations.
2. WST-004 → WST-009 deliver authoring, channel, trigger, live-control, capture, and modulation capabilities.
3. WST-010 is advanced signal processing and must remain visibly experimental until qualified.
4. WST-011 → WST-013 complete interchange, desktop integration, and explicit experimental boundaries.
5. WST-014 → WST-015 qualify and close the train.

## Definition of done

This train is complete only when:

- the workstation performs open/edit/validate/play/control/mark/save/render journeys;
- original TODO rows are reconciled one by one;
- missing features are delivered or explicitly dispositioned with a defensible reason;
- compatibility reports and provenance remain intact;
- no menu item, CLI flag, or document claims functionality that is absent;
- local tests and package builds pass before merge;
- creativity remains deferred unless separately authorized.
