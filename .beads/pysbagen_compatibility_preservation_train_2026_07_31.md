# PySbagen Compatibility, Preservation, and Inspection Beadtrain

**Date:** July 31, 2026  
**Status:** Queued — active next product train  
**Base:** `main` after `a8ffbea9060ed255b6f0f64f75d600cd6f92c468`  
**Planned implementation branch:** `agent/compatibility-preservation-train-20260731`  
**Product direction:** I-Doser/SBaGen compatibility, preservation, inspection, and ordinary-user reliability

## Train goal

Turn PySbagen's existing SBG and DRG support into an honest, inspectable, local-first compatibility product.

A user who imports a lawfully possessed `.sbg` or `.drg` artifact must be able to learn, before playback or conversion:

- exactly what the file contains;
- what PySbagen preserved;
- what PySbagen approximated;
- what PySbagen cannot represent;
- which external sources are missing;
- whether the audio path is suitable;
- whether playback or rendering can proceed without silently changing the protocol;
- where the imported artifact and its provenance are stored locally.

This is product work, not an audit-only train. The compatibility matrix and fixtures exist to drive implementation and block dishonest success states.

## Product decisions carried into the train

- Creativity implementation remains deferred.
- Existing repaired behavior is not reopened without evidence of a regression.
- Proprietary I-Doser content is not copied, bundled, or redistributed.
- DRG work operates on user-supplied or legally distributable fixtures.
- Unknown or unsupported semantics are never silently discarded.
- Import, inspection, and preservation must work without cloud accounts.
- Raw source artifacts remain immutable; derived files receive new identities.
- Human-readable reports and machine-readable manifests are both required.
- Playback is downstream of inspection. A misleading import must not be called successful merely because it produces sound.
- The advanced laboratory may expose technical detail, but ordinary users receive clear explanations and actionable compatibility states.

## Shared compatibility vocabulary

Every imported construct and artifact receives one explicit state:

- **supported** — preserved and executed with intended semantics;
- **equivalent** — represented differently but expected to behave equivalently;
- **partial** — only part of the original behavior is preserved;
- **approximated** — substituted behavior is used and disclosed;
- **unsupported** — recognized but not executable;
- **unknown** — not understood well enough to classify;
- **missing-source** — references an unavailable external asset;
- **rendered-only** — playable audio exists without an editable recipe;
- **unsafe-to-render** — proceeding would materially misrepresent the source;
- **intentionally-excluded** — understood but outside supported product scope.

No parser, importer, converter, GUI, or CLI may reduce these states to a single optimistic success flag.

---

# Beads

## Bead CP-001 — Establish the canonical import contract

**Status:** queued  
**Priority source:** honest SBG/DRG import report  
**Depends on:** none

Create one canonical import-result model used by SBG, DRG, CLI, GUI, tests, and the future local library.

The contract must record:

- source path, source type, byte size, and cryptographic hash;
- detected encoding and version clues;
- immutable source identity;
- metadata discovered;
- package elements discovered;
- parsed constructs and source locations;
- preserved, equivalent, partial, approximated, unsupported, unknown, and excluded features;
- missing referenced audio or other package dependencies;
- inferred start mode, end behavior, duration, and loop behavior;
- warnings and blocking errors;
- a final render disposition: safe, safe-with-disclosed-changes, inspection-only, or blocked;
- importer and schema versions.

### Acceptance

- SBG and DRG importers return the same top-level result type.
- The result serializes deterministically to JSON.
- Human-readable text is rendered from the structured result rather than maintained as a second truth.
- Tests prove that warnings, blockers, and unsupported constructs survive serialization.

---

## Bead CP-002 — Deliver the honest SBG import report

**Status:** queued  
**Priority source:** honest SBG/DRG import report  
**Depends on:** CP-001

Add a non-destructive SBG inspection path that parses and reports without immediately rendering.

The report must explain:

- tone-set declarations;
- schedule events and transitions;
- wall-clock versus relative timing;
- explicit and inferred duration;
- silence, continuation, fades, and end behavior;
- file references and their resolved locations;
- missing files;
- options that affect playback or export;
- every construct that is not fully supported.

### User paths

- CLI inspection with human output;
- CLI JSON output for tooling;
- application API entry point;
- no playback side effect.

### Acceptance

- A valid supported SBG reports safe-to-render.
- A missing source reports `missing-source` and cannot masquerade as a complete import.
- A recognized unsupported construct appears in both text and JSON reports.
- Wall-clock schedules clearly state why immediate playback could begin in silence.

---

## Bead CP-003 — Preserve the complete DRG package

**Status:** queued  
**Priority source:** preserve DRG metadata and package elements  
**Depends on:** CP-001

Replace the current schedule-and-image-only return path with a structured DRG package model.

Preserve, when present:

- original raw package bytes;
- decoded package elements in original order;
- encrypted and decrypted identities where lawful and technically available;
- text metadata and labels;
- embedded image bytes plus detected media type;
- decrypted SBG source;
- unknown elements without inventing meanings;
- decoding warnings, offsets, and hashes;
- provenance linking every extracted artifact to the original DRG hash.

The current decoder's positional assumptions must become explicit, validated, and reportable rather than hidden in tuple return values.

### Acceptance

- No decoded element is silently dropped.
- Unknown elements are preserved as opaque elements.
- A malformed package produces a partial structured report rather than console-only warnings.
- Extracted schedule and image files receive deterministic provenance manifests.
- Tests use synthetic or legally distributable fixtures only.

---

## Bead CP-004 — Deliver the honest DRG import report

**Status:** queued  
**Priority source:** honest SBG/DRG import report; DRG preservation  
**Depends on:** CP-002, CP-003

Compose package inspection and nested SBG inspection into one report.

The report must distinguish:

- DRG package parsing success;
- element decoding success;
- nested SBG decoding success;
- nested SBG semantic compatibility;
- embedded image preservation;
- opaque or unknown package elements;
- whether the artifact can be inspected, extracted, rendered, or only preserved.

### Acceptance

- A DRG with a valid package but unsupported nested schedule is not labeled fully compatible.
- A recoverable image failure does not erase the successfully preserved schedule or other elements.
- A schedule failure does not erase preserved raw package evidence.
- Text and JSON reports provide the same compatibility disposition.

---

## Bead CP-005 — Build the original-SBaGen semantic compatibility matrix

**Status:** queued  
**Priority source:** complete original-SBaGen semantic compatibility matrix  
**Depends on:** CP-001

Create a versioned compatibility matrix covering the original language and runtime semantics rather than only parser tokens already encountered.

The matrix must include, at minimum:

- tone-set definitions and references;
- carriers, beat relationships, amplitudes, and channel behavior;
- multiple simultaneous generators;
- silence and source removal;
- absolute/wall-clock and relative event timing;
- abrupt events, continuation, interpolation, and trailing transition forms;
- finite, open-ended, and looping schedules;
- referenced audio files and looping behavior;
- quoted paths and schedule-relative paths;
- command/options lines affecting duration or output;
- noise generators and modulation forms documented by original SBaGen;
- unsupported historical device/backend controls;
- malformed and ambiguous syntax behavior.

For each construct, record:

- source documentation or fixture provenance;
- parser support;
- execution support;
- render support;
- round-trip support;
- compatibility state;
- known deviations;
- test fixture IDs.

### Deliverables

- `docs/compatibility/SBAGEN_SEMANTIC_COMPATIBILITY_MATRIX.md`;
- machine-readable matrix data used by tests and import reporting;
- fixture manifest with licensing/provenance fields.

### Acceptance

- Every matrix row has an explicit state; no blank implied support.
- Reported support is backed by at least one fixture test.
- Documentation and machine-readable data are checked for drift.

---

## Bead CP-006 — Close fixture-proven semantic gaps

**Status:** queued  
**Priority source:** complete semantic compatibility matrix  
**Depends on:** CP-002, CP-005

Implement missing SBG semantics in dependency order, driven by fixtures from the matrix.

Rules:

- Fix real parser/runtime gaps rather than adding speculative syntax.
- Preserve source locations so reports can identify the exact unsupported construct.
- Do not weaken existing silence, crossfade, streaming, atomic-output, or duration guarantees.
- When exact support is not practical, classify the construct honestly and block misleading conversion.

### Acceptance

- Each newly supported construct moves from unsupported/partial to supported/equivalent only with tests.
- Existing schedule and sleep-product tests remain green.
- Round-trip tests prove what source form is preserved and what formatting may change.
- Unknown constructs remain visible in the import report.

---

## Bead CP-007 — Make unsupported-feature handling impossible to miss

**Status:** queued  
**Priority source:** explicit compatibility states; unsupported-feature handling  
**Depends on:** CP-002, CP-004, CP-005

Add one policy layer that decides what inspection, playback, export, conversion, and library actions are allowed for each compatibility state.

Required behavior:

- supported/equivalent: proceed normally;
- partial/approximated: require visible disclosure and preserve the report with the output;
- unsupported/unknown: block semantic conversion by default;
- missing-source: allow preservation and inspection, but block complete playback/render claims;
- rendered-only: allow playback while clearly denying editability or recipe recovery;
- unsafe-to-render: block rendering unless a future explicit expert override is designed and documented.

### Acceptance

- CLI exits use stable documented codes.
- GUI controls cannot bypass the same policy.
- Rendered outputs receive sidecars naming every approximation.
- No importer prints a warning and then returns an indistinguishable success object.

---

## Bead CP-008 — Build the timeline and source inspector model

**Status:** queued  
**Priority source:** timeline/source inspector before playback  
**Depends on:** CP-002, CP-005, CP-007

Create an inspection model independent of any one GUI toolkit.

It must expose:

- chronological events and transitions;
- currently active tone sets and source files over any selected interval;
- carrier, beat, amplitude, channel, and generator type;
- source file start, loop, stop, and missing state;
- silence and inactive spans;
- wall-clock mapping versus play-from-start mapping;
- inferred duration and open-ended sections;
- warnings attached to exact events or sources;
- a compact rapid-preview timeline that does not pretend to be actual playback.

### Acceptance

- The model is produced from canonical parsed/import data, not by reparsing display strings.
- Tests cover simultaneous layers, silence, transitions, missing audio, and open-ended schedules.
- The model can serialize for CLI and future UI use.

---

## Bead CP-009 — Put inspection before playback in the user interfaces

**Status:** queued  
**Priority source:** timeline/source inspector before playback  
**Depends on:** CP-008

Add human-facing inspection to the advanced studio without burying compatibility failures in logs.

The interface must show:

- source identity and provenance;
- overall compatibility disposition;
- timeline overview;
- source/layer details;
- missing and unsupported items;
- play-from-start versus honor-wall-clock choice where relevant;
- why playback or rendering is blocked;
- export of the import report.

Ordinary user flows may use simplified language, but must not hide material changes.

### Acceptance

- Playback/export controls reflect the canonical policy from CP-007.
- The interface remains usable when optional plotting dependencies are absent, with a textual inspector fallback.
- Inspection itself does not require PyAudio or a live output device.

---

## Bead CP-010 — Qualify imported audio sources

**Status:** queued  
**Priority source:** qualify source suitability  
**Depends on:** CP-001, CP-002

Analyze every referenced or user-provided audio source before it is mixed.

Record:

- container and codec;
- sample rate and sample format;
- channel count and channel layout clues;
- duration;
- peak level, clipping, and available headroom;
- mono, near-mono, or strongly correlated stereo detection;
- DC offset or obvious decode anomalies where practical;
- resampling and channel conversion that PySbagen will apply;
- whether the source is suitable for binaural layering, general ambience, or rendered-only playback.

### Acceptance

- Analysis is local and bounded-memory for long sources.
- Missing optional decoders produce actionable reports.
- Near-mono content is disclosed rather than automatically rejected when it is still useful as ambience.
- Qualification results become part of import reports and output sidecars.

---

## Bead CP-011 — Qualify the listening and rendering path

**Status:** queued  
**Priority source:** qualify stereo routing, sample rate, processing, and source suitability  
**Depends on:** CP-007, CP-010

Add a practical preflight for the path between generated audio and the listener.

Cover what PySbagen can directly test or truthfully explain:

- selected output device and supported channel count;
- requested versus negotiated sample rate;
- channel routing and left/right test playback;
- mono-downmix risk;
- Bluetooth, spatial enhancement, loudness normalization, equalization, and other external processing as disclosed user checks when direct detection is unavailable;
- headphones versus speakers suitability for the selected entrainment method;
- export-only qualification when no live device is available.

### Acceptance

- Direct measurements are separated from user-confirmed checks and explanatory warnings.
- The product never claims it detected operating-system processing it cannot observe.
- A stereo routing test is optional, brief, and stoppable.
- Preflight results can be saved with a session or render manifest.

---

## Bead CP-012 — Build the local-first provenance library

**Status:** queued  
**Priority source:** local-first library with provenance and explicit compatibility states  
**Depends on:** CP-001, CP-003, CP-007, CP-010

Create a local library that stores records and derived artifacts without requiring accounts or cloud services.

The library must distinguish:

- SBG recipes;
- DRG packages;
- extracted package elements;
- rendered WAV/other audio;
- user-provided source audio;
- missing external sources;
- platform or catalog variants sharing a display name;
- archived, superseded, withdrawn, incompatible, and research-only items.

Each entry must record:

- immutable content hash and source provenance;
- original filename and import time;
- artifact kind;
- compatibility state and latest report;
- relationships among package, recipe, source audio, and rendered output;
- whether the item is editable, playable, renderable, or preservation-only;
- user labels without replacing canonical identity;
- backup/export manifest.

### Acceptance

- Duplicate content is detected by hash without erasing distinct provenance records.
- Renaming a display label does not change protocol identity.
- Source files are not silently moved or deleted.
- The library can export a self-describing manifest and verify stored hashes.
- Missing sources remain visible rather than causing entries to disappear.

---

## Bead CP-013 — Prove the complete compatibility journeys

**Status:** queued  
**Priority source:** all six priorities  
**Depends on:** CP-001 through CP-012

Qualify complete user journeys rather than isolated units.

Required journeys:

1. Inspect a fully supported SBG, view its timeline, qualify sources, and render with matching sidecars.
2. Inspect an SBG with wall-clock timing and deliberately choose play-from-start.
3. Import an SBG with a missing audio source and prove that preservation works while complete rendering is blocked.
4. Import a DRG, preserve every package element, inspect its nested SBG, and export a provenance bundle.
5. Import a DRG whose nested SBG contains an unsupported construct and prove it is not labeled fully compatible.
6. Import two same-named but non-identical artifacts and preserve distinct hashes and platform provenance.
7. Reopen library entries offline and reproduce their reports without contacting a server.
8. Verify an exported manifest and detect changed or missing artifacts.

### Qualification gates

- Full local test suite passes.
- Compile checks pass for the canonical package and desktop entry points.
- A wheel builds and contains compatibility schemas, matrix data, and required documentation.
- Fixture licensing/provenance is documented.
- No proprietary I-Doser dose is committed.
- Existing sleep and advanced-generation paths remain operational.
- Documentation states exact supported, partial, and unsupported behavior.

---

# Train completion definition

This train is complete only when all of the following are true:

- SBG and DRG imports produce one honest canonical report.
- DRG package elements are preserved instead of reduced to a schedule/image tuple.
- The original-SBaGen semantic matrix is versioned, fixture-backed, and synchronized with tests.
- Unsupported or unknown behavior cannot silently flow into a successful conversion.
- Users can inspect timeline and active sources before playback.
- Source audio and listening-path qualifications clearly separate measured facts from user checks.
- The local library preserves identity, provenance, compatibility, and artifact relationships offline.
- Complete end-to-end journeys are qualified.

## Explicitly deferred beyond this train

- creativity implementation;
- sensor integrations or fake sensor endpoints;
- cloud accounts, cloud synchronization, or vendor restoration systems;
- redistribution of proprietary I-Doser files;
- claims that imported protocols reliably cause named mental or medical outcomes;
- personal outcome-history and adaptive recommendation learning, which should become a separate train after exact protocol identity and provenance exist;
- speculative support for undocumented formats without lawful fixtures.

## Recommended execution order

Run beads in numerical order, with these parallel opportunities only after their dependencies are complete:

- CP-003 and CP-005 may proceed after CP-001;
- CP-010 may proceed after CP-002 while semantic work continues;
- CP-008 may proceed while CP-006 closes remaining fixture-proven gaps;
- CP-011 may proceed after qualification data and policy exist;
- CP-012 begins only after identity, preservation, and policy contracts are stable.

Do not skip directly to the library or GUI. Without the import contract, preservation model, semantic matrix, and policy layer, those surfaces would fossilize misleading compatibility behavior.
