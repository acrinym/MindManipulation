# Original SBaGen TODO — PySbagen Product Gap Matrix

**Date:** July 31, 2026  
**Status:** Active source-to-product reconciliation  
**Primary sources:**

- `sbagen-1.4.5/sbagen-1.4.5/TODO.txt`
- `https://uazu.net/sbagen/` — official SBaGen TODO and unmaintained-project notice
- `docs/research/IDOSER_SBAGEN_PAIN_POINT_SCOUT.md`

## Why this matrix exists

The compatibility, preservation, and inspection train completed the six priorities under the scout's **"Priority pain points for the next compatibility train"** section. It did **not** complete the separate workstation/product features listed in the original SBaGen TODO.

Those TODO items were not incidental research notes. Together they describe the missing experimenter's workstation around the synthesis engine: editable authoring, reusable sequencing, independent channels, live control, event capture, richer modulation, sample cues, conversion, and desktop integration.

The previous product-priority record incorrectly allowed "compatibility delivered" to read as though the entire pain-point handoff had been delivered. This matrix corrects that scope.

## State vocabulary

- **delivered** — available through a supported product path and covered by current tests/docs;
- **partial** — some foundation exists, but the original user capability is not complete;
- **missing** — no supported product path currently provides the capability;
- **experimental-lane** — valid idea, but requires explicit safety, hardware, or platform qualification;
- **intentionally-excluded** — understood and explicitly rejected from the modern product, with rationale.

## Reconciliation matrix

| Original SBaGen TODO / website request | Current PySbagen state | Evidence and gap | Product disposition |
|---|---|---|---|
| Easy clickable GUI separate from the command line | **partial** | `sbgpy-gui`, the Sleep Guide, and Compatibility Inspector exist, but the advanced studio is not yet a complete open/edit/validate/play/transport/write workstation for SBG documents. | Build the full experimenter workstation rather than another disconnected GUI. |
| Minimal GUI with open/save, editor, test, play, and write-WAV views | **partial** | Schedule selection and export exist. Raw schedule editing, syntax diagnostics tied to source locations, save/save-as, transport position, and reversible visual editing are incomplete. | Deliver as the workstation document/editor shell. |
| Reusable `sbagenlib` sequencing/audio engine | **delivered** | `pysbagen.api`, parser, mixer, generators, importer, and plugin entry-point foundations provide a reusable Python engine used by multiple front ends. | Preserve and harden; do not rebuild. |
| Several independent channels with independent slides | **missing** | The compatibility matrix classifies `spin:` and `slide:` as approximated; the underlying tone may render, but motion semantics and concurrently sliding channels do not. | Build channel buses, per-channel envelopes, and fixture-proven independent motion. |
| Trigger WAV/MP3 samples or voice cues at sequence points | **missing** | Background files can be layered, but there is no event-triggered one-shot sample/cue model. | Add scheduled one-shot cues, overlap policy, gain, pan, and provenance. |
| Mark interesting combinations during a session without leaving the experience | **missing** | No live marker command or transport-bound session marker exists. | Add keyboard/button markers captured against the exact session clock and recipe identity. |
| Record timing of user events from clicks or keypresses | **missing** | No local event ledger accompanies playback. | Add append-only session event records and privacy-safe export. |
| Random variation within declared frequency ranges | **missing** | `SBG-RANDOM-001` is currently unsupported. Generated sleep worlds may use deterministic variation, but ordinary SBG range semantics do not exist. | Add seeded, bounded, inspectable random/range modulation with exact replay manifests. |
| Global volume control without editing the sequence | **missing** | Individual amplitudes exist, but there is no canonical master gain shared by CLI, GUI, live playback, and render receipts. | Add master gain with clipping forecast and receipt capture. |
| Keyboard control to fade between sequences in real time | **missing** | No scene bank, live crossfade transport, or keyboard mapping exists. | Add safe live scene crossfades using the shared callback/transport engine. |
| General volume modulation in addition to beating | **partial** | Isochronic and sleep envelopes modulate amplitude in specialized generators. There is no general schedule-level modulation/envelope object applicable to arbitrary channels and sources. | Add reusable amplitude modulation/envelope semantics. |
| Sinusoidal, Gaussian, or user-defined sweeps | **missing** | Linear interval crossfades exist, but sweep-shape selection and reusable curves do not. | Add curve-defined parameter automation with serializable shapes. |
| Logarithmic fades | **missing** | Current crossfades are documented as linear. | Add explicit linear, equal-power, logarithmic, and custom-curve fades. |
| Colored noise beyond the original basics | **partial** | White and pink noise are supported. Brown/red, blue, violet, gray, and configurable spectral slopes are not. | Add qualified colored-noise generators with deterministic seeds and spectra tests. |
| Noise amplitude and pitch/filter modulation | **missing** | Noise is not exposed through the same automation system as tones and source buses. | Add bounded modulation through the general automation engine. |
| More organic low-frequency variation in carriers/beats | **partial** | Some guided generators evolve over time, but ordinary studio recipes cannot apply reproducible drift to arbitrary parameters. | Add seeded drift/LFO controls as an authoring feature. |
| Isochronic tones | **delivered** | `IsochronicSpec`, CLI, GUI, and sleep routes already provide isochronic layers. | Keep; extend through the common automation/channel model. |
| `mixspin:` on imported/mixed audio | **missing** | No mix-stream spin implementation exists. | Add a fixture-proven spatial/motion processor with clear headphone/speaker behavior. |
| `mixbeat:` to create binaural shifting from recordings | **missing** | No Hilbert/analytic-signal mixbeat processor exists. | Research, implement, and verify as an advanced processor; never label it equivalent before signal tests pass. |
| Convert to and from Gnaural XML | **missing** | Import reports and manifests exist, but Gnaural XML interchange does not. | Add loss-aware import/export with explicit reversible/approximated fields. |
| Automatic conversion of old SBG into a new visual format | **partial** | SBG parsing and a serializable timeline exist, but there is no editable canonical project document with round-trip visual editing. | Add a project schema that preserves raw source and records every semantic edit. |
| AIFF output in addition to WAV | **partial** | SoundFile may support AIFF depending on runtime, but the public CLI/GUI, tests, estimates, and docs are WAV-centered. | Add explicit capability detection, AIFF output selection, and qualification tests. |
| Mac desktop access / double-click SBG files | **missing** | No packaged desktop file association or launcher integration is recorded. | Add platform launchers/file associations after the workstation command is stable. |
| Callback-model backend and JACK support | **partial** | Modern Python playback exists through PyAudio, but there is no canonical callback transport/backend abstraction or JACK path. | Build backend-neutral transport first; provide JACK where available without coupling core logic to it. |
| Screen flashing synchronized to beats | **experimental-lane** | Not implemented or dispositioned. Flashing visuals carry photosensitive-seizure risk. | Only as disabled-by-default, strongly warned, rate/contrast-limited experimental output with no ordinary-user default. |
| AudioStrobe / light-glasses output | **experimental-lane** | Not implemented; hardware and safety qualification are absent. | Document and isolate behind an explicit hardware research interface before any implementation. |
| LPT1 light-glasses control | **intentionally-excluded** | Obsolete, platform-specific direct hardware control conflicts with a modern cross-platform product and lacks available test hardware. | Do not implement in core; allow future third-party hardware plugins through a documented interface. |

## Correct product conclusion

The compatibility train was valuable and remains complete **within its declared scope**. The full pain-point handoff is not complete until the workstation train addresses the missing and partial product rows above, or explicitly records a justified exclusion.

The active implementation plan is:

- `.beads/pysbagen_experimenter_workstation_train_2026_07_31.md`

Creativity remains parked. This train completes the SBaGen experimenter/workstation foundation before outcome-specific creativity product work begins.
