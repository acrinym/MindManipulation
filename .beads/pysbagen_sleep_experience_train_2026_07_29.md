# PySbagen Sleep Experience Beadtrain

**Date:** July 29, 2026  
**Branch:** `agent/full-product-reliability-train-20260729`  
**Purpose:** Turn the research discussion into the first complete human-facing sleep product without wiring fictional sensors or mixing research controls into ordinary use.

## Product decisions carried into the build

- The ordinary person answers human questions instead of choosing frequencies.
- Sleep Descent and Sleep Support are different phases.
- Sensor-driven closed-loop timing remains documented future work and is not stubbed.
- Pleasant audio is a first-class part of the journey.
- PySbagen supports generated sound and user-provided audio in broadly decodable formats.
- Binaural, monaural, isochronic, and Harmonic Box X-style layers are independently selectable.
- Research doses belong in a separate future environment.
- All three initial sleep complaints receive materially different routes.

## Beads completed

### 1. Human sleep request model

Added `pysbagen.sleep` with:

- racing-mind, threshold-crossing, and repeated-waking problems;
- pleasant sound-world choices;
- gentle, balanced, and immersive strengths;
- recommended layer blends that differ by use case;
- 10–180 minute validated journeys;
- exact recipe and manifest capture.

### 2. Time-changing sleep journey engine

Added `SleepJourneySpec` with:

- separate descent and support envelopes;
- changing beat relationships rather than one static difference;
- continuously accumulated phase across chunks;
- binaural, monaural, smooth isochronic, and Harmonic Box X-style layers;
- strong post-descent reduction of active stimulation;
- gradual bed and final-output fading;
- deterministic generation seeds;
- chunk metadata identifying route, stage, current beat, sound world, and layers.

### 3. Generated pleasant audio

Added four local sound worlds:

- warm evolving ambient chords;
- slow night music with long chord crossfades and sparse melody;
- a stateful soft rain-like room;
- a deep low-stimulation night environment.

### 4. User audio without WAV-only restrictions

Refactored file loading so SoundFile is tried first and FFmpeg becomes a general fallback rather than an MP3-only exception. User audio can be looped underneath the full journey and receives the same descent/support/fade treatment.

### 5. Conversational terminal guide

Added `sbgpy-sleep`.

It asks four ordinary questions, shows the matched route, then saves a reproducible journey or starts immediate playback with `--play`.

### 6. Conversational desktop guide

Added `sbgpy-sleep-gui` as a separate normal-user application rather than burying the experience in the advanced studio.

It provides:

- four guided pages;
- optional custom layer choices;
- immediate headphone playback;
- stop control;
- audio export;
- exact recipe sidecar export;
- clear listening-safety language.

The existing `sbgpy-gui` remains the advanced laboratory.

### 7. Reproducibility

Saved journeys receive `.sleep.json` manifests with exact timing, carrier/beat movement, layers, seed, and source-audio hash.

### 8. Documentation and packaging

- Version advanced to `0.3.0`.
- Added both sleep entry points.
- Added `docs/SLEEP_GUIDE.md`.
- Reframed the README around the human product while preserving advanced SBaGen instructions.
- Extended recognized schedule audio extensions while retaining file-existence detection.

## Intentionally not built

- EEG, watch, movement, heart-rate, breathing, or other sensor adapters;
- fake closed-loop endpoints;
- blinded research assignment;
- consent or study protocol UI inside the ordinary app;
- claims that a frequency guarantees sleep, dopamine, pain relief, migraine relief, or sobriety.

## Qualification

Local reconstructed exact product path:

- `PYTHONPATH=. pytest -q` — **26 passed**;
- `python -m compileall -q pysbagen sleep_gui.py` — **passed**;
- `python -m pip wheel . --no-deps --no-build-isolation` — **built `pysbagen-0.3.0-py3-none-any.whl`**;
- wheel contents inspected — sleep model, generator, terminal guide, playback helper, and desktop guide included.

## Truthful continuation

The next product choices are not sensor plumbing. The useful next trains are:

1. local next-morning feedback and personal preference learning;
2. more generated musical worlds and user-audio transformation controls;
3. the separately launched Research Dose Environment with consent, sham/control conditions, and exact protocol assignment;
4. only after supported hardware exists, a real closed-loop Sleep Support integration.
