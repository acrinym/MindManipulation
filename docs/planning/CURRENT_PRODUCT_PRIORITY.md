# PySbagen Current Product Priority

**Date:** July 31, 2026  
**Status:** Compatibility phase delivered; original-SBaGen workstation completion is the active next priority

## Scope correction

The I-Doser/SBaGen compatibility, preservation, inspection, and local-library train was successfully implemented and merged through:

- `.beads/pysbagen_compatibility_preservation_train_2026_07_31.md`
- PR `#9` — **Build the SBaGen and DRG compatibility product**
- merge commit `0c95a67ca65db22d6441b123a5709bcaf929a064`
- completion receipt `.beads/pysbagen_compatibility_preservation_train_2026_07_31_COMPLETION.md`

That train completed the six priorities explicitly listed under the pain-point scout's **next compatibility train** section.

It did **not** complete the separate workstation/product features recorded by the original SBaGen website and bundled TODO. The previous wording "compatibility priority delivered" was accurate only for that bounded phase, but it could be read as though the entire pain-point handoff had been delivered. This document corrects that interpretation.

## Delivered compatibility foundation

1. Honest SBG/DRG import reports and render dispositions.
2. Complete DRG package preservation.
3. Original-SBaGen semantic compatibility matrix.
4. Timeline/source inspection before playback.
5. Audio-source and listening-path qualification.
6. Local-first provenance library.

These remain delivered and are the foundation for the next train.

## Active next priority: experimenter workstation

The original SBaGen TODO gap is now mapped in:

- `docs/planning/SBAGEN_ORIGINAL_TODO_GAP_MATRIX.md`

The implementation plan is:

- `.beads/pysbagen_experimenter_workstation_train_2026_07_31.md`

Planned implementation branch:

- `agent/experimenter-workstation-train-20260731`

The active train covers the missing or partial product layer, including:

- a full editable schedule/project workstation;
- independent channels and concurrent slides;
- one-shot WAV/MP3/voice triggers;
- reproducible random ranges and organic variation;
- master volume and backend-neutral live transport;
- keyboard scene crossfades;
- session markers and user-event timing records;
- generalized modulation, sweep curves, logarithmic fades, and broader colored noise;
- `mixspin` and qualified experimental `mixbeat` processing;
- Gnaural XML interchange and explicit WAV/AIFF output;
- desktop file association and launch behavior;
- explicit safety/hardware disposition for flashing, AudioStrobe, light-glasses, and obsolete LPT1 control.

## Creativity status

The creativity research and product gap remain valid and preserved in:

- `docs/research/CREATIVITY_AUDIO_RESEARCH_FOUNDATIONS.md`
- `docs/planning/CREATIVITY_PRODUCT_GAP_CHECK.md`

However, **creativity implementation remains deferred**. The experimenter workstation train takes precedence and does not silently authorize the Creative Cycle train.

## Completion rule

The full pain-point handoff must not be described as complete until every row in `SBAGEN_ORIGINAL_TODO_GAP_MATRIX.md` is either:

- delivered through a supported, tested product path;
- retained as an explicit experimental lane with appropriate safeguards; or
- intentionally excluded with a documented rationale.
