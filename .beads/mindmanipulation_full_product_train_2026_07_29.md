# MindManipulation / pysbagen — Full Product Reliability Train

**Date:** 2026-07-29
**Branch:** `agent/full-product-reliability-train-20260729`
**Base:** `main` at `f44025323a438dda7ff6c247fb679b46d2824cea`

## Train goal

Turn the existing Python SBaGen work into one dependable product path that can be installed, used from the command line, used from the GUI, tested, and extended without choosing between duplicate implementations.

This train is product work, not a recursive audit. The smell/bug check below exists only to identify what must be repaired now.

## Smell and bug check

### Product-path failures found

1. The public README contained only a title and one sentence, leaving installation, CLI, GUI, schedules, and supported generators undocumented.
2. The GUI changed `sys.argv` to impersonate the CLI instead of calling a real application API.
3. The GUI referenced matplotlib `Figure` and `FigureCanvasTkAgg` without importing them.
4. The GUI required PyAudio at import time even for users who only wanted file export.
5. The visualization path generated a generic sine tone while waiting for `isochronic` metadata, so its update condition could never become true.
6. The soundscape path constructed `FileSpec` positionally, assigning values to inherited dataclass fields rather than `path` and `amp`, and then called `generator(..., loop=True)` even though that argument did not exist.
7. `mix_generators` stopped the entire mix when any one input ended. A short background file could truncate a longer tone session.
8. Empty scheduled spans emitted no audio, collapsing intentional silence and shifting all later events earlier.
9. Explicit schedule durations shorter than the schedule could still process later events instead of ending cleanly.
10. Unknown tone-set names were silently ignored, producing incomplete or silent output instead of a useful error.
11. File references containing `/` were split incorrectly; ordinary relative and absolute paths could not be parsed reliably.
12. Schedule-relative audio paths were resolved against the process working directory rather than the schedule file's directory.
13. Non-44.1 kHz source audio raised an error instead of being resampled despite SciPy already being a dependency.
14. Mono and multichannel files did not have an explicit stereo normalization path.
15. Pink-noise filter state reset every chunk, creating discontinuities.
16. The package carried a second nested source copy. It is preserved for provenance, but packaging did not explicitly exclude it from the canonical distribution.
17. Tests covered only one happy-path parser example and one one-second mixer example.

## Beads

### Bead 1 — Resolve live state and define the train
**Status:** complete

- Resolved `main`, open PRs, open issues, and recent history.
- Confirmed there were no open PRs or issues to inherit.
- Converted concrete product failures into this bounded train.

### Bead 2 — Establish one public application API
**Status:** complete

- Add `pysbagen.api` for building specs, rendering schedules/specs, and streaming output to disk.
- Make both CLI and GUI use the same API.
- Export the stable entry points from `pysbagen`.

### Bead 3 — Make timeline and mixing behavior truthful
**Status:** complete

- Render exact requested durations.
- Continue mixing when one source ends.
- Preserve silence instead of collapsing the schedule.
- Honor explicit duration boundaries.
- Fail loudly on unknown tone-set names.
- Remove scheduled generators by object identity so equal-valued tone sets do not erase each other.

### Bead 4 — Make schedules and background audio dependable
**Status:** complete

- Parse quoted paths and paths containing directory separators.
- Resolve files relative to the SBG file.
- Resample source files to 44.1 kHz.
- Normalize mono/multichannel input to stereo.
- Make looping a real `FileSpec` behavior.
- Keep pink-noise filter state continuous.
- Expand user paths consistently for MP3 and other supported audio files.

### Bead 5 — Repair the GUI as a real front door
**Status:** complete

- Remove the `sys.argv`/CLI impersonation hack.
- Keep export usable without PyAudio.
- Repair quick, advanced, schedule, I-Doser, tone-builder, soundscape, preview, and visualization paths.
- Route worker-thread completion back through Tk's main thread.

### Bead 6 — Make installation and ownership unambiguous
**Status:** complete

- Configure package discovery explicitly.
- Exclude the preserved nested source copy and tests from the wheel.
- Add CLI and GUI entry points plus `dev` and `gui` extras.
- Document the canonical package and the preserved legacy/original source.

### Bead 7 — Prove the user journeys
**Status:** complete

- Add parser, mixer, schedule, file, API, and CLI tests.
- Add GitHub Actions qualification on supported Python versions.
- Validate locally with 14 passing tests, compile checks, and a wheel build.

## Qualification

- `PYTHONPATH=. pytest -q` → **14 passed**
- `python -m compileall -q pysbagen gui.py visualization.py drg_decoder.py` → **passed**
- `python -m pip wheel . --no-deps --no-build-isolation` → **wheel built successfully**
- Wheel inspection confirms the canonical `pysbagen` package and GUI modules are included while the nested duplicate and tests are excluded.

## Next product train

Do not reopen this train as an audit of the audit. The next useful train should be one of:

1. **Schedule compatibility train:** qualify against a representative library of original SBaGen schedules and implement unsupported schedule operators from real fixtures.
2. **I-Doser compatibility train:** add legal user-supplied `.drg` fixtures, harden decoding, and prove image/schedule extraction end to end.
3. **Live studio train:** add pause/stop, transport position, non-blocking preview, and live parameter changes to the GUI.
