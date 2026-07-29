# pysbagen Architecture

## Product boundary

The maintained Python product has one direction of dependency:

```text
CLI / GUI
   ↓
pysbagen.api
   ↓
parser + mixer
   ↓
generator specifications
   ↓
NumPy / SciPy / SoundFile / pydub
```

The CLI and GUI are adapters. They gather user input and call the public API; neither owns a separate rendering engine.

## Canonical code

- `pysbagen/api.py` — public render/write operations and quick-spec construction.
- `pysbagen/parser.py` — SBG text and file parsing, including schedule-relative media paths.
- `pysbagen/mixer.py` — exact-duration stereo timeline, limiting, silence, and scheduled activation.
- `pysbagen/generators/` — built-in audio-source specifications.
- `gui.py` — Tk desktop adapter.
- `pysbagen/cli.py` — command-line adapter.

`pysbagen/src/pysbagen/` is an earlier duplicate retained for provenance. Package discovery explicitly excludes it so installed behavior cannot depend on which copy Python happens to find.

## Audio contract

Built-in and third-party generators expose:

```python
generator(duration) -> iterator[(chunk, info)]
```

The mixer expects 44.1 kHz. Chunks may be mono, stereo, or multichannel and may have different lengths; the mixer normalizes them to stereo, buffers uneven chunks, and emits exactly the requested frame count. A generator that ends early becomes silence rather than ending the entire session.

The writer receives stereo float chunks and streams them to disk. This keeps long renders bounded in memory.

## Schedule contract

The parser returns:

```python
(tone_sets, schedule)
```

- `tone_sets` maps labels to generator objects.
- `schedule` is an ordered list of `(timestamp_seconds, tokens)` events.

The scheduler renders the state active before each event, applies the event at its timestamp, and continues until the chosen endpoint. Empty active state produces silence. An unprefixed event is absolute; `+` and `-` events are relative.

The final timestamp is the endpoint when the caller does not provide an explicit duration. Schedules should therefore end with `off` when they need a final audible section.

## Optional capabilities

Core export requires NumPy, SciPy, SoundFile, and pydub. GUI rendering, artwork, and visualization use the `gui` extra. PyAudio is only required for live preview; missing playback support must never block file generation.

## Compatibility work

Compatibility additions must be driven by real schedules or legal user-supplied fixtures. Add a failing fixture/test first, implement the missing operator or decoder behavior, then update the supported-syntax documentation. Avoid speculative parser branches that silently reinterpret unknown syntax.
