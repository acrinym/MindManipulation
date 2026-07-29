# MindManipulation / pysbagen

`pysbagen` is a local Python audio-session builder inspired by SBaGen. It can generate and combine:

- binaural tones;
- isochronic tones;
- Harmonic Box X-style tones;
- white and pink noise;
- arbitrary sine, square, triangle, and sawtooth tones;
- WAV, OGG, FLAC, AIFF, and MP3 background audio;
- SBG schedule files;
- supported I-Doser `.drg` files through the desktop GUI;
- Chladni-style frequency visualizations.

The repository also preserves the original SBaGen 1.4.5 source and earlier Python experiments for reference. The **canonical maintained Python product** is the top-level `pysbagen/` package, the `sbgpy` command, and the `sbgpy-gui` desktop application.

## Install

Python 3.10 or newer is required.

```bash
python -m pip install .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

For the desktop GUI and live playback:

```bash
python -m pip install -e ".[gui]"
```

PyAudio can require an operating-system audio package or compiler toolchain. File generation does **not** depend on PyAudio; the GUI still opens and exports audio when live playback is unavailable.

MP3 input is decoded directly through FFmpeg, which must be installed and discoverable on the system path. WAV, OGG, FLAC, and AIFF input use `soundfile` directly.

## Quick CLI generation

Create a 60-second binaural session:

```bash
sbgpy --base 200 --beat 10 --duration 60 --outfile session.wav
```

Create an isochronic session with pink noise:

```bash
sbgpy \
  --isochronic 220 8 \
  --noise 12 \
  --noise-kind pink \
  --duration 300 \
  --outfile focus.wav
```

Combine Harmonic Box X-style audio with a looping soundscape:

```bash
sbgpy \
  --harmonic-box 180 5 8 \
  --music "soundscapes/rain.wav" \
  --music-amp 40 \
  --loop-music \
  --duration 600 \
  --outfile rain-session.wav
```

The writer streams chunks directly to disk instead of buffering an entire long session in memory.

## SBG schedules

Generate a schedule:

```bash
sbgpy schedules/example.sbg --outfile scheduled.wav
```

Override its endpoint:

```bash
sbgpy schedules/example.sbg --duration 900 --outfile scheduled.wav
```

A small schedule looks like this:

```text
# Components are declared first.
alpha: 200+10/50 pink/8
pulse: iso:220,8/35
bed: "audio/soft rain.wav/40"

# Events follow. The final event normally marks the end.
NOW alpha +bed
5:00 pulse +bed
10:00 off
```

Supported component forms:

```text
200+10/50             # base + beat / amplitude
200-10/50             # negative beat difference
white/10              # white noise / amplitude
pink/10               # pink noise / amplitude
iso:220,8/40          # frequency, beat / amplitude
hbox:180,5,8/35       # base, difference, modulation / amplitude
file:audio/rain.wav/40
"audio/soft rain.wav/40"
```

Audio paths inside schedules are resolved relative to the schedule file, not the shell's current directory. Quoted paths may contain spaces. Source audio is converted to stereo and resampled to 44.1 kHz when necessary.

Schedule event behavior:

- an unprefixed event replaces the active tone sets;
- `+name` adds a tone set;
- `-name` removes a tone set;
- `off`, `-`, or `alloff` clears audio as appropriate;
- silent spans remain real silence and do not collapse the timeline;
- unknown tone-set names produce an error instead of silently disappearing.

When no explicit `--duration` is supplied, the final schedule timestamp is treated as the endpoint. End a schedule with an `off` event when the final active section needs a defined length.

## Desktop GUI

After installing the GUI extra:

```bash
sbgpy-gui
```

The application provides:

- quick binaural export;
- advanced isochronic, Harmonic Box X, noise, and music sessions;
- SBG selection and duration override;
- I-Doser loading, artwork preview, and schedule generation;
- a multi-tone waveform builder with optional looping soundscape;
- Chladni visualization;
- optional live preview through PyAudio.

All export paths use the same `pysbagen.api` functions as the CLI. The GUI no longer changes process arguments or maintains a separate audio engine.

## Python API

```python
from pysbagen.api import build_quick_specs, render_specs, write_audio

specs = build_quick_specs(
    base=200,
    beat=10,
    noise=8,
    noise_kind="pink",
)
result = write_audio(render_specs(specs, 60), "session.wav")
print(result.duration, result.outfile)
```

For a schedule:

```python
from pysbagen.api import render_schedule, write_audio

result = write_audio(render_schedule("session.sbg"), "session.wav")
```

Third-party generator packages can register entry points under `pysbagen.generators`. A generator must expose a `generator(duration)` iterator that yields `(audio_chunk, info)` pairs at 44.1 kHz.

## Repository map

```text
pysbagen/                 Canonical maintained Python package
pysbagen/generators/      Built-in generator specifications
pysbagen/tests/           Product-path qualification
pysbagen/src/pysbagen/    Preserved duplicate from the earlier refactor; excluded from builds
gui.py                    Desktop application entry point
visualization.py          Chladni visualization functions
drg_decoder.py            I-Doser decoder
sbagen-1.4.5/             Preserved original SBaGen source and examples
.beads/                   Product-train state and continuation notes
```

## Qualification

Run:

```bash
pytest
python -m pip wheel . --no-deps
```

GitHub Actions runs the test suite on Python 3.10 through 3.13 and verifies that the package can build.

## Listening safety

Keep playback at a comfortable volume and stop if audio causes discomfort, headache, dizziness, agitation, or other unwanted effects. This software is an audio experimentation and creative-session tool; it is not medical treatment and does not promise diagnostic, therapeutic, or behavioral outcomes.

## License

The Python package metadata declares `GPL-2.0-only`, consistent with the preserved SBaGen lineage. Before distributing binaries or derivative packages, verify that all included third-party media and dependencies have compatible licenses.
