# MindManipulation / pysbagen

`pysbagen` is a local layered-audio product for people who want help changing the conditions around a difficult moment without needing to understand audio engineering.

Its first complete human use-case is **sleep difficulty**:

- “My mind will not stop.”
- “I feel relaxed, but I cannot cross into sleep.”
- “I fall asleep, then keep waking back up.”

The Sleep Guide asks a few ordinary questions, creates a matched time-changing journey, and either plays it immediately or saves it for later. Pleasant generated music, ambient sound, rain-like sound, deep-night sound, or the listener’s own audio can carry independently controlled binaural, monaural, isochronic, and Harmonic Box X-style layers.

The repository also preserves the advanced SBaGen-compatible laboratory for schedules, tone construction, I-Doser decoding, and exact audio experiments. The **canonical maintained Python product** is the top-level `pysbagen/` package.

## Start with the Sleep Guide

Install the GUI and playback extras:

```bash
python -m pip install -e ".[gui]"
```

Open the conversational desktop guide:

```bash
sbgpy-sleep-gui
```

Or use the terminal guide:

```bash
sbgpy-sleep
```

To answer the questions and begin live playback instead of saving a file:

```bash
sbgpy-sleep --play
```

The guide asks:

1. what is keeping you awake;
2. what kind of sound feels pleasant or tolerable tonight;
3. how present the underlying layers should feel;
4. how long the journey should remain before fading away.

It then selects one of three materially different routes:

- **Racing Mind Descent** — more initial movement, followed by a long reduction in novelty;
- **Crossing the Threshold** — a quieter descent for someone already relaxed;
- **Stay-Asleep Support** — a shorter descent and longer stable support bed.

Each route contains a **Sleep Descent** period and a quieter **Sleep Support** period. The present implementation uses a planned transition. Sensor-driven sleep-state detection remains documented future work and has intentionally not been wired to nonexistent devices or endpoints.

### Pleasant audio choices

PySbagen currently generates four original sound worlds:

- warm evolving ambient chords;
- slow night music with long chord movement and a sparse fading melody;
- a soft rain-like room;
- a dark low-stimulation night environment.

The listener can instead supply their own music, ambience, recording, or other audio. SoundFile handles common formats directly; FFmpeg is used as a broad fallback for other locally decodable formats. This is not a WAV-only workflow.

### Underlying layers

Normal users can accept a recommended blend. They may also choose among:

- binaural;
- monaural;
- soft isochronic modulation;
- Harmonic Box X-style multi-layer modulation.

The layers change over time and recede strongly during the support period. PySbagen does not encode one technique as universally superior.

When a journey is saved, PySbagen also writes a `.sleep.json` manifest containing the exact route, timing, carrier and beat movement, layer choices, seed, and—when supplied—the source-audio SHA-256. That makes a personally useful session reproducible without turning ordinary use into a research study.

See [`docs/SLEEP_GUIDE.md`](docs/SLEEP_GUIDE.md) for the complete product behavior and [`docs/research/SLEEP_AUDIO_RESEARCH_FOUNDATIONS.md`](docs/research/SLEEP_AUDIO_RESEARCH_FOUNDATIONS.md) for the research basis and evidence limits.

## Install the advanced studio

Python 3.10 or newer is required.

```bash
python -m pip install .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

For desktop applications and live playback:

```bash
python -m pip install -e ".[gui]"
```

PyAudio can require an operating-system audio package or compiler toolchain. File generation does **not** depend on PyAudio; both desktop applications can still export audio when live playback is unavailable.

FFmpeg should be installed and discoverable on the system path for formats that SoundFile cannot decode.

## Advanced CLI generation

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

Audio paths inside schedules are resolved relative to the schedule file, not the shell’s current directory. Quoted paths may contain spaces. Source audio is converted to stereo and resampled to 44.1 kHz when necessary.

Schedule event behavior:

- an unprefixed event replaces the active tone sets;
- `+name` adds a tone set;
- `-name` removes a tone set;
- `off`, `-`, or `alloff` clears audio as appropriate;
- silent spans remain real silence and do not collapse the timeline;
- unknown tone-set names produce an error instead of silently disappearing.

When no explicit `--duration` is supplied, the final schedule timestamp is treated as the endpoint. End a schedule with an `off` event when the final active section needs a defined length.

## Advanced desktop studio

```bash
sbgpy-gui
```

The advanced application provides:

- quick binaural export;
- isochronic, Harmonic Box X, noise, and background-audio sessions;
- SBG selection and duration override;
- I-Doser loading, artwork preview, and schedule generation;
- a multi-tone waveform builder with optional looping soundscape;
- Chladni visualization;
- optional live preview through PyAudio.

All export paths use the same `pysbagen.api` functions as the CLI.

## Python API

Create a sleep journey:

```python
from pysbagen import SleepRequest, render_sleep, write_audio
from pysbagen.sleep import build_sleep_recipe, write_recipe_manifest

request = SleepRequest(
    problem="racing_mind",
    sound_world="slow_night_music",
    intensity="balanced",
    duration_minutes=45,
)
recipe = build_sleep_recipe(request)
result = write_audio(render_sleep(request), "sleep-journey.wav")
write_recipe_manifest(recipe, result.outfile)
```

Create a direct generator session:

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

Third-party generator packages can register entry points under `pysbagen.generators`. A generator must expose a `generator(duration)` iterator that yields `(audio_chunk, info)` pairs at 44.1 kHz.

## Ordinary use and research use are separate

The Sleep Guide is help-oriented. It does not assign blinded conditions, present sham sessions, or make the user feel like an unwitting experiment.

A future **Research Dose Environment** is documented separately for consenting volunteers, protocol versioning, exact condition assignment, pre/post measures, and adverse-effect reporting. It is intentionally not stubbed into the ordinary GUI, TUI, or playback path yet.

## Repository map

```text
pysbagen/                    Canonical maintained Python package
pysbagen/sleep.py           Human sleep requests and matched recipes
pysbagen/generators/sleep.py Time-changing layered sleep synthesis
pysbagen/sleep_cli.py       Conversational terminal Sleep Guide
pysbagen/playback.py        Immediate streaming playback
sleep_gui.py                Conversational desktop Sleep Guide
pysbagen/generators/        Advanced built-in generator specifications
pysbagen/tests/             Product-path qualification
gui.py                      Advanced desktop studio
visualization.py            Chladni visualization functions
drg_decoder.py              I-Doser decoder
sbagen-1.4.5/                Preserved original SBaGen source and examples
docs/research/              Evidence and product-research foundations
.beads/                     Product-train state and continuation notes
```

## Qualification

Run:

```bash
pytest
python -m pip wheel . --no-deps
```

GitHub Actions runs the test suite on Python 3.10 through 3.13 and verifies that the package can build.

## Listening safety

Keep playback at a comfortable volume and stop if audio causes discomfort, headache, dizziness, agitation, worsened symptoms, or other unwanted effects. PySbagen is an audio experimentation and sleep-support tool. It does not promise diagnosis, treatment, dopamine delivery, guaranteed sleep, pain relief, migraine relief, sobriety, or behavioral outcomes.

Do not use sleep audio while driving, operating machinery, or doing anything that requires alertness.

## License

The Python package metadata declares `GPL-2.0-only`, consistent with the preserved SBaGen lineage. Before distributing binaries or derivative packages, verify that all included third-party media and dependencies have compatible licenses.
