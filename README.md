# MindManipulation / PySbagen

PySbagen is a local layered-audio product with two front doors:

- **Sleep Guide**: answer ordinary questions and receive a matched, gradually fading audio journey.
- **Advanced Studio**: build SBaGen schedules, binaural, monaural, isochronic, Harmonic Box X-style, noise, music, I-Doser, and visualization sessions.

Its first complete human use case is sleep difficulty:

- "My mind will not stop."
- "I feel relaxed, but cannot cross into sleep."
- "I fall asleep, then keep waking back up."

## Install

Python 3.10 or newer is required.

```bash
python -m pip install .
```

Export-capable desktop interfaces:

```bash
python -m pip install -e ".[gui]"
```

Immediate playback:

```bash
python -m pip install -e ".[playback]"
```

Full desktop setup:

```bash
python -m pip install -e ".[desktop]"
```

For development:

```bash
python -m pip install -e ".[dev]"
pytest
```

PyAudio can require operating-system audio packages. Export remains available without it. SoundFile handles common formats; FFmpeg is the broad streaming fallback for other locally decodable formats.

## Sleep Guide

Open the desktop guide:

```bash
sbgpy-sleep-gui
```

Or use the terminal guide:

```bash
sbgpy-sleep
sbgpy-sleep --play
```

The guide asks:

1. What is keeping you awake?
2. What sound world feels tolerable or pleasant tonight?
3. How present should the underlying layers feel?
4. How long should the journey remain?

It chooses among three materially different routes:

- **Racing Mind Descent**: more initial movement followed by a long reduction in novelty.
- **Crossing the Threshold**: a quieter descent for someone already relaxed.
- **Stay-Asleep Support**: a shorter descent and longer stable support period.

Every route contains a **Sleep Descent** phase and a quieter **Sleep Support** phase. The current transition is planned by time. Sensor integration remains documented future work and has intentionally not been wired to nonexistent hardware or endpoints.

Generated sound worlds include warm ambience, slow night music, a rain-like room, and a deep-night environment. User-provided audio is also supported in broadly decodable formats and streams in bounded chunks instead of being loaded or tiled into a session-sized array.

Underlying binaural, monaural, isochronic, and Harmonic Box X-style layers are independently selectable, change over time, and recede during support. Saved journeys include an exact `.sleep.json` recipe.

See:

- `docs/SLEEP_GUIDE.md`
- `docs/research/SLEEP_AUDIO_RESEARCH_FOUNDATIONS.md`

## Advanced Studio

```bash
sbgpy-gui
```

The installed entry point uses a guarded wrapper that:

- allows one export and one preview at a time;
- captures Tk values before worker threads start;
- keeps Tk updates on the UI thread;
- closes replaced Matplotlib figures.

Quick CLI examples:

```bash
sbgpy --base 200 --beat 10 --duration 60 --outfile session.wav
sbgpy --isochronic 220 8 --noise 12 --noise-kind pink --duration 300 --outfile focus.wav
sbgpy --harmonic-box 180 5 8 --music "soundscapes/rain.flac" --loop-music --duration 600 --outfile rain-session.wav
```

## SBG schedules

```bash
sbgpy schedules/example.sbg --outfile scheduled.wav
```

Example:

```text
alpha: 200+10/50 pink/8
pulse: iso:220,8/35
bed: "audio/soft rain.flac/40"

NOW alpha +bed
5:00 pulse +bed ->
10:00 alpha +bed
15:00 off
```

Schedule behavior:

- an unprefixed event replaces active tone sets;
- `+name` adds and `-name` removes;
- `off`, `-`, and `alloff` clear audio as appropriate;
- silence remains on the timeline;
- still-active generators retain phase and file position across events;
- trailing `->` crossfades over the full interval to the next timed event;
- unknown names and malformed transitions are errors.

Output is written to a same-directory temporary file and atomically replaces the destination only after success. A failed or empty render preserves an existing file.

## Python API

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

## Ordinary use and research use are separate

The Sleep Guide does not secretly assign sham or blinded conditions. A future separately launched **Research Dose Environment** is reserved for informed consent, eligibility, protocol assignment, exact recipes, pre/post measures, adverse-effect reporting, and data export.

## Qualification

```bash
pytest
python -m compileall -q pysbagen gui.py gui_safe.py sleep_gui.py visualization.py drg_decoder.py
python -m pip wheel . --no-deps
```

GitHub Actions qualifies Python 3.10 through 3.13.

## Safety

Use a comfortable volume and stop on discomfort, headache, dizziness, agitation, or worsened symptoms. Do not use sleep audio while driving or doing alertness-critical work.

PySbagen does not promise diagnosis, treatment, dopamine delivery, guaranteed sleep, pain or migraine relief, sobriety, or behavioral outcomes. Severe or unusual symptoms, dangerous withdrawal, overdose, self-harm risk, and urgent crises require real-world professional, emergency, or crisis support.

## License

The Python package metadata declares `GPL-2.0-only`, consistent with the preserved SBaGen lineage. Verify media and dependency licenses before distribution.
