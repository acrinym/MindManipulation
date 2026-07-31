# MindManipulation / PySbagen

PySbagen is a local-first SBaGen/DRG compatibility inspector, layered-audio studio, and guided sleep-audio product.

Its compatibility rule is direct:

> Inspect imported schedules honestly before playback or conversion. Preserve what is present, disclose what changes, and never call an incomplete import successful merely because it produces sound.

## Product front doors

- **Compatibility Inspector** — inspect SBG/DRG files, preserve DRG packages, view timelines, qualify audio sources and listening routes, and store provenance offline.
- **Sleep Guide** — answer ordinary questions and receive a matched, gradually fading audio journey.
- **Advanced Studio** — build and render SBaGen schedules, binaural, monaural, isochronic, Harmonic Box X-style, noise, music, and visualization sessions.

Creativity-audio implementation remains deferred. Version 0.4.0 focuses on I-Doser/SBaGen compatibility, preservation, inspection, and ordinary-user reliability.

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

Development:

```bash
python -m pip install -e ".[dev]"
pytest
```

Inspection and export remain available without PyAudio. SoundFile handles common formats; FFmpeg and ffprobe are broad local fallbacks when installed.

## Compatibility Inspector

Desktop:

```bash
sbgpy-inspect-gui
```

Terminal:

```bash
sbgpy-inspect inspect schedules/example.sbg
sbgpy-inspect inspect owned-dose.drg --preserve-to preserved/owned-dose
sbgpy-inspect inspect schedules/example.sbg --json
```

Every import records:

- immutable source size and SHA-256 identity;
- source type, encoding, and version clues;
- DRG package elements in original order;
- metadata, embedded image, nested schedule, and opaque elements;
- supported, equivalent, partial, approximated, unsupported, unknown, missing-source, rendered-only, unsafe-to-render, and intentionally-excluded findings;
- start mode, end behavior, inferred duration, and loop behavior;
- exact reasons playback or rendering is permitted, disclosed, inspection-only, or blocked.

Render dispositions:

- `safe` — proceed normally;
- `safe-with-disclosed-changes` — inspect and explicitly accept partial or approximated behavior;
- `inspection-only` — preserve and inspect without a complete render claim;
- `blocked` — malformed or unsafe content cannot render.

Open-ended schedules require an explicit duration:

```bash
sbgpy open-ended.sbg --duration 1800 --outfile session.wav
```

Approximated historical constructs require acknowledgement:

```bash
sbgpy legacy-slide.sbg \
  --duration 600 \
  --allow-disclosed-changes \
  --outfile disclosed.wav
```

Schedule renders automatically receive `OUTPUT.wav.pysbagen.json`, containing the source import report, output hash, exact duration, peak, disclosed changes, and any attached listening-path qualification.

See:

- `docs/compatibility/COMPATIBILITY_INSPECTOR_GUIDE.md`
- `docs/compatibility/SBAGEN_SEMANTIC_COMPATIBILITY_MATRIX.md`

## DRG preservation

PySbagen preserves the original DRG bytes and every decoded element rather than returning only a schedule/image tuple. A preservation bundle includes:

- immutable source `.drg`;
- element files in original order;
- encoded, decoded, and decrypted hashes;
- recovered schedule and image when available;
- unknown opaque elements;
- metadata and warnings;
- provenance linking every derivative to the original DRG hash.

Only lawfully possessed or legally distributable files should be imported. Proprietary I-Doser content is not bundled or redistributed.

## Timeline and source inspection

The toolkit-independent timeline model shows chronological events, active tone sets, layers, silence, transitions, file sources, parameters, and open-ended spans. It is produced from canonical parsed data, not display-string reparsing.

Qualify a source:

```bash
sbgpy-inspect source soundscapes/rain.flac
```

The bounded-memory analyzer reports container/codec, channels, sample rate, duration, peak, clipping, stereo correlation, near-mono state, and disclosed resampling.

Qualify a listening route:

```bash
sbgpy-inspect path \
  --method binaural \
  --route headphones \
  --channels 2 \
  --sample-rate 44100 \
  --save path-qualification.json
```

Attach it to a render manifest:

```bash
sbgpy schedules/example.sbg \
  --outfile scheduled.wav \
  --path-qualification path-qualification.json
```

PySbagen distinguishes direct measurements from declared external processing. It does not claim to detect operating-system spatial enhancement, normalization, Bluetooth processing, or equalization when it cannot observe them.

## Local-first provenance library

```bash
sbgpy-inspect inspect schedules/example.sbg --add-to-library
sbgpy-inspect library list
sbgpy-inspect library show SHA256_ID
sbgpy-inspect library verify SHA256_ID
sbgpy-inspect library export SHA256_ID --destination backup-manifest.json
sbgpy-inspect library archive SHA256_ID
```

Library records distinguish recipes, packages, extracted elements, rendered audio relationships, missing sources, incompatible items, archived items, and superseded items. Duplicate bytes share canonical content identity while retaining distinct provenance records. No account or cloud service is required.

## Sleep Guide

```bash
sbgpy-sleep-gui
sbgpy-sleep
sbgpy-sleep --play
```

The guide asks what is keeping the person awake, which sound world is tolerable, how present the layers should feel, and how long the journey should remain. It chooses among Racing Mind Descent, Crossing the Threshold, and Stay-Asleep Support. Every route contains a Sleep Descent phase and a quieter Sleep Support phase.

Generated sound worlds include warm ambience, slow night music, a rain-like room, and a deep-night environment. User-provided audio is supported in broadly decodable formats and streams in bounded chunks. Saved journeys include an exact `.sleep.json` recipe.

See `docs/SLEEP_GUIDE.md` and `docs/research/SLEEP_AUDIO_RESEARCH_FOUNDATIONS.md`.

## Advanced Studio

```bash
sbgpy-gui
```

Quick CLI examples:

```bash
sbgpy --base 200 --beat 10 --duration 60 --outfile session.wav
sbgpy --isochronic 220 8 --noise 12 --noise-kind pink --duration 300 --outfile focus.wav
sbgpy --harmonic-box 180 5 8 --music "soundscapes/rain.flac" --loop-music --duration 600 --outfile rain-session.wav
```

The advanced desktop wrapper permits one export and preview at a time, captures Tk values before worker threads start, keeps Tk updates on the UI thread, and closes replaced Matplotlib figures.

## Python API

```python
from pysbagen import inspect_artifact, render_schedule, write_audio

artifact = inspect_artifact("session.sbg")
print(artifact.report.to_text())

# Rendering re-imports and enforces the same canonical policy.
result = write_audio(render_schedule("session.sbg"), "session.wav")
print(result.manifest)
```

## Ordinary use and research use are separate

The Sleep Guide does not secretly assign sham or blinded conditions. A future separately launched Research Dose Environment remains reserved for informed consent, eligibility, exact protocols, pre/post measures, adverse-effect reporting, and data export.

## Qualification

```bash
pytest
python -m compileall -q pysbagen gui.py gui_safe.py inspect_gui.py sleep_gui.py visualization.py drg_decoder.py
python -m pip wheel . --no-deps
```

GitHub Actions qualifies Python 3.10 through 3.13 when Actions budget is available. Local qualification remains a release gate.

## Safety

Use a comfortable volume and stop on discomfort, headache, dizziness, agitation, or worsened symptoms. Do not use sleep audio while driving or doing alertness-critical work.

PySbagen does not promise diagnosis, treatment, dopamine delivery, guaranteed sleep, pain or migraine relief, sobriety, creativity, or behavioral outcomes. Severe or unusual symptoms, dangerous withdrawal, overdose, self-harm risk, and urgent crises require real-world professional, emergency, or crisis support.

## License

The Python package metadata declares `GPL-2.0-only`, consistent with the preserved SBaGen lineage. Verify media and dependency licenses before distribution.
