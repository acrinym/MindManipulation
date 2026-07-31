# PySbagen Compatibility Inspector

PySbagen 0.4.0 places inspection before schedule playback or export. SBG and DRG artifacts now share one canonical import report, one render policy, one timeline model, and one local-first provenance library.

## Inspect before rendering

Human-readable report and timeline:

```bash
sbgpy-inspect inspect path/to/session.sbg
sbgpy-inspect inspect path/to/dose.drg --preserve-to preserved/dose
```

Machine-readable output:

```bash
sbgpy-inspect inspect path/to/session.sbg --json
```

Desktop inspector:

```bash
sbgpy-inspect-gui
```

The inspector shows the immutable source hash, package elements, nested schedule status, missing files, duration/end behavior, timeline layers, and the final render disposition.

## Render dispositions

- `safe`: render normally.
- `safe-with-disclosed-changes`: inspect the partial or approximated behavior, then explicitly accept it with `--allow-disclosed-changes` or the desktop acknowledgement control.
- `inspection-only`: preserve and inspect, but do not claim a complete semantic render.
- `blocked`: malformed or unsafe content cannot render.

Open-ended schedules also require an explicit duration. PySbagen does not guess an export length.

```bash
sbgpy path/to/open-ended.sbg --duration 1800 --outfile session.wav
sbgpy path/to/legacy-slide.sbg --duration 600 --allow-disclosed-changes --outfile disclosed.wav
```

Successful schedule renders receive `OUTPUT.wav.pysbagen.json` with the exact source report, output hash, duration, peak, disclosed changes, and any attached listening-path qualification.

## DRG preservation

A preserved DRG bundle contains:

- the original immutable `.drg` bytes;
- a package manifest with source and element hashes;
- every decoded element in original order;
- decrypted schedule and image when available;
- unknown elements retained as opaque files;
- package warnings without erasing successfully recovered elements.

Only lawfully possessed or legally distributable files should be imported. The project does not bundle proprietary I-Doser content.

## Audio source qualification

```bash
sbgpy-inspect source ambience.flac
```

The bounded-memory analyzer records codec/container, channels, sample rate, duration, peak, clipping, stereo correlation, near-mono status, and disclosed resampling. Near-mono material can remain useful as ambience but is not represented as an independently separated binaural source.

## Listening-path qualification

```bash
sbgpy-inspect path \
  --method binaural \
  --route headphones \
  --channels 2 \
  --sample-rate 44100 \
  --save path-qualification.json
```

Attach the saved result to a render:

```bash
sbgpy path/to/session.sbg \
  --outfile session.wav \
  --path-qualification path-qualification.json
```

Optional declarations cover Bluetooth, spatial processing, and loudness normalization. PySbagen clearly separates what it can inspect from external processing it cannot directly detect. Mono routing or spatial cross-mixing blocks a binaural start.

## Local-first library

Add an inspected artifact:

```bash
sbgpy-inspect inspect path/to/session.sbg --add-to-library
```

Manage it offline:

```bash
sbgpy-inspect library list
sbgpy-inspect library show SHA256_ID
sbgpy-inspect library verify SHA256_ID
sbgpy-inspect library export SHA256_ID --destination backup.json
sbgpy-inspect library archive SHA256_ID
```

Each item stores the immutable source, import report, timeline, package bundle when applicable, compatibility state, provenance, relationships, and verification hashes. Duplicate content keeps one canonical content identity while preserving distinct provenance records. No account or cloud service is required.

## Compatibility exit codes

`sbgpy-inspect inspect` returns:

- `0` for `safe`;
- `1` for `safe-with-disclosed-changes`;
- `2` for `inspection-only` or `blocked`.

These codes allow scripts to separate clean compatibility, required human acknowledgement, and non-renderable imports without scraping prose.
