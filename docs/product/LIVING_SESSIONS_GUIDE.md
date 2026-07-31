# PySbagen Living Sessions Guide

**Status:** Wave 1 qualified  
**Command:** `sbgpy-session`  
**Storage:** Local-only by default

## What Living Sessions changes

A PySbagen session is no longer only a generated WAV file. It has:

- a memorable title and motif;
- an exact recipe SHA-256;
- a lineage and parent session;
- a declared variation mode;
- optional remembered moments called echoes;
- optional before/after state context;
- an immutable outcome record;
- a transparent route to the next session.

The memorable identity never replaces technical provenance. Every variation remains inspectable.

## Create a root sleep session

```bash
sbgpy-session new-sleep \
  --problem racing_mind \
  --sound-world rain_room \
  --intensity balanced \
  --duration 45 \
  --seed 17
```

Optional pre-session state context:

```bash
sbgpy-session new-sleep \
  --problem racing_mind \
  --sound-world rain_room \
  --pre-valence -0.4 \
  --pre-arousal 0.8 \
  --pre-agency 0.3
```

Valence is recorded from `-1` to `1`. Arousal and agency are recorded from `0` to `1`. These are personal descriptive notes, not clinical measurements.

The command prints an identity similar to:

```text
Velvet Threshold · rain / hush / return
Session: 6f8d...
Lineage: 3c2a... · generation 0 · mode root
Recipe SHA-256: ...
```

Names and motifs are deterministic consequences of exact recipe identity. They are not random labels stored separately from the recipe.

## Render the exact session

```bash
sbgpy-session render SESSION_ID -o tonight.wav
```

The command:

1. loads the immutable plan;
2. reconstructs and validates the stored `SleepRequest`;
3. enforces the declared backend policy;
4. renders only through a currently qualified backend;
5. writes the normal exact sleep recipe manifest;
6. hashes the output;
7. appends a render event containing recipe, backend, reason, and output identity.

Backend behavior in Wave 1:

- `python` — renders through the qualified portable Python backend and records that explicit policy;
- `auto` — selects Python while native rendering remains unqualified and records the selection reason;
- `sbagenx` — fails closed instead of silently rendering through Python.

A native-required plan becomes renderable only after the typed SBaGenX render/context/writer and receipt train is qualified.

## Mark a memorable echo

```bash
sbgpy-session mark SESSION_ID \
  --kind echo \
  --at 123.5 \
  --label "The rain became a room"
```

Other event kinds:

- `shift`;
- `insight`;
- `discomfort`;
- `custom`.

An echo is a metadata anchor, not an extracted audio clip. Later orchestration can use selected echoes without silently copying or modifying audio.

Optional structured payload:

```bash
sbgpy-session mark SESSION_ID \
  --kind echo \
  --label "Warm threshold" \
  --payload '{"context":"lights off","certainty":"approximate"}'
```

## Finish with an optional outcome

```bash
sbgpy-session finish SESSION_ID \
  --rating 5 \
  --would-repeat yes \
  --comfort comfortable \
  --post-valence 0.4 \
  --post-arousal 0.2 \
  --post-agency 0.7 \
  --tag rain \
  --tag settled
```

Outcome records are immutable. The tool does not rewrite history when later sessions differ.

## Choose the next route

Automatic transparent route:

```bash
sbgpy-session next SESSION_ID --mode auto
```

Explicit route:

```bash
sbgpy-session next SESSION_ID --mode return
sbgpy-session next SESSION_ID --mode branch
sbgpy-session next SESSION_ID --mode contrast
sbgpy-session next SESSION_ID --mode wander
```

### Return

Reuses the exact recipe hash, title, and motif. It creates a new occurrence in the same lineage.

### Branch

Changes exactly one disclosed dimension. This is the preferred mode for learning what mattered.

### Contrast

Changes exactly one high-salience audible product dimension. Seed-only novelty is excluded whenever an audible contrast is available. Automatic mode selects contrast after an uncomfortable or low-rated outcome.

### Wander

Combines at most two compatible disclosed changes. The plan is marked experimental and less causally interpretable.

## Inspect a session

```bash
sbgpy-session show SESSION_ID
sbgpy-session show SESSION_ID --json
```

List all local sessions:

```bash
sbgpy-session list
sbgpy-session list --json
```

## View the personal atlas

```bash
sbgpy-session atlas
sbgpy-session atlas --json
```

The atlas can show:

- session counts by state;
- lineage count and title history;
- echo count;
- optional average rating;
- optional would-repeat rate;
- optional average affect delta;
- descriptive sound-world or variation-mode candidates after repeated observations.

The atlas does not claim medical effectiveness. It describes the user's own recorded history.

## Local storage

Default locations:

- Linux: `$XDG_DATA_HOME/pysbagen/living-sessions` or `~/.local/share/pysbagen/living-sessions`;
- Windows: `%LOCALAPPDATA%\PySbagen\living-sessions`;
- macOS currently follows the XDG-style fallback unless `XDG_DATA_HOME` is configured.

Use another root:

```bash
sbgpy-session --root ./my-session-atlas new-sleep ...
sbgpy-session --root ./my-session-atlas atlas
```

Each session directory contains:

- `plan.json` — immutable identity and exact recipe;
- `events.jsonl` — append-only event and echo ledger;
- `outcome.json` — optional immutable outcome.

## Automatic recommendation rules

`--mode auto` currently uses deliberately simple rules:

- no outcome → branch;
- uncomfortable or rating 1–2 → contrast;
- rating 4–5 plus would-repeat → exact return once;
- another strong exact return → branch;
- otherwise → branch.

These rules are visible in code and can be explained from the local record. There is no hidden model or remote scoring service.

## Qualification receipt

Wave 1 passed GitHub Actions Python qualification run `#57`:

- Python 3.10 — passed;
- Python 3.11 — passed;
- Python 3.12 — passed;
- Python 3.13 — passed;
- complete repository result — **68 tests passed**;
- source distribution and wheel build — passed;
- wheel contains `living_sessions.py`, `living_session_policy.py`, and `session_cli.py`;
- modern SPDX license metadata builds without the prior setuptools license-table warning.

## What this deliberately avoids

- points, badges, daily streaks, or leaderboards;
- undisclosed random changes;
- opaque personalization;
- cloud accounts;
- social comparison;
- medical claims based on subjective history;
- replacing SBaGenX DSP;
- locking the experience layer to one GUI.

## Next product work

The active train records:

- constellation visualization;
- two-parent confluence sessions;
- echo weaving into backend-independent orchestration;
- shareable seed capsules;
- Living Sessions for imported SBG, SBGF, and research protocols.

See `.beads/pysbagen_living_sessions_train_2026_07_31.md`.
