# HTE / InvisiSynth Gap Synthesis: Living Sessions

**Date:** July 31, 2026  
**Source pack:** project-owner supplied `HTE-Newest.zip`  
**Status:** Applied to product design and implementation  
**Code boundary:** No HTE engine code was vendored into PySbagen

## Question

What would make PySbagen fun and memorable enough to become a repeatedly used personal system rather than a generator someone tries once and puts in a drawer?

The answer must preserve the product boundary established by the SBaGenX interoperability train:

- SBaGenX owns advanced native SBaGen DSP, SBGF execution, authoring, plotting, export, and frontend work.
- PySbagen owns compatibility truth, provenance, guided products, local history, personal learning, research records, and backend-independent experience orchestration.

## HTE material used

The reasoning pass used the supplied pack's:

- `11-invisisynth-updated-*` — gap detection, domain-gap analysis, confidence tiers;
- `08-learning-*` — prediction, outcome, mismatch, and calibration loop;
- `09-affect-*` — affect-tagged memory and state-context recall;
- `10-parallel-oracle-*` — explicit combination/permutation search;
- `06-synthesis-*` — collapse competing ideas into an implementable product;
- `core-thinking-updated-*` — surprising cross-domain connections;
- `holo_intersection-*` and `holo_lateral-*` concepts — mechanism intersections and lateral substitutions.

The uploaded scripts were treated as reasoning references. They were not executed against private services, copied into the repository, or made runtime dependencies.

## InvisiSynth gap report

### Gap 1 — The generator is not the relationship

Advanced audio generation, sequencing, and native rendering can be excellent while still producing a disposable experience. A rendered file has no continuing identity, no ancestry, no remembered moments, and no reason to return beyond manually repeating it.

**Confidence:** high. The current repository already had exact recipes, seeds, hashes, manifests, and local library records, but no layer joining them into continuing personal sessions.

### Gap 2 — Repetition and novelty are treated as opposites

Audio products often force a bad choice:

- exact repetition becomes stale;
- uncontrolled novelty destroys reproducibility and learning.

The missing mechanism is a lineage where the listener can deliberately choose:

- exact return;
- one-variable branch;
- high-salience contrast;
- bounded multi-variable exploration.

Every change remains disclosed and reproducible.

### Gap 3 — Personal history is stored but not made useful

A manifest proves what happened technically. It does not help a person remember:

- which experience felt meaningful;
- where a shift happened;
- what they would repeat;
- which sound worlds or variation modes appear promising;
- whether a new variation differs in one way or five.

The missing product is an inspectable personal atlas built from immutable local records.

### Gap 4 — Emotional memory is absent from protocol identity

A SHA-256 is exact but not memorable. A marketing title is memorable but often disconnected from the exact recipe.

The bridge is a deterministic human identity generated from the recipe hash:

- evocative two-word title;
- three-word motif;
- exact recipe SHA-256 underneath.

The identity is memorable without weakening provenance.

### Gap 5 — Feedback systems usually become either manipulative or vague

Streaks, points, fake achievements, and opaque recommendations create engagement without creating understanding. Pure journaling creates understanding but asks too much effort and may never affect the next session.

The missing middle is:

- an optional small outcome record;
- transparent next-mode rules;
- descriptive pattern candidates only after repeated observations;
- no medical-efficacy inference;
- no hidden scoring model;
- no cloud account.

## Cross-domain intersection synthesis

| Domain | Useful mechanism | PySbagen combination |
|---|---|---|
| Version control | parent commits, branches, immutable identity | session lineage, parent session, recipe hash |
| Music | motif, reprise, variation, memorable movement | title/motif identity, exact return, disclosed branch |
| Experimental design | change one variable to learn causality | branch mode |
| Procedural generation | deterministic seeds and bounded variation | reproducible generated-bed changes |
| Journaling | event and outcome memory | echoes, notes, affect snapshots, outcome records |
| Cartography | locations connected into a meaningful whole | personal session atlas and lineages |
| Ritual | named return, threshold, remembered moment | stable session identity and echoes without mandatory mysticism |
| Games | replayability through meaningful choice | return/branch/contrast/wander without points or coercive streaks |

## Original product synthesis

### Living Sessions

A session is no longer a disposable output file. It is a reproducible experience with:

- a memorable identity;
- exact recipe identity;
- a lineage;
- optional state context;
- append-only events;
- a later outcome;
- transparent next routes.

### Four return modes

#### Return

Recreate the exact remembered recipe. Same recipe hash, same title, same motif, new session occurrence.

#### Branch

Change exactly one disclosed product-level dimension. This is the preferred learning mode.

Current dimensions:

- deterministic generated-bed seed;
- duration;
- intensity;
- sound-world binding;
- one underlying layer toggle.

The stated sleep problem is never silently changed.

#### Contrast

Change one high-salience dimension to deliberately test a clearly different route after a poor or uncomfortable result.

#### Wander

Combine at most two compatible disclosed changes for exploration. The plan explicitly marks the result as less causally interpretable.

### Echoes

An echo is a user-marked moment worth remembering. It records:

- session and lineage identity;
- optional transport position;
- a human label;
- optional local structured payload.

Echoes are not audio extraction yet. A later orchestration bead may turn selected echoes into cue or structural-reprise inputs without copying rendered audio blindly.

### Personal atlas

The atlas reports:

- session, completion, lineage, and echo counts;
- exact lineage histories;
- average optional rating and would-repeat rate;
- average recorded affect delta when before/after snapshots exist;
- descriptive pattern candidates only after repeated observations.

Patterns are labeled as local descriptive records, not medical conclusions.

## Anti-drawer test

A feature passes only when continued use creates new value.

| Test | Requirement |
|---|---|
| Return value | A prior session can be recreated exactly |
| Memory value | Marked moments and outcomes remain useful later |
| Learning value | Branches reveal exactly what changed |
| Surprise value | Exploration is bounded, named, and reproducible |
| Identity value | A person can remember a session without losing its hash |
| Accumulation value | The atlas becomes more informative with honest history |
| Autonomy value | No account, cloud service, streak pressure, or hidden ranking |
| Backend value | The experience layer works above Python now and SBaGenX later |

## Rejected ideas

- arbitrary badges, points, streaks, or daily-pressure mechanics;
- random frequencies or undisclosed parameter mutation;
- fake AI claims based on one session;
- global leaderboards or social comparison;
- opaque personalization;
- importing game terminology directly into the user interface;
- changing the user's stated problem merely to create novelty;
- replacing exact recipes with only human-friendly names;
- burying session memory inside a single GUI.

## Implementation chosen

The first product foundation is implemented as:

- `pysbagen/living_sessions.py`;
- `pysbagen/session_cli.py`;
- `sbgpy-session`;
- local archive under the platform data directory;
- deterministic identities and lineages;
- return/branch/contrast/wander planning;
- exact sleep-recipe reconstruction and rendering;
- echoes, outcomes, affect snapshots, and atlas summaries;
- focused tests.

## Next original combinations

These remain product work, not promises that they already exist:

1. **Constellation view** — a visual lineage/echo map rather than a file list.
2. **Confluence** — combine two well-understood lineages while preserving both parent identities and disclosing every inherited dimension.
3. **Echo weaving** — use selected remembered moments as structural or cue anchors above either renderer.
4. **Seasonal return** — surface a remembered session because its context resembles a prior useful context, without push-pressure mechanics.
5. **Shareable seed capsule** — export a self-contained recipe/lineage identity with no personal outcome history unless explicitly included.
6. **Mystery with receipts** — reveal the evocative identity first, then show every exact parameter and source identity before playback.

## Confidence and limits

This pass establishes an original product synthesis for this repository. It does not claim worldwide patent novelty or prove that no adjacent product has ever used any individual primitive. The originality is in the explicit combination, product boundary, provenance rules, and implemented local-first loop.
