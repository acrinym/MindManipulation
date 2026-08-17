# Living Sessions Confluence

Confluence Sessions let two remembered Living Sessions become the ancestors of a new experience.

The product idea is deliberately **not** “merge two configuration files.” A Confluence starts from two experiences that already have identities, echoes, outcomes, and history. It then makes a new session by choosing what should remain recognizable from Parent A, what should remain recognizable from Parent B, and where the meeting itself should create something new.

## The experience

A Confluence can:

- inherit a recognizable dimension from Parent A;
- inherit another dimension from Parent B;
- preserve a dimension both parents already share;
- create a deliberate bridge where two parent choices differ;
- create a fresh deterministic generation seed so the child is its own reproducible experience;
- retain both ancestors;
- become an ancestor of later Confluences;
- use the ordinary Living Sessions render, echo, outcome, and return paths after creation.

Each inheritance item is labeled as one of:

- `A` — inherited from Parent A;
- `B` — inherited from Parent B;
- `both` — already shared by both parents;
- `new` — created by the meeting itself.

That vocabulary is there to make the hybrid understandable, not to create a verification tribunal.

## What can be inherited today

The first Confluence implementation works with Living Sessions backed by PySbagen sleep recipe manifests and considers these experiential dimensions:

- intent / sleep problem;
- sound world;
- presence / intensity;
- duration;
- layer blend.

The generation seed is always new and is derived deterministically from both parent recipe identities plus the selected blend. It is never silently copied from one parent.

## Suggested inheritance

PySbagen can suggest a blend instead of making the listener manually decide every dimension.

The suggestion is intentionally simple and visible. It can use:

- recorded outcome rating;
- whether the listener marked the experience worth repeating;
- remembered echoes;
- whether a dimension is already shared;
- whether two different values have a meaningful middle or combined form.

Examples of a genuinely new bridge include:

- `gentle` + `immersive` presence becoming `balanced`;
- 30 minutes + 90 minutes becoming a 60-minute bridge;
- two compatible layer blends becoming a union that neither parent used exactly.

A suggestion is not a claim that one parent caused an outcome or that the hybrid will have a particular effect. It is a transparent product choice based on the local memory already present in Living Sessions.

## CLI

Preview a meeting without creating it:

```bash
sbgpy-confluence suggest SESSION_A SESSION_B
```

Force selected dimensions to come from specific parents while leaving the rest available for suggestion:

```bash
sbgpy-confluence suggest SESSION_A SESSION_B \
  --from-a sound_world \
  --from-b problem
```

Multiple dimensions can be comma-separated:

```bash
sbgpy-confluence create SESSION_A SESSION_B \
  --from-a sound_world,layers \
  --from-b problem
```

Inspect a created Confluence:

```bash
sbgpy-confluence show CONFLUENCE_SESSION_ID
```

Then use the normal Living Sessions workflow:

```bash
sbgpy-session render CONFLUENCE_SESSION_ID -o confluence.wav
sbgpy-session mark CONFLUENCE_SESSION_ID --kind echo --at 180 --label "The two worlds finally met"
sbgpy-session finish CONFLUENCE_SESSION_ID --rating 5 --would-repeat yes --comfort comfortable
```

A completed Confluence can be supplied as either parent in a later `sbgpy-confluence create` command.

## Dual-parent constellation

The ordinary Living Sessions plan keeps Parent A in the existing primary-parent field so existing session storage and rendering remain compatible.

The Confluence event stores the second ancestor and the inheritance story. The Confluence constellation view uses that information to add a real second edge:

```bash
sbgpy-confluence constellation
sbgpy-confluence constellation --focus CONFLUENCE_SESSION_ID
sbgpy-confluence constellation --html confluence-family.html
```

The resulting graph shows both ancestral routes without pretending that intentional cross-lineage ancestry is a broken single-parent lineage.

## Identity

A Confluence receives:

- a new session ID;
- a new lineage ID derived from both parent lineages and the new recipe;
- a new recipe SHA-256;
- a hybrid title assembled from recognizable pieces of the parent identities when possible;
- a motif that carries memories from both parents plus a meeting motif;
- a generation one greater than the older of its two parents.

The goal is that the child feels related to both parents while still being something the listener can remember by itself.

## Renderer boundary

Confluence lives above the renderer.

It does not add DSP, duplicate SBaGenX, or qualify native rendering. If both parents use the same backend policy, the child keeps it. If the parents disagree, the child uses `auto` rather than silently preferring one ancestor's renderer policy.

The existing Living Sessions rendering rules still apply. Native SBaGenX rendering remains gated on the separate typed-render qualification work.

## Product boundary

Confluence exists to make repeated use create more interesting things to return to.

It does **not** add:

- an audit engine;
- provenance scoring;
- recursive verification layers;
- causality adjudication;
- engagement points, streaks, or leaderboards;
- cloud accounts or hidden recommendation models.

HTE, InvisiSynth, memory, affect, and synthesis ideas were used as design reasoning. Their runtime machinery is not vendored into PySbagen.

The permission-gated Cycloside idea remains parked and untouched.
