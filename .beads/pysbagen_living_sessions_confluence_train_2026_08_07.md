# PySbagen LIV-012 — Living Sessions Confluence Train

Date: 2026-08-07
Product: PySbagen
Stack base: `agent/living-sessions-constellation-20260805`
Train branch: `agent/living-sessions-confluence-20260807`

## Product objective

Make two remembered Living Sessions capable of producing a third experience that is recognizably descended from both, while remaining understandable and reusable.

This train is successful when the listener can say:

> I remember those two sessions. This new one carries something from each of them, adds something of its own, and I can see and hear how it became this experience.

## Beads

### LIV-012.1 — Dual remembered ancestors

Create a Confluence from two distinct stored Living Sessions rather than from detached recipe dictionaries.

Acceptance:

- both ancestors remain named and addressable;
- both parent recipe identities remain visible in Confluence metadata;
- same-session self-Confluence is rejected;
- a Confluence can later be used as an ancestor.

### LIV-012.2 — Experiential inheritance

Represent inherited dimensions as `A`, `B`, `both`, or `new`.

Initial dimensions:

- intent / problem;
- sound world;
- intensity;
- duration;
- layer blend.

Acceptance:

- explicit inheritance choices are supported;
- conflicting explicit A/B choices for the same trait fail clearly;
- differences remain visible as creative tensions.

### LIV-012.3 — Memory-guided suggestion

Use existing local memory to suggest understandable inheritance.

Signals may include:

- outcome rating;
- would-repeat choice;
- echoes;
- shared dimensions;
- available bridge values.

Acceptance:

- no hidden model;
- no efficacy claim;
- the suggestion explains each choice;
- both ancestors contribute when the source material permits a meaningful distinction.

### LIV-012.4 — Newness created by the meeting

Create dimensions that belong to neither parent exactly when a meaningful bridge exists.

Initial bridges:

- intensity midpoint;
- duration midpoint;
- compatible layer union;
- always-new deterministic generation seed.

Acceptance:

- `new` means a real value not copied from either parent;
- child recipe identity differs when the experience differs;
- the result remains reproducible.

### LIV-012.5 — Memorable hybrid identity

Give the Confluence its own title, motif, session ID, lineage ID, generation, and recipe identity.

Acceptance:

- identity carries recognizable material from both parents when possible;
- identity is not merely `ParentA+ParentB` or a hash dump;
- the child can be remembered and referred to on its own.

### LIV-012.6 — Normal Living Session life

After creation, a Confluence is a normal Living Session for rendering, echoes, and outcomes.

Acceptance:

- existing `sbgpy-session render` accepts the stored sleep recipe under existing backend policy;
- existing `mark` and `finish` paths apply;
- no parallel Confluence-only outcome store is created.

### LIV-012.7 — Dual-parent constellation

Add the second ancestor as a real constellation edge.

Acceptance:

- Parent A keeps the existing primary-parent compatibility path;
- Parent B receives a second graph edge;
- intentional cross-lineage ancestry is not mislabeled as a broken lineage;
- the existing offline HTML renderer can display the enriched graph.

### LIV-012.8 — Product command surface

Expose `sbgpy-confluence` with:

- `suggest`;
- `create`;
- `show`;
- `constellation`.

Acceptance:

- JSON forms remain scriptable;
- ordinary output emphasizes experience and inheritance rather than internal bookkeeping.

### LIV-012.9 — Product guide

Document the experience, inheritance vocabulary, CLI journey, dual-parent graph, renderer boundary, and anti-audit boundary.

### LIV-012.10 — Qualification

Run the repository's supported Python qualification and packaging checks on the exact final train head.

## Hard boundaries

Do not build:

- audit machinery;
- audit-of-audit machinery;
- provenance quality scoring;
- causality tribunal machinery;
- duplicate SBaGenX DSP;
- HTE runtime integration;
- Cycloside integration.

Existing identity hashes, parent identities, and structural visibility exist because a person needs to understand and return to the experience.

## Product rule

Repeated use must create new value through memory, recognizable variation, lineage, and creativity—not through engagement pressure.
