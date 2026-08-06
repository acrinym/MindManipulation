# Living Sessions Constellation Guide

**Command:** `sbgpy-constellation`  
**Storage:** generated local HTML or JSON snapshot  
**Network requirement:** none

## Why it exists

Living Sessions accumulate roots, returns, branches, contrasts, wanders, echoes, outcomes, and render receipts. The Constellation makes that history navigable without weakening its provenance.

Each node is a real preserved session. Each connection is a real parent relationship. The view does not invent similarity links or hide the exact recipe identity behind visual labels.

## Create an offline constellation

```bash
sbgpy-constellation -o my-constellation.html
```

Open it automatically:

```bash
sbgpy-constellation -o my-constellation.html --open
```

The HTML contains its data, layout, styles, and interaction code. It does not contact a server or load remote assets.

## Focus one lineage

By lineage ID:

```bash
sbgpy-constellation --lineage LINEAGE_ID -o lineage.html
```

By a known session:

```bash
sbgpy-constellation --session SESSION_ID -o focused-lineage.html
```

A focused session is selected automatically, and the snapshot includes its complete lineage rather than only an isolated node.

## Navigate the HTML

The navigator supports:

- text search across titles, motifs, IDs, mutations, events, and outcome tags;
- lineage filtering;
- mode filtering for root, return, branch, contrast, and wander;
- state filtering for planned, active, and completed sessions;
- click-through details for ancestry, journey parameters, mutations, events, echoes, outcomes, backend policy, and recipe hash;
- selected-node highlighting with its immediate parent and children retained visually.

Generations run left-to-right. Multiple sessions in the same generation remain separate. Multiple lineages are separated vertically.

## Privacy-redacted snapshot

A normal snapshot contains the selected local records, which may include notes, event labels, event payloads, local output paths, or a user-audio path.

For a reduced-detail export:

```bash
sbgpy-constellation --redact-notes -o redacted-constellation.html
```

Redacted mode removes:

- free-text session rationale;
- affect notes;
- outcome notes;
- event labels and event payloads;
- user-audio paths.

It retains topology, session identity, recipe hashes, backend policy, modes, mutation keys/values, ratings, comfort, and repeat intent. This is still personal data; review it before sharing.

## JSON snapshot

```bash
sbgpy-constellation --format json -o constellation.json
```

Print an export receipt as JSON:

```bash
sbgpy-constellation --format json --summary-json -o constellation.json
```

The receipt includes:

- output path and format;
- graph SHA-256;
- selected focus and lineage filter;
- redaction state;
- node, edge, lineage, completion, echo, and warning counts.

## Snapshot identity

The graph SHA-256 is calculated from:

- graph schema;
- complete node records and deterministic layout;
- parent-child edges;
- lineage summaries;
- integrity warnings;
- focus identity.

The export timestamp is excluded, so rebuilding an unchanged archive selection produces the same graph identity.

## Integrity warnings

The Constellation preserves structural warnings instead of silently repairing history, including:

- a non-root generation with no parent;
- a missing or out-of-snapshot parent;
- a parent from another lineage;
- a generation that does not follow its parent;
- a cycle in parent relationships.

Warnings do not invent replacement edges.

## Product boundary

The Constellation does not generate or alter audio. SBaGenX remains the advanced native SBG/SBGF engine. PySbagen provides the continuing memory, identity, navigation, and outcome layer above the renderer.

The possible Cycloside `/newideas` connection remains parked behind its explicit permission gate. This feature contains no Cycloside integration or preparation work.
