# PySbagen Living Sessions — Constellation Train

**Date:** August 5, 2026  
**Status:** Implementation complete; qualification pending  
**Branch:** `agent/living-sessions-constellation-train-20260805`  
**Stack base:** PR `#12` / `agent/living-sessions-train-20260731`

## Mission

Turn Living Sessions history into an offline navigation surface people can repeatedly use to understand, revisit, compare, and extend their personal session lineages.

This is not a decorative graph export. A constellation must answer:

- Where did this session come from?
- What changed from its parent?
- Which exact experience am I looking at?
- What echoes, outcomes, and render receipts belong to it?
- Which backend and recipe identity were involved?
- What should I return to or branch from next?

## Product boundary

- SBaGenX remains responsible for advanced native SBG/SBGF DSP and authoring.
- PySbagen owns lineage identity, memory, outcomes, navigation, and provenance above either backend.
- No SBaGenX DSP is duplicated.
- The Cycloside `/newideas` path remains permission-gated and untouched.
- No cloud account, remote script, analytics beacon, leaderboard, streak, or social ranking is introduced.

## Beads

### LIV-011A — Truth-derived graph model

**Status:** complete

- derive every node from stored Living Session plans, events, and outcomes;
- derive every edge from preserved parent IDs;
- retain recipe SHA-256, backend policy, generation, mode, mutations, and experimental state;
- invent no relationships and maintain no shadow graph database.

### LIV-011B — Deterministic layout and lineage summaries

**Status:** complete

- place generations left-to-right;
- keep sibling sessions visually distinct;
- separate multiple lineages vertically;
- summarize roots, latest sessions, depth, completed sessions, and echoes;
- produce a stable graph identity independent of export time.

### LIV-011C — Offline interactive navigator

**Status:** complete

- generate one self-contained HTML file;
- use no CDN, remote font, remote script, remote stylesheet, or web service;
- render parent-child curves and mode-specific node styling;
- highlight the selected session, its parent, and immediate children;
- remain usable from `file://` without a server.

### LIV-011D — Search and filters

**Status:** complete

- search title, motif, session ID, lineage ID, mutations, events, and tags;
- filter by lineage, mode, and session state;
- hide relationships when either endpoint is filtered out;
- focus one session and automatically scope to its complete lineage.

### LIV-011E — Provenance detail surface

**Status:** complete

For each selected node expose:

- exact memory phrase;
- session, lineage, parent, and generation identity;
- mode and experimental state;
- backend policy;
- complete recipe SHA-256;
- journey parameters;
- disclosed mutations and reasons;
- echoes and other events;
- outcome and tags;
- snapshot identity.

### LIV-011F — Privacy-aware export

**Status:** complete

- support private full-detail snapshots;
- support `--redact-notes` snapshots;
- remove free-text rationale, affect notes, outcome notes, event labels/payloads, and user-audio paths in redacted mode;
- preserve technical identity and topology;
- prevent embedded archive text from terminating the JSON script block.

### LIV-011G — CLI and machine-readable format

**Status:** complete

Add `sbgpy-constellation` with:

- whole-atlas export;
- lineage restriction;
- focused-session export;
- HTML and JSON formats;
- deterministic graph SHA-256 receipt;
- optional browser opening;
- optional redaction;
- explicit warning counts.

### LIV-011H — Qualification and product guide

**Status:** implementation complete; qualification pending

Coverage includes:

- exact node/edge derivation;
- lineage focus;
- deterministic layout;
- echo, render, outcome, backend, and recipe provenance;
- script-safe self-contained HTML;
- redaction behavior;
- orphan-parent warnings;
- CLI JSON export;
- source distribution and wheel inclusion.

## Anti-drawer acceptance

The Constellation earns repeat use because it becomes more useful as lineages grow. It is a place to navigate accumulated meaning, not a novelty animation.

It deliberately avoids:

- fake stars unrelated to real sessions;
- arbitrary force-directed motion that hides ancestry;
- badges or completion pressure;
- hidden recommendation scores;
- rewriting old outcomes;
- dropping provenance for visual simplicity;
- adding a Cycloside integration without explicit permission.

## Next train after qualification

**LIV-012 — Confluence Sessions** may combine two parent lineages with complete inherited-dimension receipts. It must not begin until Constellation qualification is complete and the current stack remains healthy.
