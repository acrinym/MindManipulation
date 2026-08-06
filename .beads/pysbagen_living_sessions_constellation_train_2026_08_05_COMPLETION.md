# PySbagen Living Sessions — Constellation Completion Receipt

**Date:** August 5, 2026  
**Status:** Complete and qualified  
**Branch:** `agent/living-sessions-constellation-train-20260805`  
**Pull request:** `#13` — Build offline Living Sessions constellation navigator  
**Stack base:** PR `#12` / `agent/living-sessions-train-20260731`

## Product delivered

The Constellation turns the local Living Sessions archive into a reusable offline navigation surface.

Delivered capabilities:

- nodes derived only from immutable session plans, events, and outcomes;
- edges derived only from preserved parent-session identities;
- deterministic generation-based layout;
- separated multi-lineage layout;
- complete-lineage focus from any session ID;
- title, motif, mode, state, backend, recipe identity, and provenance display;
- mutation labels and reasons;
- echo, render, and other event inspection;
- outcome, comfort, repeat-intent, and tag inspection;
- search across titles, motifs, IDs, mutations, events, and tags;
- lineage, mode, and state filters;
- stable graph SHA-256 identity independent of export time;
- integrity warnings without invented repairs;
- full-detail private export;
- privacy-reduced `--redact-notes` export;
- self-contained HTML usable from `file://`;
- machine-readable JSON export;
- `sbgpy-constellation` CLI and export receipts.

## Product paths

```bash
sbgpy-constellation -o my-constellation.html
sbgpy-constellation --session SESSION_ID -o focused-lineage.html --open
sbgpy-constellation --lineage LINEAGE_ID --redact-notes -o redacted.html
sbgpy-constellation --format json --summary-json -o constellation.json
```

## Offline and privacy guarantees

The HTML snapshot loads no remote scripts, stylesheets, fonts, analytics, accounts, or services.

Private snapshots may contain local notes and event details. Redacted snapshots remove:

- session rationale;
- affect notes;
- outcome notes;
- event labels and payloads;
- user-audio paths.

Technical topology and provenance remain visible. Redaction is not presented as anonymization.

Archive strings containing closing-script sequences are encoded so they remain data and cannot terminate the embedded JSON block. This is tested for both private and redacted exports.

## Integrity behavior

The Constellation reports rather than silently repairs:

- missing or out-of-snapshot parents;
- non-root generations without parents;
- cross-lineage parent relationships;
- generation discontinuities;
- parent cycles.

No similarity edges or replacement ancestry are invented.

## Product boundary retained

- SBaGenX remains responsible for advanced native SBG/SBGF synthesis and authoring.
- PySbagen owns session memory, lineage, navigation, outcomes, and provenance above the renderer.
- No native DSP was duplicated.
- No Cycloside analysis, design, branch, issue, prototype, or integration work was performed.
- `docs/planning/CYCLOSIDE_LIVING_SESSIONS_PERMISSION_GATE.md` remains authoritative.

## Qualification

GitHub Actions qualification run `#66` passed implementation head:

`7f4e9a5801ad98ec1c76d9518b6efe086d1ff719`

Results:

- Python 3.10 — passed;
- Python 3.11 — passed;
- Python 3.12 — passed;
- Python 3.13 — passed;
- **74 tests passed**;
- source distribution built;
- wheel built;
- wheel contains:
  - `pysbagen/constellation.py`;
  - `pysbagen/constellation_model.py`;
  - `pysbagen/constellation_render.py`;
  - `pysbagen/constellation_cli.py`;
  - `pysbagen/data/constellation_template.html`;
  - the `sbgpy-constellation` entry point.

## Qualification corrections

### Embedded renderer removal

The first CI run exposed an invalid Python f-string containing JavaScript template expressions. The embedded renderer was deleted, not masked. The final architecture separates:

- graph truth and integrity;
- HTML rendering;
- packaged offline template;
- CLI/export behavior;
- public facade.

### Private-export script-safety coverage

The first focused safety assertion relied on redaction to remove hostile event text. Final coverage proves that an unredacted private snapshot preserves the text only as escaped JSON and cannot terminate the data script.

## Review truth

- Bugbot is not enabled and performed no review.
- CodeRabbit automatic review skipped the stacked non-default base.
- A manual `@coderabbitai review` request was posted for graph integrity, redaction, HTML safety, deterministic identity, CLI behavior, packaging, and boundary preservation.
- Any later actionable thread must be fixed before merge; no bot review is claimed merely from a status check.

## Anti-drawer acceptance

Repeated use creates increasing value because the view accumulates real branches, echoes, outcomes, and provenanced return points. The Constellation is a place to remember and choose—not a decorative animation, streak mechanic, ranking system, or opaque recommendation score.

## Next train

`LIV-012 — Confluence Sessions` is the next queued product train. It has not started. It may combine two parent lineages only with complete inherited-dimension and provenance receipts.
