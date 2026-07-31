# Optional SBaGenX Backend Guide

**Status:** Discovery, API-47 validation, combined truth reports, and SBGF preservation  
**PySbagen behavior:** SBaGenX is optional; native rendering is not enabled yet

## Product boundary

SBaGenX is the advanced native SBaGen engine. PySbagen remains responsible for compatibility truth, DRG/SBGF preservation, provenance, local library records, listening-path qualification, guided products, session history, and research workflows.

The current adapter can discover and qualify an installed SBaGenX runtime, preserve `.sbgf` as a first-class artifact, and compare PySbagen inspection with API-47 native validation. It does not silently change rendering behavior.

## Probe an installation

```bash
sbgpy-inspect backend
```

Machine-readable report:

```bash
sbgpy-inspect backend --json
```

Locate configured candidates without executing the CLI or loading the shared library:

```bash
sbgpy-inspect backend --discover-only
```

Explicit paths:

```bash
sbgpy-inspect backend \
  --executable /path/to/sbagenx \
  --library /path/to/libsbagenx.so
```

## Preserve and inspect SBGF without a native runtime

```bash
sbgpy-inspect inspect curve.sbgf
```

Store the immutable source in the local content-addressed library:

```bash
sbgpy-inspect inspect curve.sbgf --add-to-library
```

PySbagen records:

- original byte count, SHA-256, and encoding;
- parameter declarations and expressions;
- solve directives and source lines;
- output/expression assignments;
- referenced function names;
- quoted media dependencies, resolved paths, and missing sources;
- unclassified lines without dropping them;
- an explicit `inspection-only` state and native-runtime requirement.

PySbagen does not reinterpret `.sbgf` as ordinary SBG events, does not invent a fake timeline, and does not claim to render the function language independently.

## Compare SBG or SBGF through both truth layers

```bash
sbgpy-inspect backend \
  --library /path/to/libsbagenx.so \
  --validate session.sbg
```

```bash
sbgpy-inspect backend \
  --validate curve.sbgf \
  --json
```

The result contains two separate reports:

1. **PySbagen compatibility truth** — preservation, missing sources, unsupported or approximated semantics, provenance, and render disposition.
2. **SBaGenX native validation** — API-47 diagnostics from the exact same source bytes.

The combined discrepancy section records when:

- the source hashes differ;
- PySbagen accepts a source that SBaGenX rejects;
- SBaGenX validates a source that PySbagen keeps limited or inspection-only;
- native error diagnostics exist;
- SBaGenX may preserve semantics that PySbagen currently approximates.

Native success never erases a PySbagen blocker, missing source, unsupported state, approximation, or provenance warning.

### Exit status

- `0` — source identity matches, SBaGenX validates it, PySbagen marks it safe, and no discrepancy remains;
- `1` — validation succeeded but a PySbagen limitation or cross-engine discrepancy must remain visible;
- `2` — source identity mismatch, native rejection, unsupported native API, missing/unloadable backend, or operational failure.

## Native validation identity

The API-47 binding records:

- immutable source byte count and SHA-256;
- detected UTF-8 BOM, UTF-8, or Latin-1 decoding;
- source kind and absolute path;
- native library path, `sbx_version()`, and `sbx_api_version()`;
- structured severity, code, line, column, range, and message diagnostics.

The binding is deliberately fail-closed and currently qualifies **SBaGenX API 47 only**, matching the source revision reviewed for this train. A new API revision requires an explicit struct/signature review rather than automatic acceptance.

## Environment overrides

- `SBAGENX_BIN` — SBaGenX command-line executable
- `SBAGENXLIB_PATH` — `sbagenxlib` shared-library path or loader name

Without overrides, PySbagen checks PATH for `sbagenx` and asks the platform dynamic-library resolver for `sbagenx`/`sbagenxlib`.

## Reported backend identity

The probe records where available:

- executable path and version parsed from the first `sbagenx -h` banner line;
- native-library path;
- `sbx_version()`;
- `sbx_api_version()`;
- presence of symbols needed for:
  - native float rendering;
  - SBG validation;
  - SBGF validation;
  - container writing;
  - live parameter control;
  - mix-stream processing.

SBaGenX uses `-V` for master volume; it does not publish a `--version` contract in the reviewed source. The adapter therefore uses the help banner for CLI identity and native functions for authoritative library identity.

A discovered candidate is not automatically a usable backend. Normal probing succeeds only when the executable or library returns a qualified identity. Discovery-only mode reports locations without claiming qualification.

## Missing backend

When SBaGenX is not installed, existing PySbagen behavior remains available:

- SBG/SBGF/DRG inspection and preservation;
- local-first library;
- audio/path qualification;
- current Python rendering for supported SBG and generated recipes;
- Sleep Guide and exact sleep recipes.

## Safety and trust boundary

Running the normal probe may execute the configured `sbagenx -h` command and load the configured shared library to read its API and symbols. Use `--discover-only` when inspecting an unfamiliar path without executing or loading it.

Native validation loads and calls the selected shared library. PySbagen never downloads, installs, vendors, updates, or renders through SBaGenX automatically.

## Remaining native-render gates

Native rendering remains blocked until:

1. context/render/writer functions have exact typed bindings and cleanup rules;
2. representative shared semantics pass parity and discrepancy fixtures;
3. backend selection is explicit and capability-based;
4. native output receives source, backend, API, configuration, and output-hash receipts;
5. Python fallback and existing guided products remain green.

See:

- `docs/planning/SBAGENX_DIFFERENTIATION_AND_INTEROP_MATRIX.md`
- `.beads/pysbagen_sbagenx_interoperability_train_2026_07_31.md`
