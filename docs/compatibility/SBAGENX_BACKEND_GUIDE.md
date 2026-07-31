# Optional SBaGenX Backend Guide

**Status:** Discovery and API-47 native validation foundation  
**PySbagen behavior:** SBaGenX is optional; native rendering is not enabled yet

## Product boundary

SBaGenX is the advanced native SBaGen engine. PySbagen remains responsible for compatibility truth, DRG/SBGF preservation, provenance, local library records, listening-path qualification, guided products, session history, and research workflows.

The current adapter can discover and qualify an installed SBaGenX runtime and validate `.sbg`/`.sbgf` source through the reviewed API-47 diagnostic contract. It does not silently change rendering behavior.

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

## Validate SBG or SBGF through SBaGenX

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

Native validation records:

- immutable source byte count and SHA-256;
- detected UTF-8 BOM, UTF-8, or Latin-1 decoding;
- source kind and absolute path;
- native library path, `sbx_version()`, and `sbx_api_version()`;
- structured severity, code, line, column, range, and message diagnostics;
- a truthful validity result that does not erase PySbagen compatibility findings.

The binding is deliberately fail-closed and currently qualifies **SBaGenX API 47 only**, matching the source revision reviewed for this train. New API revisions require an explicit binding review rather than automatic acceptance.

## Environment overrides

- `SBAGENX_BIN` — SBaGenX command-line executable
- `SBAGENXLIB_PATH` — `sbagenxlib` shared-library path or loader name

Without overrides, PySbagen checks PATH for `sbagenx` and asks the platform dynamic-library resolver for `sbagenx`/`sbagenxlib`.

## Reported identity

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

SBaGenX uses `-V` for master volume; it does not publish a `--version` contract in the reviewed source. The adapter therefore uses the documented help banner for CLI identity and the native functions for authoritative library identity.

A discovered candidate is not automatically a usable backend. Normal probing succeeds only when the executable or library returns a qualified identity. Discovery-only mode reports locations without claiming qualification.

## Missing backend

When SBaGenX is not installed, the command reports that explicitly. Existing PySbagen behavior remains available:

- SBG/DRG inspection and preservation;
- local-first library;
- audio/path qualification;
- current Python rendering;
- Sleep Guide and exact sleep recipes.

## Safety and trust boundary

Running the normal probe may execute the configured `sbagenx -h` command and load the configured shared library to read its API and symbols. Use `--discover-only` when inspecting an unfamiliar path without executing/loading it.

Native validation loads and calls the selected shared library. It never downloads, installs, vendors, updates, or renders through SBaGenX automatically.

## Remaining implementation gates

Native rendering remains blocked until:

1. render/context/writer functions have exact typed bindings and cleanup rules;
2. representative shared semantics pass parity/discrepancy fixtures;
3. backend selection is explicit and capability-based;
4. native output receives source, backend, API, configuration, and output-hash receipts;
5. Python fallback and existing guided products remain green.

Native validation still needs composition into a single dual-engine compatibility/discrepancy report; the standalone API-47 result is the completed foundation for that step.

See:

- `docs/planning/SBAGENX_DIFFERENTIATION_AND_INTEROP_MATRIX.md`
- `.beads/pysbagen_sbagenx_interoperability_train_2026_07_31.md`
