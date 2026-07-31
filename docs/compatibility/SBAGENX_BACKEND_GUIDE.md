# Optional SBaGenX Backend Guide

**Status:** Initial interoperability foundation  
**PySbagen behavior:** SBaGenX is optional; native rendering is not enabled yet

## Product boundary

SBaGenX is the advanced native SBaGen engine. PySbagen remains responsible for compatibility truth, DRG/SBGF preservation, provenance, local library records, listening-path qualification, guided products, session history, and research workflows.

The first adapter stage discovers and qualifies an installed SBaGenX runtime. It does not silently change rendering behavior.

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

A path or version is not permission to render. Later beads add exact ctypes signatures, API-version gates, cleanup rules, validation adapters, parity fixtures, and provenance sidecars before native output becomes selectable.

## Missing backend

When SBaGenX is not installed, the command reports that explicitly. Existing PySbagen behavior remains available:

- SBG/DRG inspection and preservation;
- local-first library;
- audio/path qualification;
- current Python rendering;
- Sleep Guide and exact sleep recipes.

## Safety and trust boundary

Running the normal probe may execute the configured `sbagenx -h` command and load the configured shared library to read its API and symbols. Use `--discover-only` when inspecting an unfamiliar path without executing/loading it.

PySbagen does not download, install, vendor, or update SBaGenX automatically.

## Next implementation gates

Native validation/rendering remains blocked until:

1. every used C function has an exact typed binding;
2. supported API revisions are declared;
3. required symbols are checked per operation;
4. native diagnostics preserve PySbagen compatibility findings;
5. cross-engine fixture comparisons are recorded;
6. backend identity and output hashes appear in render receipts.

See:

- `docs/planning/SBAGENX_DIFFERENTIATION_AND_INTEROP_MATRIX.md`
- `.beads/pysbagen_sbagenx_interoperability_train_2026_07_31.md`
