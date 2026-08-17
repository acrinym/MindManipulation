"""First-class preservation and structural inspection for SBaGenX ``.sbgf`` files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .compatibility import (
    CompatibilityFinding,
    CompatibilityState,
    ImportReport,
    MissingSource,
    RenderDisposition,
    SourceLocation,
    sha256_bytes,
)
from .importers import ImportedArtifact

_PARAM_RE = re.compile(r"^\s*param\s+([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", re.IGNORECASE)
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$")
_SOLVE_RE = re.compile(r"^\s*solve\b(.*)$", re.IGNORECASE)
_FUNCTION_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_MEDIA_RE = re.compile(
    r"(?P<quote>['\"])(?P<path>[^'\"]+\.(?:wav|wave|aif|aiff|flac|ogg|mp3))(?P=quote)",
    re.IGNORECASE,
)


def import_sbgf(path: str | Path) -> ImportedArtifact:
    """Preserve and structurally inspect one SBaGenX function-curve source."""

    source_path = Path(path).expanduser().resolve()
    if source_path.suffix.lower() != ".sbgf":
        raise ValueError("SBGF import requires a .sbgf source")
    data = source_path.read_bytes()
    text, encoding = _decode_text(data)

    parameters: dict[str, str] = {}
    assignments: dict[str, str] = {}
    solve_directives: list[dict[str, Any]] = []
    expression_functions: set[str] = set()
    referenced_media: dict[str, set[int]] = {}
    findings: list[CompatibilityFinding] = [
        CompatibilityFinding(
            "sbgf-source-preserved",
            "SBaGenX function source preserved",
            CompatibilityState.SUPPORTED,
            "The original bytes, encoding, source locations, and SHA-256 identity are retained.",
        )
    ]

    for line_number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        location = (SourceLocation(line=line_number, text=raw),)
        parameter_match = _PARAM_RE.match(raw)
        if parameter_match:
            name, expression = parameter_match.groups()
            if name in parameters:
                findings.append(
                    CompatibilityFinding(
                        "sbgf-duplicate-parameter",
                        "Duplicate SBGF parameter",
                        CompatibilityState.PARTIAL,
                        f"Parameter {name!r} is declared more than once; native validation decides final legality.",
                        location,
                    )
                )
            parameters[name] = expression
            expression_functions.update(_FUNCTION_RE.findall(expression))
            _record_media(expression, line_number, referenced_media)
            continue
        solve_match = _SOLVE_RE.match(raw)
        if solve_match:
            solve_directives.append({"line": line_number, "expression": solve_match.group(1).strip()})
            expression_functions.update(_FUNCTION_RE.findall(solve_match.group(1)))
            continue
        assignment_match = _ASSIGN_RE.match(raw)
        if assignment_match:
            name, expression = assignment_match.groups()
            if name in assignments:
                findings.append(
                    CompatibilityFinding(
                        "sbgf-duplicate-assignment",
                        "Duplicate SBGF assignment",
                        CompatibilityState.PARTIAL,
                        f"Output/expression {name!r} is assigned more than once; native validation decides final legality.",
                        location,
                    )
                )
            assignments[name] = expression
            expression_functions.update(_FUNCTION_RE.findall(expression))
            _record_media(expression, line_number, referenced_media)
            continue
        findings.append(
            CompatibilityFinding(
                "sbgf-unclassified-line",
                "SBGF line preserved but not structurally classified",
                CompatibilityState.UNKNOWN,
                "PySbagen preserves this line and defers its exact semantics to qualified SBaGenX validation.",
                location,
            )
        )

    if not assignments:
        findings.append(
            CompatibilityFinding(
                "sbgf-no-assignments",
                "No SBGF output assignments were identified",
                CompatibilityState.UNKNOWN,
                "The artifact remains preserved, but PySbagen cannot infer a runnable curve program from structure alone.",
            )
        )

    missing_sources: list[MissingSource] = []
    media_records: list[dict[str, Any]] = []
    for raw_path, lines in sorted(referenced_media.items()):
        resolved = (source_path.parent / raw_path).resolve()
        present = resolved.is_file()
        media_records.append(
            {
                "declared_path": raw_path,
                "resolved_path": str(resolved),
                "source_lines": sorted(lines),
                "present": present,
            }
        )
        if not present:
            missing_sources.append(MissingSource(str(resolved), tuple(f"line {line}" for line in sorted(lines))))
            findings.append(
                CompatibilityFinding(
                    "sbgf-missing-media",
                    "Referenced SBGF media is missing",
                    CompatibilityState.MISSING_SOURCE,
                    str(resolved),
                    tuple(SourceLocation(line=line) for line in sorted(lines)),
                    remediation="Restore the referenced media or deliberately relink it before native execution.",
                )
            )

    findings.append(
        CompatibilityFinding(
            "sbgf-native-runtime-required",
            "Qualified SBaGenX runtime required for execution",
            CompatibilityState.UNSUPPORTED,
            "PySbagen preserves and inspects SBGF but does not independently reinterpret its expression language or render it.",
            remediation="Validate and later render through a supported SBaGenX native backend.",
        )
    )

    report = ImportReport(
        source_path=str(source_path),
        source_type="sbgf",
        source_size=len(data),
        source_sha256=sha256_bytes(data),
        encoding=encoding,
        version_clues=["SBaGenX function-curve source"],
        metadata={
            "parameters": parameters,
            "parameter_count": len(parameters),
            "assignments": assignments,
            "assignment_count": len(assignments),
            "solve_directives": solve_directives,
            "expression_functions": sorted(expression_functions),
            "referenced_media": media_records,
        },
        findings=findings,
        missing_sources=missing_sources,
        start_mode="function-driven",
        end_behavior="defined by SBGF program/runtime arguments",
        loop_behavior="defined by SBaGenX runtime configuration",
        render_disposition=RenderDisposition.INSPECTION_ONLY,
    )
    return ImportedArtifact(report, {}, [], source_text=text)


def _record_media(expression: str, line_number: int, records: dict[str, set[int]]) -> None:
    """Record quoted media references while retaining all source-line provenance."""

    for match in _MEDIA_RE.finditer(expression):
        records.setdefault(match.group("path"), set()).add(line_number)


def _decode_text(data: bytes) -> tuple[str, str]:
    """Decode source bytes without losing a deterministic encoding record."""

    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig"), "utf-8-sig"
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1"
