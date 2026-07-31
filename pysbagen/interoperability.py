"""Combined PySbagen/SBaGenX inspection without weakening either engine's findings."""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .compatibility import CompatibilityState, ImportReport, RenderDisposition
from .importers import import_sbg
from .sbgf import import_sbgf
from .sbagenx_native import NativeValidationReport, validate_sbagenx_source


@dataclass(frozen=True)
class EngineDiscrepancy:
    """One explicit difference between PySbagen and SBaGenX conclusions."""

    code: str
    severity: str
    title: str
    detail: str


@dataclass
class InteroperabilityReport:
    """One source inspected through PySbagen and a qualified SBaGenX library."""

    source_path: str
    source_kind: str
    source_sha256: str
    pysbagen_report: ImportReport
    sbagenx_report: NativeValidationReport
    discrepancies: list[EngineDiscrepancy] = field(default_factory=list)
    schema_version: str = "pysbagen.sbagenx-interoperability.v1"

    @property
    def source_identity_matches(self) -> bool:
        """Return whether both engines inspected the same preserved source bytes."""

        return self.pysbagen_report.source_sha256 == self.sbagenx_report.source_sha256

    @property
    def native_valid(self) -> bool:
        """Return SBaGenX's validation result without changing PySbagen policy."""

        return self.sbagenx_report.valid

    @property
    def pysbagen_disposition(self) -> RenderDisposition:
        """Return PySbagen's independent render/inspection disposition."""

        return self.pysbagen_report.render_disposition

    def to_dict(self) -> dict[str, Any]:
        """Serialize both reports and their discrepancies deterministically."""

        return {
            "schema_version": self.schema_version,
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "source_sha256": self.source_sha256,
            "source_identity_matches": self.source_identity_matches,
            "native_valid": self.native_valid,
            "pysbagen_disposition": self.pysbagen_disposition.value,
            "pysbagen_report": self.pysbagen_report.to_dict(),
            "sbagenx_report": self.sbagenx_report.to_dict(),
            "discrepancies": [asdict(item) for item in self.discrepancies],
        }

    def to_text(self) -> str:
        """Render both engines' conclusions without flattening them together."""

        lines = [
            "PySbagen × SBaGenX interoperability report",
            f"Source: {self.source_path}",
            f"Type: {self.source_kind}",
            f"SHA-256: {self.source_sha256}",
            f"Source identity matches: {'yes' if self.source_identity_matches else 'NO'}",
            f"PySbagen disposition: {self.pysbagen_disposition.value}",
            f"SBaGenX native valid: {'yes' if self.native_valid else 'no'}",
            "",
            "--- PySbagen compatibility truth ---",
            self.pysbagen_report.to_text(),
            "",
            "--- SBaGenX native validation ---",
            self.sbagenx_report.to_text(),
            "",
            "--- Discrepancies ---",
        ]
        if self.discrepancies:
            for item in self.discrepancies:
                lines.append(f"[{item.severity.upper()}] {item.title}: {item.detail}")
        else:
            lines.append("none")
        return "\n".join(lines)


def inspect_with_sbagenx(
    source: str | Path,
    *,
    library: str | Path | None = None,
    loader: Callable[[str], Any] = ctypes.CDLL,
) -> InteroperabilityReport:
    """Inspect one SBG/SBGF source through both product truth layers."""

    source_path = Path(source).expanduser().resolve()
    suffix = source_path.suffix.lower()
    if suffix == ".sbg":
        artifact = import_sbg(source_path)
    elif suffix == ".sbgf":
        artifact = import_sbgf(source_path)
    else:
        raise ValueError("Interoperability inspection requires a .sbg or .sbgf source")

    native = validate_sbagenx_source(source_path, library=library, loader=loader)
    discrepancies = _compare_reports(artifact.report, native)
    return InteroperabilityReport(
        source_path=str(source_path),
        source_kind=suffix.lstrip("."),
        source_sha256=artifact.report.source_sha256,
        pysbagen_report=artifact.report,
        sbagenx_report=native,
        discrepancies=discrepancies,
    )


def _compare_reports(
    pysbagen: ImportReport,
    native: NativeValidationReport,
) -> list[EngineDiscrepancy]:
    """Describe meaningful differences while preserving PySbagen blockers."""

    discrepancies: list[EngineDiscrepancy] = []
    if pysbagen.source_sha256 != native.source_sha256:
        discrepancies.append(
            EngineDiscrepancy(
                "source-identity-mismatch",
                "error",
                "The engines did not inspect identical source bytes",
                f"PySbagen={pysbagen.source_sha256}; SBaGenX={native.source_sha256}",
            )
        )
        return discrepancies

    pysbagen_runnable = pysbagen.render_disposition in {
        RenderDisposition.SAFE,
        RenderDisposition.SAFE_WITH_DISCLOSED_CHANGES,
    }
    if pysbagen_runnable and not native.valid:
        discrepancies.append(
            EngineDiscrepancy(
                "pysbagen-accepts-native-rejects",
                "error",
                "PySbagen accepts a source that SBaGenX rejects",
                "Do not claim cross-engine portability until the native diagnostics are resolved.",
            )
        )
    elif native.valid and not pysbagen_runnable:
        limited_states = sorted(
            {
                finding.state.value
                for finding in pysbagen.findings
                if finding.state
                in {
                    CompatibilityState.PARTIAL,
                    CompatibilityState.APPROXIMATED,
                    CompatibilityState.UNSUPPORTED,
                    CompatibilityState.UNKNOWN,
                    CompatibilityState.MISSING_SOURCE,
                    CompatibilityState.INTENTIONALLY_EXCLUDED,
                    CompatibilityState.UNSAFE_TO_RENDER,
                }
            }
        )
        detail = (
            "SBaGenX validates the source, while PySbagen keeps it "
            f"{pysbagen.render_disposition.value} because of: {', '.join(limited_states) or 'product policy'}.",
        )
        discrepancies.append(
            EngineDiscrepancy(
                "native-accepts-pysbagen-limited",
                "warning",
                "Native validity does not erase PySbagen limitations",
                detail[0],
            )
        )

    native_errors = [item for item in native.diagnostics if item.severity == "error"]
    if native_errors:
        discrepancies.append(
            EngineDiscrepancy(
                "native-diagnostics-present",
                "error",
                "SBaGenX reported native errors",
                f"{len(native_errors)} native error diagnostic(s) remain attached to this exact source hash.",
            )
        )

    approximations = pysbagen.findings_by_state(CompatibilityState.APPROXIMATED)
    if native.valid and approximations:
        discrepancies.append(
            EngineDiscrepancy(
                "native-may-preserve-pysbagen-approximates",
                "warning",
                "SBaGenX may preserve semantics that PySbagen approximates",
                "; ".join(finding.title for finding in approximations),
            )
        )
    return discrepancies
