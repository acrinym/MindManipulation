from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class CompatibilityState(str, Enum):
    SUPPORTED = "supported"
    EQUIVALENT = "equivalent"
    PARTIAL = "partial"
    APPROXIMATED = "approximated"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"
    MISSING_SOURCE = "missing-source"
    RENDERED_ONLY = "rendered-only"
    UNSAFE_TO_RENDER = "unsafe-to-render"
    INTENTIONALLY_EXCLUDED = "intentionally-excluded"


class RenderDisposition(str, Enum):
    SAFE = "safe"
    SAFE_WITH_DISCLOSED_CHANGES = "safe-with-disclosed-changes"
    INSPECTION_ONLY = "inspection-only"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class SourceLocation:
    line: int | None = None
    column: int | None = None
    text: str | None = None


@dataclass(frozen=True)
class CompatibilityFinding:
    code: str
    title: str
    state: CompatibilityState
    detail: str
    locations: tuple[SourceLocation, ...] = ()
    blocking: bool = False
    remediation: str | None = None


@dataclass(frozen=True)
class PackageElement:
    index: int
    role: str
    size: int
    sha256: str
    encoding: str | None = None
    media_type: str | None = None
    text_preview: str | None = None
    stored_name: str | None = None


@dataclass(frozen=True)
class MissingSource:
    path: str
    referenced_by: tuple[str, ...] = ()


@dataclass
class ImportReport:
    source_path: str
    source_type: str
    source_size: int
    source_sha256: str
    encoding: str | None = None
    version_clues: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    package_elements: list[PackageElement] = field(default_factory=list)
    findings: list[CompatibilityFinding] = field(default_factory=list)
    missing_sources: list[MissingSource] = field(default_factory=list)
    inferred_duration: float | None = None
    start_mode: str = "unknown"
    end_behavior: str = "unknown"
    loop_behavior: str = "unknown"
    render_disposition: RenderDisposition = RenderDisposition.INSPECTION_ONLY
    schema_version: str = "pysbagen.import-report.v1"
    importer_version: str = "0.4.0"

    @property
    def can_render(self) -> bool:
        return self.render_disposition in {
            RenderDisposition.SAFE,
            RenderDisposition.SAFE_WITH_DISCLOSED_CHANGES,
        }

    @property
    def requires_acknowledgement(self) -> bool:
        return self.render_disposition is RenderDisposition.SAFE_WITH_DISCLOSED_CHANGES

    def findings_by_state(self, state: CompatibilityState) -> list[CompatibilityFinding]:
        return [finding for finding in self.findings if finding.state is state]

    def require_renderable(self, *, allow_disclosed_changes: bool = False) -> None:
        if self.render_disposition is RenderDisposition.SAFE:
            return
        if (
            self.render_disposition is RenderDisposition.SAFE_WITH_DISCLOSED_CHANGES
            and allow_disclosed_changes
        ):
            return
        if self.render_disposition is RenderDisposition.SAFE_WITH_DISCLOSED_CHANGES:
            raise ValueError(
                "Import contains disclosed approximations or partial semantics. "
                "Inspect the report and explicitly allow disclosed changes before rendering."
            )
        raise ValueError(
            f"Import is {self.render_disposition.value}; rendering is not permitted. "
            "Inspect the compatibility report for blocking findings."
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json() + "\n", encoding="utf-8")
        return destination

    def to_text(self) -> str:
        lines = [
            f"Source: {self.source_path}",
            f"Type: {self.source_type}",
            f"SHA-256: {self.source_sha256}",
            f"Compatibility: {self.render_disposition.value}",
            f"Encoding: {self.encoding or 'unknown'}",
            f"Start mode: {self.start_mode}",
            f"End behavior: {self.end_behavior}",
            f"Loop behavior: {self.loop_behavior}",
            "Duration: " + (
                f"{self.inferred_duration:.3f}s"
                if self.inferred_duration is not None
                else "open-ended or unknown"
            ),
        ]
        if self.version_clues:
            lines.append("Version clues: " + ", ".join(self.version_clues))
        if self.metadata:
            lines.append("Metadata:")
            for key in sorted(self.metadata):
                lines.append(f"  {key}: {self.metadata[key]}")
        if self.package_elements:
            lines.append("Package elements:")
            for element in self.package_elements:
                lines.append(
                    f"  [{element.index}] {element.role}: {element.size} bytes, "
                    f"{element.sha256[:12]}…"
                )
        if self.missing_sources:
            lines.append("Missing sources:")
            for source in self.missing_sources:
                suffix = f" ({', '.join(source.referenced_by)})" if source.referenced_by else ""
                lines.append(f"  {source.path}{suffix}")
        if self.findings:
            lines.append("Findings:")
            for finding in self.findings:
                marker = "BLOCK" if finding.blocking else finding.state.value.upper()
                lines.append(f"  [{marker}] {finding.title}: {finding.detail}")
                if finding.remediation:
                    lines.append(f"    Action: {finding.remediation}")
        else:
            lines.append("Findings: none")
        return "\n".join(lines)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def choose_render_disposition(
    findings: Iterable[CompatibilityFinding],
    *,
    parsed: bool,
    has_editable_recipe: bool = True,
) -> RenderDisposition:
    findings = list(findings)
    if not parsed or any(finding.blocking for finding in findings):
        return RenderDisposition.BLOCKED
    states = {finding.state for finding in findings}
    if CompatibilityState.UNSAFE_TO_RENDER in states:
        return RenderDisposition.BLOCKED
    if states.intersection(
        {
            CompatibilityState.UNSUPPORTED,
            CompatibilityState.UNKNOWN,
            CompatibilityState.MISSING_SOURCE,
        }
    ):
        return RenderDisposition.INSPECTION_ONLY
    if not has_editable_recipe or CompatibilityState.RENDERED_ONLY in states:
        return RenderDisposition.INSPECTION_ONLY
    if states.intersection(
        {
            CompatibilityState.PARTIAL,
            CompatibilityState.APPROXIMATED,
            CompatibilityState.EQUIVALENT,
        }
    ):
        return RenderDisposition.SAFE_WITH_DISCLOSED_CHANGES
    return RenderDisposition.SAFE


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value
