from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compatibility import (
    CompatibilityFinding,
    CompatibilityState,
    ImportReport,
    MissingSource,
    RenderDisposition,
    SourceLocation,
    choose_render_disposition,
    sha256_bytes,
)
from .drg import DrgPackage, parse_drg_package, preserve_drg_package
from .parser import parse_sbg_from_string


@dataclass
class ImportedArtifact:
    report: ImportReport
    tone_sets: dict[str, list[Any]]
    schedule: list[tuple[float, list[str]]]
    source_text: str | None = None
    package: DrgPackage | None = None

    def require_duration(self, requested: float | None) -> float:
        if requested is not None:
            requested = float(requested)
            if requested <= 0:
                raise ValueError("Render duration must be positive")
            return requested
        if self.report.inferred_duration is None or self.report.inferred_duration <= 0:
            raise ValueError(
                "This schedule is open-ended or has no positive inferred duration. "
                "Choose an explicit render duration after inspecting the timeline."
            )
        return self.report.inferred_duration


def import_artifact(path: str | Path, *, preserve_to: str | Path | None = None) -> ImportedArtifact:
    source_path = Path(path).expanduser().resolve()
    if source_path.suffix.lower() == ".drg":
        return import_drg(source_path, preserve_to=preserve_to)
    if source_path.suffix.lower() == ".sbg":
        return import_sbg(source_path)
    raise ValueError(f"Unsupported import type: {source_path.suffix or '<none>'}")


def import_sbg(path: str | Path) -> ImportedArtifact:
    source_path = Path(path).expanduser().resolve()
    data = source_path.read_bytes()
    text, encoding = _decode_text(data)
    return import_sbg_text(
        text,
        source_path=source_path,
        source_bytes=data,
        encoding=encoding,
        base_dir=source_path.parent,
    )


def import_sbg_text(
    text: str,
    *,
    source_path: str | Path,
    source_bytes: bytes | None = None,
    encoding: str = "utf-8",
    source_type: str = "sbg",
    base_dir: str | Path | None = None,
) -> ImportedArtifact:
    path = Path(source_path)
    data = source_bytes if source_bytes is not None else text.encode(encoding, errors="replace")
    findings = _scan_sbg_source(text)
    tone_sets: dict[str, list[Any]] = {}
    schedule: list[tuple[float, list[str]]] = []
    parsed = False
    try:
        tone_sets, schedule = parse_sbg_from_string(text, base_dir=base_dir)
        parsed = True
    except (TypeError, ValueError) as exc:
        findings.append(
            CompatibilityFinding(
                "sbg-parse-failed",
                "Schedule could not be parsed",
                CompatibilityState.UNSAFE_TO_RENDER,
                str(exc),
                blocking=True,
                remediation="Correct the reported source line before playback or conversion.",
            )
        )

    missing_sources: list[MissingSource] = []
    audio_source_reports: list[dict[str, Any]] = []
    if parsed:
        missing_sources = _find_missing_sources(tone_sets)
        for missing in missing_sources:
            findings.append(
                CompatibilityFinding(
                    "missing-audio-source",
                    "Referenced audio source is missing",
                    CompatibilityState.MISSING_SOURCE,
                    missing.path,
                    remediation="Restore the referenced file or intentionally replace it and import again.",
                )
            )

        from .inspector import inspect_audio_source

        missing_paths = {missing.path for missing in missing_sources}
        for referenced_path, labels in _referenced_sources(tone_sets).items():
            if referenced_path in missing_paths:
                continue
            qualification = inspect_audio_source(referenced_path)
            payload = qualification.to_dict()
            payload["referenced_by"] = sorted(labels)
            audio_source_reports.append(payload)

            # Channel character and disclosed resampling belong in the report, but
            # they do not by themselves change an ambient file layer's schedule
            # semantics. Keep those visible as equivalent, not acknowledgement-gated.
            if qualification.state in {CompatibilityState.EQUIVALENT, CompatibilityState.PARTIAL}:
                state = (
                    CompatibilityState.PARTIAL
                    if qualification.clipping_samples
                    else CompatibilityState.EQUIVALENT
                )
                findings.append(
                    CompatibilityFinding(
                        "audio-source-qualification",
                        "Referenced audio source qualification",
                        state,
                        f"{referenced_path}: {qualification.suitability}",
                        remediation=(
                            "Review clipping and headroom before rendering."
                            if state is CompatibilityState.PARTIAL
                            else "Review the recorded source properties when protocol fidelity depends on stereo separation."
                        ),
                    )
                )
            elif qualification.state is CompatibilityState.UNKNOWN:
                findings.append(
                    CompatibilityFinding(
                        "audio-source-unqualified",
                        "Referenced audio source could not be fully qualified",
                        CompatibilityState.PARTIAL,
                        f"{referenced_path}: {qualification.suitability}",
                        remediation="Install a supported decoder or inspect the source manually before relying on its channel behavior.",
                    )
                )

    inferred_duration, end_behavior = _infer_duration_and_end(schedule)
    start_mode = _infer_start_mode(text, schedule)
    if parsed and not schedule:
        findings.append(
            CompatibilityFinding(
                "no-schedule-events",
                "No schedule events were found",
                CompatibilityState.UNSAFE_TO_RENDER,
                "Tone-set definitions exist without a playable timeline.",
                blocking=True,
            )
        )
    elif parsed and inferred_duration is None:
        findings.append(
            CompatibilityFinding(
                "open-ended-schedule",
                "Schedule has no explicit end",
                CompatibilityState.SUPPORTED,
                "Playback can proceed, but file rendering requires an explicit duration.",
                remediation="Choose the intended render duration in the inspector or CLI.",
            )
        )
    if parsed and start_mode.startswith("delayed-relative-or-wall-clock"):
        findings.append(
            CompatibilityFinding(
                "timing-mode-ambiguity",
                "Delayed or wall-clock-looking start requires a playback choice",
                CompatibilityState.PARTIAL,
                "PySbagen preserves the numeric timeline but does not silently decide whether it should follow the current wall clock or play from the beginning.",
                remediation="Inspect the timeline and deliberately choose play-from-start or wall-clock interpretation.",
            )
        )

    report = ImportReport(
        source_path=str(path),
        source_type=source_type,
        source_size=len(data),
        source_sha256=sha256_bytes(data),
        encoding=encoding,
        version_clues=_version_clues(text),
        metadata={
            "tone_set_count": len(tone_sets),
            "schedule_event_count": len(schedule),
            "audio_sources": audio_source_reports,
        },
        findings=findings,
        missing_sources=missing_sources,
        inferred_duration=inferred_duration,
        start_mode=start_mode,
        end_behavior=end_behavior,
        loop_behavior="source-specific; file layers do not loop unless explicitly configured",
        render_disposition=choose_render_disposition(findings, parsed=parsed),
    )
    return ImportedArtifact(report, tone_sets, schedule, text)


def import_drg(path: str | Path, *, preserve_to: str | Path | None = None) -> ImportedArtifact:
    source_path = Path(path).expanduser().resolve()
    package = parse_drg_package(source_path)
    if preserve_to is not None:
        preserve_drg_package(package, preserve_to)

    findings = [
        CompatibilityFinding(
            "drg-package-preserved",
            "DRG package elements retained",
            CompatibilityState.SUPPORTED,
            f"Preserved {len(package.elements)} package elements and immutable source bytes.",
        )
    ]
    for warning in package.warnings:
        state = CompatibilityState.UNSAFE_TO_RENDER if "schedule" in warning.lower() else CompatibilityState.PARTIAL
        findings.append(
            CompatibilityFinding(
                "drg-package-warning",
                "DRG package warning",
                state,
                warning,
                blocking=state is CompatibilityState.UNSAFE_TO_RENDER,
            )
        )

    if package.schedule_text is None:
        report = ImportReport(
            str(source_path),
            "drg",
            len(package.source_bytes),
            package.source_sha256,
            encoding=package.text_encoding,
            version_clues=[package.header] if package.header else [],
            metadata=package.metadata,
            package_elements=package.package_elements(),
            findings=findings,
            render_disposition=RenderDisposition.BLOCKED,
        )
        return ImportedArtifact(report, {}, [], package=package)

    nested = import_sbg_text(
        package.schedule_text,
        source_path=f"{source_path}::schedule.sbg",
        source_bytes=package.schedule_text.encode("utf-8"),
        encoding=str(package.metadata.get("schedule_encoding", "utf-8")),
        source_type="drg-schedule",
        base_dir=source_path.parent,
    )
    findings.extend(nested.report.findings)
    metadata = dict(package.metadata)
    metadata.update(
        {
            "tone_set_count": len(nested.tone_sets),
            "schedule_event_count": len(nested.schedule),
            "has_image": package.image_bytes is not None,
            "element_count": len(package.elements),
            "audio_sources": nested.report.metadata.get("audio_sources", []),
        }
    )
    report = ImportReport(
        str(source_path),
        "drg",
        len(package.source_bytes),
        package.source_sha256,
        encoding=package.text_encoding,
        version_clues=[package.header] if package.header else [],
        metadata=metadata,
        package_elements=package.package_elements(),
        findings=findings,
        missing_sources=nested.report.missing_sources,
        inferred_duration=nested.report.inferred_duration,
        start_mode=nested.report.start_mode,
        end_behavior=nested.report.end_behavior,
        loop_behavior=nested.report.loop_behavior,
        render_disposition=choose_render_disposition(findings, parsed=bool(nested.schedule)),
    )
    return ImportedArtifact(report, nested.tone_sets, nested.schedule, package.schedule_text, package)


def _scan_sbg_source(text: str) -> list[CompatibilityFinding]:
    findings: list[CompatibilityFinding] = []
    labels: dict[str, int] = {}
    recognized = False
    for line_number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        location = (SourceLocation(line=line_number, text=raw),)
        if _is_definition(stripped):
            label, rest = stripped.split(":", 1)
            label = label.strip()
            if label in labels:
                findings.append(
                    CompatibilityFinding(
                        "duplicate-tone-set",
                        "Duplicate tone-set label",
                        CompatibilityState.UNSAFE_TO_RENDER,
                        f"{label!r} was first defined on line {labels[label]} and would otherwise be silently replaced.",
                        location,
                        True,
                    )
                )
            else:
                labels[label] = line_number
            if re.search(r"(?i)(?:^|\s)(spin|slide):", rest):
                findings.append(
                    CompatibilityFinding(
                        "legacy-motion-component",
                        "Legacy spin/slide component is approximated",
                        CompatibilityState.APPROXIMATED,
                        "PySbagen currently renders the underlying tone without original motion semantics.",
                        location,
                        remediation="Review the timeline and explicitly acknowledge the approximation before rendering.",
                    )
                )
            if re.search(r"(?i)(?:^|\s)(iso|hbox):", rest):
                findings.append(
                    CompatibilityFinding(
                        "pysbagen-extension",
                        "PySbagen extension detected",
                        CompatibilityState.SUPPORTED,
                        "Isochronic or Harmonic Box syntax is supported by PySbagen but is not original SBaGen syntax.",
                        location,
                    )
                )
            if "{" in rest or "}" in rest or re.search(r"\[[^\]]+\]", rest):
                findings.append(
                    CompatibilityFinding(
                        "random-range-syntax",
                        "Range or random syntax is not supported",
                        CompatibilityState.UNSUPPORTED,
                        "The construct was preserved in source but cannot be executed faithfully.",
                        location,
                    )
                )
            recognized = True
            continue

        if re.match(r"(?i)^(include|import)\b", stripped) or re.match(r"^-[A-Za-z]", stripped):
            findings.append(
                CompatibilityFinding(
                    "source-directive",
                    "Original command/directive is unsupported",
                    CompatibilityState.UNSUPPORTED,
                    "Command-line or include directives are not executed from imported schedules.",
                    location,
                    remediation="Resolve the directive into an explicit self-contained schedule.",
                )
            )
            continue

        operation = _schedule_operation_text(stripped)
        leading = operation.split(maxsplit=1)[0] if operation else ""
        if leading in {"--", "<>", "<-", "=-", "-="}:
            findings.append(
                CompatibilityFinding(
                    "legacy-transition-operator",
                    "Legacy transition operator is approximated",
                    CompatibilityState.APPROXIMATED,
                    f"Leading operator {leading!r} is normalized to PySbagen replacement semantics; its original envelope behavior is not reproduced.",
                    location,
                )
            )
        elif leading == "==":
            findings.append(
                CompatibilityFinding(
                    "replacement-operator",
                    "Replacement operator normalized",
                    CompatibilityState.EQUIVALENT,
                    "The explicit replacement marker is represented by PySbagen's replacement event model.",
                    location,
                )
            )

    if recognized:
        findings.insert(
            0,
            CompatibilityFinding(
                "core-schedule-syntax",
                "Core schedule syntax recognized",
                CompatibilityState.SUPPORTED,
                "Tone sets, timed events, add/remove operations, silence, and trailing crossfades are inspectable.",
            ),
        )
    return findings


def _is_definition(line: str) -> bool:
    return (
        not (line.upper().startswith("NOW") or line[0].isdigit() or line.startswith("+"))
        and ":" in line
        and not re.match(r"^\d{1,3}:\d", line)
    )


def _schedule_operation_text(line: str) -> str:
    if line.upper().startswith("NOW"):
        return line[3:].strip()
    parts = line.split(maxsplit=1)
    return parts[1].strip() if len(parts) == 2 else ""


def _referenced_sources(tone_sets: dict[str, list[Any]]) -> dict[str, set[str]]:
    references: dict[str, set[str]] = {}
    for label, specs in tone_sets.items():
        for spec in specs:
            path = getattr(spec, "path", None)
            if path:
                references.setdefault(str(path), set()).add(label)
    return references


def _find_missing_sources(tone_sets: dict[str, list[Any]]) -> list[MissingSource]:
    missing: dict[str, set[str]] = {}
    for label, specs in tone_sets.items():
        for spec in specs:
            path = getattr(spec, "path", None)
            if path and not Path(path).is_file():
                missing.setdefault(str(path), set()).add(label)
    return [
        MissingSource(path, tuple(sorted(labels)))
        for path, labels in sorted(missing.items())
    ]


def _infer_duration_and_end(schedule: list[tuple[float, list[str]]]) -> tuple[float | None, str]:
    if not schedule:
        return None, "no-events"
    final_time, tokens = schedule[-1]
    if {token.lower() for token in tokens if token != "->"}.intersection({"off", "alloff", "-"}):
        return float(final_time), "explicit-silence"
    return None, "open-ended"


def _infer_start_mode(text: str, schedule: list[tuple[float, list[str]]]) -> str:
    if any(line.strip().upper().startswith("NOW") for line in text.splitlines()):
        return "play-now"
    if schedule and schedule[0][0] == 0:
        return "relative-zero"
    return "delayed-relative-or-wall-clock; inspect before playback" if schedule else "unknown"


def _version_clues(text: str) -> list[str]:
    lowered = text.lower()
    clues = []
    if "iso:" in lowered or "hbox:" in lowered:
        clues.append("PySbagen extension syntax")
    if "spin:" in lowered or "slide:" in lowered:
        clues.append("original SBaGen motion syntax")
    if "now" in lowered:
        clues.append("NOW-relative schedule")
    return clues


def _decode_text(data: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1"), "latin-1"
