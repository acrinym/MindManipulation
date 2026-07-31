"""Optional, version-aware discovery for locally installed SBaGenX runtimes."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class BackendCapability:
    """One capability observed on an installed SBaGenX backend."""

    name: str
    available: bool
    evidence: str


@dataclass
class SBaGenXProbe:
    """Discovery and capability result for an optional SBaGenX installation."""

    executable_path: str | None = None
    executable_version: str | None = None
    library_path: str | None = None
    library_version: str | None = None
    api_version: int | None = None
    capabilities: list[BackendCapability] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def candidate_found(self) -> bool:
        """Return whether discovery found an executable or library candidate."""

        return bool(self.executable_path or self.library_path)

    @property
    def native_api_available(self) -> bool:
        """Return whether a native library exposed a readable API revision."""

        return self.api_version is not None

    @property
    def usable(self) -> bool:
        """Return whether at least one candidate passed identity qualification."""

        return bool(self.executable_version or self.native_api_available)

    @property
    def available(self) -> bool:
        """Return a backward-friendly alias for a qualified usable backend."""

        return self.usable

    def to_dict(self) -> dict[str, Any]:
        """Serialize the probe result for deterministic JSON output."""

        payload = asdict(self)
        payload["candidate_found"] = self.candidate_found
        payload["native_api_available"] = self.native_api_available
        payload["usable"] = self.usable
        payload["available"] = self.available
        return payload

    def to_text(self) -> str:
        """Render a compact human-readable qualification report."""

        lines = ["SBaGenX backend probe"]
        lines.append(f"Candidate found: {'yes' if self.candidate_found else 'no'}")
        lines.append(f"Usable backend: {'yes' if self.usable else 'no'}")
        lines.append(f"Executable: {self.executable_path or 'not found'}")
        if self.executable_version:
            lines.append(f"Executable version: {self.executable_version}")
        lines.append(f"Native library: {self.library_path or 'not found'}")
        if self.library_version:
            lines.append(f"Library version: {self.library_version}")
        if self.api_version is not None:
            lines.append(f"Library API: {self.api_version}")
        if self.capabilities:
            lines.append("Capabilities:")
            for capability in self.capabilities:
                marker = "yes" if capability.available else "no"
                lines.append(f"  {capability.name}: {marker} ({capability.evidence})")
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {warning}" for warning in self.warnings)
        return "\n".join(lines)


_CAPABILITY_SYMBOLS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("native_float_render", ("sbx_context_render_f32",)),
    ("sbg_validation", ("sbx_validate_sbg_text",)),
    ("sbgf_validation", ("sbx_validate_sbgf_text",)),
    ("container_writing", ("sbx_audio_writer_create_path", "sbx_audio_writer_write_f32")),
    ("live_parameter_control", ("sbx_context_set_live_control", "sbx_context_ramp_live_control")),
    ("mix_stream_processing", ("sbx_context_mix_stream_sample",)),
)


def _normalize_candidate(value: str | os.PathLike[str] | None) -> str | None:
    """Normalize an optional path/name while preserving loader-friendly names."""

    if value is None:
        return None
    text = os.fspath(value).strip()
    return str(Path(text).expanduser()) if text else None


def _find_executable(explicit: str | os.PathLike[str] | None) -> str | None:
    """Resolve an explicit/environment/PATH SBaGenX executable candidate."""

    candidate = _normalize_candidate(explicit) or _normalize_candidate(os.getenv("SBAGENX_BIN"))
    if candidate:
        path = Path(candidate)
        if path.is_file():
            return str(path.resolve())
        resolved = shutil.which(candidate)
        return str(Path(resolved).resolve()) if resolved else None
    resolved = shutil.which("sbagenx")
    return str(Path(resolved).resolve()) if resolved else None


def _find_library(explicit: str | os.PathLike[str] | None) -> str | None:
    """Resolve an explicit/environment/system SBaGenX shared-library candidate."""

    candidate = _normalize_candidate(explicit) or _normalize_candidate(os.getenv("SBAGENXLIB_PATH"))
    if candidate:
        path = Path(candidate)
        if path.is_absolute() or path.parent != Path("."):
            return str(path.resolve()) if path.is_file() else None
        return candidate
    for name in ("sbagenx", "sbagenxlib"):
        discovered = ctypes.util.find_library(name)
        if discovered:
            return discovered
    return None


def _probe_executable(path: str, timeout: float = 3.0) -> tuple[str | None, str | None]:
    """Read the version from SBaGenX's stable ``-h`` banner.

    SBaGenX uses ``-V`` for master volume and does not publish a ``--version``
    interface in the reviewed source. The first help line contains the version.
    """

    try:
        completed = subprocess.run(
            [path, "-h"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"Could not query SBaGenX executable: {exc}"
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    if completed.returncode != 0:
        return None, f"SBaGenX -h exited with status {completed.returncode}: {output or 'no output'}"
    if not output:
        return None, "SBaGenX -h returned no identity banner"
    first_line = output.splitlines()[0].strip()
    match = re.search(r"\bversion\s+([^\s]+)", first_line, flags=re.IGNORECASE)
    return (match.group(1) if match else first_line), None


def _read_c_string(function: Any) -> str | None:
    """Call a no-argument C function returning a UTF-8-compatible string."""

    function.restype = ctypes.c_char_p
    value = function()
    if not value:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _probe_library(
    path: str,
    loader: Callable[[str], Any] = ctypes.CDLL,
) -> tuple[str | None, int | None, list[BackendCapability], str | None]:
    """Load a candidate library and report identity plus required symbols."""

    try:
        library = loader(path)
    except (OSError, TypeError) as exc:
        return None, None, [], f"Could not load SBaGenX library: {exc}"

    library_version: str | None = None
    api_version: int | None = None
    warnings: list[str] = []

    try:
        library_version = _read_c_string(getattr(library, "sbx_version"))
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        warnings.append(f"Could not read sbx_version: {exc}")

    try:
        function = getattr(library, "sbx_api_version")
        function.restype = ctypes.c_int
        api_version = int(function())
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        warnings.append(f"Could not read sbx_api_version: {exc}")

    capabilities = []
    for name, symbols in _CAPABILITY_SYMBOLS:
        missing = [symbol for symbol in symbols if not hasattr(library, symbol)]
        capabilities.append(
            BackendCapability(
                name=name,
                available=not missing,
                evidence="symbols present" if not missing else f"missing {', '.join(missing)}",
            )
        )

    return library_version, api_version, capabilities, "; ".join(warnings) if warnings else None


def probe_sbagenx(
    *,
    executable: str | os.PathLike[str] | None = None,
    library: str | os.PathLike[str] | None = None,
    query_executable: bool = True,
    load_library: bool = True,
) -> SBaGenXProbe:
    """Discover an optional SBaGenX installation without making it mandatory.

    Environment overrides:

    - ``SBAGENX_BIN`` points at the CLI executable.
    - ``SBAGENXLIB_PATH`` points at the shared library.

    Discovery never changes PySbagen's rendering policy. Later integration
    layers must still version-gate every native operation and attach the
    selected backend to provenance/render receipts.
    """

    requested_executable = _normalize_candidate(executable) or _normalize_candidate(os.getenv("SBAGENX_BIN"))
    requested_library = _normalize_candidate(library) or _normalize_candidate(os.getenv("SBAGENXLIB_PATH"))
    report = SBaGenXProbe(
        executable_path=_find_executable(executable),
        library_path=_find_library(library),
    )

    if requested_executable and not report.executable_path:
        report.warnings.append(f"Configured SBaGenX executable was not found: {requested_executable}")
    if requested_library and not report.library_path:
        report.warnings.append(f"Configured SBaGenX library was not found: {requested_library}")

    if report.executable_path and query_executable:
        version, warning = _probe_executable(report.executable_path)
        report.executable_version = version
        if warning:
            report.warnings.append(warning)

    if report.library_path and load_library:
        version, api_version, capabilities, warning = _probe_library(report.library_path)
        report.library_version = version
        report.api_version = api_version
        report.capabilities = capabilities
        if warning:
            report.warnings.append(warning)

    if not report.candidate_found:
        report.warnings.append(
            "SBaGenX was not found. Install it separately or set SBAGENX_BIN/SBAGENXLIB_PATH; "
            "PySbagen's existing Python backend remains available."
        )
    elif (query_executable or load_library) and not report.usable:
        report.warnings.append("SBaGenX candidates were found, but none passed identity qualification.")
    return report
