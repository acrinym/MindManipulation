from __future__ import annotations

import ctypes
import ctypes.util
import os
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
    def available(self) -> bool:
        return bool(self.executable_path or self.library_path)

    @property
    def native_api_available(self) -> bool:
        return self.api_version is not None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["available"] = self.available
        payload["native_api_available"] = self.native_api_available
        return payload

    def to_text(self) -> str:
        lines = ["SBaGenX backend probe"]
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
    if value is None:
        return None
    text = os.fspath(value).strip()
    return str(Path(text).expanduser()) if text else None


def _find_executable(explicit: str | os.PathLike[str] | None) -> str | None:
    candidate = _normalize_candidate(explicit) or _normalize_candidate(os.getenv("SBAGENX_BIN"))
    if candidate:
        path = Path(candidate)
        if path.is_file():
            return str(path.resolve())
        resolved = shutil.which(candidate)
        return str(Path(resolved).resolve()) if resolved else candidate
    resolved = shutil.which("sbagenx")
    return str(Path(resolved).resolve()) if resolved else None


def _find_library(explicit: str | os.PathLike[str] | None) -> str | None:
    candidate = _normalize_candidate(explicit) or _normalize_candidate(os.getenv("SBAGENXLIB_PATH"))
    if candidate:
        return candidate
    for name in ("sbagenx", "sbagenxlib"):
        discovered = ctypes.util.find_library(name)
        if discovered:
            return discovered
    return None


def _probe_executable(path: str, timeout: float = 3.0) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            [path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"Could not query SBaGenX executable: {exc}"
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        return None, f"SBaGenX --version exited with status {completed.returncode}: {output or 'no output'}"
    return output.splitlines()[0] if output else None, None


def _read_c_string(function: Any) -> str | None:
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
    try:
        library = loader(path)
    except OSError as exc:
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

    report = SBaGenXProbe(
        executable_path=_find_executable(executable),
        library_path=_find_library(library),
    )

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

    if not report.available:
        report.warnings.append(
            "SBaGenX was not found. Install it separately or set SBAGENX_BIN/SBAGENXLIB_PATH; "
            "PySbagen's existing Python backend remains available."
        )
    return report
