"""Typed, fail-closed bindings for the reviewed SBaGenX native validation API."""

from __future__ import annotations

import ctypes
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .sbagenx_backend import _find_library

SBX_OK = 0
SBX_API_MIN = 47
SBX_API_MAX = 47
SBX_DIAG_ERROR = 1
SBX_DIAG_WARNING = 2


class SBaGenXNativeError(RuntimeError):
    """Base error for native SBaGenX loading, versioning, or calls."""


class UnsupportedSBaGenXAPI(SBaGenXNativeError):
    """Raised when a native library API revision is outside the qualified range."""


class _SbxDiagnostic(ctypes.Structure):
    """ctypes mirror of SBaGenX API-47 ``SbxDiagnostic``."""

    _fields_ = [
        ("severity", ctypes.c_int),
        ("code", ctypes.c_char * 32),
        ("line", ctypes.c_uint32),
        ("column", ctypes.c_uint32),
        ("end_line", ctypes.c_uint32),
        ("end_column", ctypes.c_uint32),
        ("message", ctypes.c_char * 256),
    ]


@dataclass(frozen=True)
class NativeDiagnostic:
    """One structured diagnostic returned by ``sbagenxlib``."""

    severity: str
    code: str
    line: int
    column: int
    end_line: int
    end_column: int
    message: str


@dataclass
class NativeValidationReport:
    """Versioned SBaGenX validation result for one preserved source artifact."""

    source_path: str
    source_kind: str
    source_encoding: str
    source_size: int
    source_sha256: str
    library_path: str
    library_version: str
    api_version: int
    status_code: int
    diagnostics: list[NativeDiagnostic] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return whether validation completed with no error-severity diagnostics."""

        return self.status_code == SBX_OK and not any(item.severity == "error" for item in self.diagnostics)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report for manifests and JSON CLI output."""

        payload = asdict(self)
        payload["valid"] = self.valid
        return payload

    def to_text(self) -> str:
        """Render a concise human-readable validation report."""

        lines = [
            f"SBaGenX native validation: {self.source_path}",
            f"Kind: {self.source_kind}",
            f"Encoding: {self.source_encoding}",
            f"Source bytes: {self.source_size}",
            f"Source SHA-256: {self.source_sha256}",
            f"Library: {self.library_path}",
            f"Version/API: {self.library_version} / {self.api_version}",
            f"Valid: {'yes' if self.valid else 'no'}",
        ]
        if not self.diagnostics:
            lines.append("Diagnostics: none")
        else:
            lines.append("Diagnostics:")
            for item in self.diagnostics:
                location = f"{item.line}:{item.column}" if item.line else "unknown"
                code = f" [{item.code}]" if item.code else ""
                lines.append(f"  {item.severity} {location}{code}: {item.message}")
        return "\n".join(lines)


class SBaGenXNative:
    """Narrow API-47 native binding used only for qualified operations."""

    def __init__(
        self,
        library_path: str,
        *,
        loader: Callable[[str], Any] = ctypes.CDLL,
        api_min: int = SBX_API_MIN,
        api_max: int = SBX_API_MAX,
    ) -> None:
        """Load, type, and version-gate one SBaGenX shared library."""

        self.library_path = library_path
        try:
            self._library = loader(library_path)
        except (OSError, TypeError) as exc:
            raise SBaGenXNativeError(f"Could not load SBaGenX library {library_path}: {exc}") from exc

        self._version = self._required("sbx_version")
        self._version.argtypes = []
        self._version.restype = ctypes.c_char_p
        self._api_version = self._required("sbx_api_version")
        self._api_version.argtypes = []
        self._api_version.restype = ctypes.c_int
        self._validate_sbg = self._bind_validator("sbx_validate_sbg_text")
        self._validate_sbgf = self._bind_validator("sbx_validate_sbgf_text")
        self._free_diagnostics = self._required("sbx_free_diagnostics")
        self._free_diagnostics.argtypes = [ctypes.POINTER(_SbxDiagnostic)]
        self._free_diagnostics.restype = None

        version_value = self._version()
        if not version_value:
            raise SBaGenXNativeError("SBaGenX library returned an empty sbx_version value")
        self.version = _decode_c_value(version_value)
        self.api_version = int(self._api_version())
        if not api_min <= self.api_version <= api_max:
            raise UnsupportedSBaGenXAPI(
                f"Unsupported SBaGenX API {self.api_version}; qualified range is {api_min}..{api_max}"
            )

    def _required(self, name: str) -> Any:
        """Return one required exported symbol or fail with a precise message."""

        try:
            return getattr(self._library, name)
        except AttributeError as exc:
            raise SBaGenXNativeError(f"SBaGenX library is missing required symbol: {name}") from exc

    def _bind_validator(self, name: str) -> Any:
        """Apply the exact API-47 signature to one validation function."""

        function = self._required(name)
        function.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.POINTER(_SbxDiagnostic)),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        function.restype = ctypes.c_int
        return function

    def validate_text(self, text: str, source_name: str, source_kind: str) -> tuple[int, list[NativeDiagnostic]]:
        """Validate UTF-8 text as ``sbg`` or ``sbgf`` and always free native diagnostics."""

        if source_kind not in {"sbg", "sbgf"}:
            raise ValueError(f"Unsupported SBaGenX source kind: {source_kind}")
        validator = self._validate_sbg if source_kind == "sbg" else self._validate_sbgf
        diagnostics = ctypes.POINTER(_SbxDiagnostic)()
        count = ctypes.c_size_t(0)
        status = int(
            validator(
                text.encode("utf-8"),
                source_name.encode("utf-8", errors="replace"),
                ctypes.byref(diagnostics),
                ctypes.byref(count),
            )
        )
        items: list[NativeDiagnostic] = []
        try:
            if count.value and not diagnostics:
                raise SBaGenXNativeError(
                    f"SBaGenX returned {count.value} diagnostics with a null diagnostic pointer"
                )
            for index in range(count.value):
                raw = diagnostics[index]
                items.append(
                    NativeDiagnostic(
                        severity={SBX_DIAG_ERROR: "error", SBX_DIAG_WARNING: "warning"}.get(
                            int(raw.severity), f"unknown-{int(raw.severity)}"
                        ),
                        code=_decode_c_value(raw.code),
                        line=int(raw.line),
                        column=int(raw.column),
                        end_line=int(raw.end_line),
                        end_column=int(raw.end_column),
                        message=_decode_c_value(raw.message),
                    )
                )
        finally:
            if diagnostics:
                self._free_diagnostics(diagnostics)
        if status != SBX_OK:
            raise SBaGenXNativeError(f"SBaGenX validation call failed with status {status}")
        return status, items


def _decode_c_value(value: Any) -> str:
    """Decode bytes or fixed ctypes character arrays without trailing NULs."""

    if isinstance(value, bytes):
        raw = value
    else:
        raw = bytes(value)
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def _read_source(path: Path) -> tuple[str, str, bytes]:
    """Read source bytes and decode with UTF-8 BOM, UTF-8, then Latin-1 fallback."""

    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig"), "utf-8-sig", data
    try:
        return data.decode("utf-8"), "utf-8", data
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1", data


def validate_sbagenx_source(
    source: str | Path,
    *,
    library: str | Path | None = None,
    loader: Callable[[str], Any] = ctypes.CDLL,
) -> NativeValidationReport:
    """Discover a native library and validate one ``.sbg`` or ``.sbgf`` file."""

    path = Path(source).expanduser().resolve()
    source_kind = path.suffix.lower().lstrip(".")
    if source_kind not in {"sbg", "sbgf"}:
        raise ValueError("Native SBaGenX validation requires a .sbg or .sbgf source")
    library_path = _find_library(library)
    if not library_path:
        raise SBaGenXNativeError(
            "SBaGenX native library was not found; set SBAGENXLIB_PATH or pass --library"
        )
    text, encoding, source_bytes = _read_source(path)
    native = SBaGenXNative(library_path, loader=loader)
    status, diagnostics = native.validate_text(text, str(path), source_kind)
    return NativeValidationReport(
        source_path=str(path),
        source_kind=source_kind,
        source_encoding=encoding,
        source_size=len(source_bytes),
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        library_path=library_path,
        library_version=native.version,
        api_version=native.api_version,
        status_code=status,
        diagnostics=diagnostics,
    )
