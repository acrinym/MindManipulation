from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path

import pytest

from pysbagen.sbagenx_native import (
    SBaGenXNative,
    SBaGenXNativeError,
    UnsupportedSBaGenXAPI,
    _SbxDiagnostic,
    validate_sbagenx_source,
)


class _FakeFunction:
    def __init__(self, implementation):
        self.implementation = implementation
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.implementation(*args)


class _FakeLibrary:
    def __init__(
        self,
        api_version: int = 47,
        with_diagnostic: bool = True,
        null_diagnostic_pointer: bool = False,
    ):
        self._diagnostic_array = None
        self._null_diagnostic_pointer = null_diagnostic_pointer
        self.sbx_version = _FakeFunction(lambda: b"3.9.0-alpha.15")
        self.sbx_api_version = _FakeFunction(lambda: api_version)
        self.sbx_validate_sbg_text = _FakeFunction(
            lambda text, source, out_diags, out_count: self._validate(out_diags, out_count, with_diagnostic)
        )
        self.sbx_validate_sbgf_text = _FakeFunction(
            lambda text, source, out_diags, out_count: self._validate(out_diags, out_count, with_diagnostic)
        )
        self.sbx_free_diagnostics = _FakeFunction(lambda diagnostics: None)

    def _validate(self, out_diags, out_count, with_diagnostic):
        count_pointer = ctypes.cast(out_count, ctypes.POINTER(ctypes.c_size_t))
        diagnostic_pointer = ctypes.cast(
            out_diags,
            ctypes.POINTER(ctypes.POINTER(_SbxDiagnostic)),
        )
        if not with_diagnostic:
            count_pointer[0] = 0
            return 0
        count_pointer[0] = 1
        if self._null_diagnostic_pointer:
            return 0
        array = (_SbxDiagnostic * 1)()
        array[0].severity = 2
        array[0].code = b"native-warning"
        array[0].line = 4
        array[0].column = 2
        array[0].end_line = 4
        array[0].end_column = 3
        array[0].message = b"Synthetic native diagnostic"
        self._diagnostic_array = array
        diagnostic_pointer[0] = ctypes.cast(array, ctypes.POINTER(_SbxDiagnostic))
        return 0


def test_native_binding_types_and_decodes_diagnostics():
    native = SBaGenXNative("fake", loader=lambda _: _FakeLibrary())

    status, diagnostics = native.validate_text("NOW tone", "fixture.sbg", "sbg")

    assert status == 0
    assert native.version == "3.9.0-alpha.15"
    assert native.api_version == 47
    assert diagnostics[0].severity == "warning"
    assert diagnostics[0].code == "native-warning"
    assert diagnostics[0].line == 4
    assert diagnostics[0].message == "Synthetic native diagnostic"


def test_native_binding_rejects_unknown_api_revision():
    with pytest.raises(UnsupportedSBaGenXAPI, match="Unsupported SBaGenX API 48"):
        SBaGenXNative("fake", loader=lambda _: _FakeLibrary(api_version=48))


def test_native_binding_rejects_null_diagnostic_pointer():
    native = SBaGenXNative(
        "fake",
        loader=lambda _: _FakeLibrary(null_diagnostic_pointer=True),
    )

    with pytest.raises(SBaGenXNativeError, match="null diagnostic pointer"):
        native.validate_text("NOW tone", "fixture.sbg", "sbg")


def test_validate_source_preserves_encoding_and_identity(tmp_path: Path):
    source = tmp_path / "latin1.sbg"
    source_bytes = "# café\nNOW off\n".encode("latin-1")
    source.write_bytes(source_bytes)

    report = validate_sbagenx_source(
        source,
        library="fake-library",
        loader=lambda _: _FakeLibrary(with_diagnostic=False),
    )

    assert report.valid
    assert report.source_kind == "sbg"
    assert report.source_encoding == "latin-1"
    assert report.source_size == len(source_bytes)
    assert report.source_sha256 == hashlib.sha256(source_bytes).hexdigest()
    assert report.api_version == 47
    assert report.library_version == "3.9.0-alpha.15"
    assert report.diagnostics == []
