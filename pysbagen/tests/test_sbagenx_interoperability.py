from __future__ import annotations

import ctypes
from pathlib import Path

from pysbagen.compatibility import RenderDisposition
from pysbagen.interoperability import inspect_with_sbagenx
from pysbagen.library import LocalLibrary
from pysbagen.sbagenx_native import _SbxDiagnostic
from pysbagen.sbgf import import_sbgf


class _FakeFunction:
    def __init__(self, implementation):
        self.implementation = implementation
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.implementation(*args)


class _ValidationLibrary:
    def __init__(self, *, error: bool = False):
        self._array = None
        self._error = error
        self.sbx_version = _FakeFunction(lambda: b"3.9.0-alpha.15")
        self.sbx_api_version = _FakeFunction(lambda: 47)
        self.sbx_validate_sbg_text = _FakeFunction(self._validate)
        self.sbx_validate_sbgf_text = _FakeFunction(self._validate)
        self.sbx_free_diagnostics = _FakeFunction(lambda pointer: None)

    def _validate(self, text, source, out_diags, out_count):
        count = ctypes.cast(out_count, ctypes.POINTER(ctypes.c_size_t))
        pointer = ctypes.cast(out_diags, ctypes.POINTER(ctypes.POINTER(_SbxDiagnostic)))
        if not self._error:
            count[0] = 0
            return 0
        array = (_SbxDiagnostic * 1)()
        array[0].severity = 1
        array[0].code = b"native-error"
        array[0].line = 2
        array[0].column = 1
        array[0].end_line = 2
        array[0].end_column = 4
        array[0].message = b"Synthetic native rejection"
        self._array = array
        pointer[0] = ctypes.cast(array, ctypes.POINTER(_SbxDiagnostic))
        count[0] = 1
        return 0


def test_sbgf_preserves_structure_identity_and_library_record(tmp_path: Path):
    source = tmp_path / "curve.sbgf"
    source.write_text(
        "# function program\n"
        "param l = 0.125\n"
        "param h = 0\n"
        "solve target = 6\n"
        "beat = b0 + (b1 - b0) * tanh(l * (m - h))\n"
        "carrier = c0 + (c1 - c0) * ramp(m, 0, T)\n",
        encoding="utf-8",
    )

    artifact = import_sbgf(source)

    assert artifact.report.source_type == "sbgf"
    assert artifact.report.render_disposition is RenderDisposition.INSPECTION_ONLY
    assert artifact.report.metadata["parameter_count"] == 2
    assert artifact.report.metadata["assignment_count"] == 2
    assert artifact.report.metadata["expression_functions"] == ["ramp", "tanh"]
    assert artifact.report.metadata["solve_directives"][0]["line"] == 4

    item = LocalLibrary(tmp_path / "library").add(artifact)
    verification = LocalLibrary(tmp_path / "library").verify(item.item_id)
    assert item.state == "incompatible"
    assert verification["valid"]


def test_sbgf_missing_media_remains_visible(tmp_path: Path):
    source = tmp_path / "media.sbgf"
    source.write_text('param bed = "missing.flac"\nbeat = 8\n', encoding="utf-8")

    artifact = import_sbgf(source)

    assert artifact.report.missing_sources
    assert artifact.report.metadata["referenced_media"][0]["present"] is False


def test_dual_sbgf_report_keeps_native_validity_and_pysbagen_limits(tmp_path: Path):
    source = tmp_path / "curve.sbgf"
    source.write_text("param l = 0.125\nbeat = tanh(l * m)\n", encoding="utf-8")

    report = inspect_with_sbagenx(
        source,
        library="fake-library",
        loader=lambda _: _ValidationLibrary(),
    )

    assert report.source_identity_matches
    assert report.native_valid
    assert report.pysbagen_disposition is RenderDisposition.INSPECTION_ONLY
    assert any(item.code == "native-accepts-pysbagen-limited" for item in report.discrepancies)


def test_dual_sbg_report_surfaces_native_rejection(tmp_path: Path):
    source = tmp_path / "session.sbg"
    source.write_text(
        "tone: 200+10/20\n"
        "off: -\n"
        "NOW tone\n"
        "+00:00:01 off\n",
        encoding="utf-8",
    )

    report = inspect_with_sbagenx(
        source,
        library="fake-library",
        loader=lambda _: _ValidationLibrary(error=True),
    )

    assert report.source_identity_matches
    assert not report.native_valid
    assert any(item.code == "pysbagen-accepts-native-rejects" for item in report.discrepancies)
    assert any(item.code == "native-diagnostics-present" for item in report.discrepancies)
