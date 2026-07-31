from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pysbagen.sbagenx_backend import _probe_executable, _probe_library, probe_sbagenx


class _FakeFunction:
    def __init__(self, value):
        self.value = value
        self.restype = None

    def __call__(self):
        return self.value


class _FakeLibrary:
    sbx_version = _FakeFunction(b"3.9.0-alpha.15")
    sbx_api_version = _FakeFunction(47)
    sbx_context_render_f32 = object()
    sbx_validate_sbg_text = object()
    sbx_validate_sbgf_text = object()
    sbx_audio_writer_create_path = object()
    sbx_audio_writer_write_f32 = object()
    sbx_context_set_live_control = object()
    sbx_context_ramp_live_control = object()
    sbx_context_mix_stream_sample = object()


def test_probe_library_reports_version_api_and_symbols():
    version, api_version, capabilities, warning = _probe_library(
        "fake-sbagenxlib",
        loader=lambda _: _FakeLibrary(),
    )

    assert version == "3.9.0-alpha.15"
    assert api_version == 47
    assert warning is None
    assert capabilities
    assert all(capability.available for capability in capabilities)


def test_probe_executable_uses_help_banner(monkeypatch):
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        return SimpleNamespace(
            returncode=0,
            stdout="SBaGenX - Sequenced Brainwave Generator, version 3.9.0-alpha.15\nmore help\n",
            stderr="",
        )

    monkeypatch.setattr("pysbagen.sbagenx_backend.subprocess.run", fake_run)

    version, warning = _probe_executable("sbagenx")

    assert observed["command"] == ["sbagenx", "-h"]
    assert version == "3.9.0-alpha.15"
    assert warning is None


def test_probe_executable_rejects_unrecognized_help_banner(monkeypatch):
    monkeypatch.setattr(
        "pysbagen.sbagenx_backend.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Some unrelated executable\nusage follows\n",
            stderr="",
        ),
    )

    version, warning = _probe_executable("not-sbagenx")

    assert version is None
    assert warning == "Could not parse SBaGenX version from help banner: Some unrelated executable"


def test_probe_can_record_explicit_paths_without_loading(tmp_path: Path):
    executable = tmp_path / "sbagenx"
    executable.write_text("not executed", encoding="utf-8")
    library = tmp_path / "libsbagenx.so"
    library.write_text("not loaded", encoding="utf-8")

    report = probe_sbagenx(
        executable=executable,
        library=library,
        query_executable=False,
        load_library=False,
    )

    assert report.candidate_found
    assert not report.usable
    assert not report.available
    assert report.executable_path == str(executable.resolve())
    assert report.library_path == str(library.resolve())
    assert report.executable_version is None
    assert report.api_version is None


def test_non_executable_candidate_is_not_usable(monkeypatch, tmp_path: Path):
    executable = tmp_path / "sbagenx"
    executable.write_text("not executable", encoding="utf-8")
    monkeypatch.setattr(
        "pysbagen.sbagenx_backend.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("permission denied")),
    )
    monkeypatch.setattr("pysbagen.sbagenx_backend.ctypes.util.find_library", lambda _: None)

    report = probe_sbagenx(executable=executable)

    assert report.candidate_found
    assert not report.usable
    assert any("permission denied" in warning for warning in report.warnings)


def test_unloadable_library_candidate_is_not_usable(monkeypatch, tmp_path: Path):
    library = tmp_path / "libsbagenx.so"
    library.write_text("not a shared library", encoding="utf-8")
    monkeypatch.setattr(
        "pysbagen.sbagenx_backend._probe_library",
        lambda path: (None, None, [], "Could not load SBaGenX library: invalid image"),
    )
    monkeypatch.setattr("pysbagen.sbagenx_backend.shutil.which", lambda _: None)

    report = probe_sbagenx(library=library, query_executable=False)

    assert report.candidate_found
    assert not report.usable
    assert any("invalid image" in warning for warning in report.warnings)


def test_probe_missing_configured_backend_names_the_path(monkeypatch, tmp_path: Path):
    missing = tmp_path / "missing-sbagenx"
    monkeypatch.setattr("pysbagen.sbagenx_backend.shutil.which", lambda _: None)
    monkeypatch.setattr("pysbagen.sbagenx_backend.ctypes.util.find_library", lambda _: None)

    report = probe_sbagenx(executable=missing, query_executable=False, load_library=False)

    assert not report.candidate_found
    assert any(str(missing) in warning for warning in report.warnings)


def test_probe_missing_backend_is_explicit(monkeypatch):
    monkeypatch.delenv("SBAGENX_BIN", raising=False)
    monkeypatch.delenv("SBAGENXLIB_PATH", raising=False)
    monkeypatch.setattr("pysbagen.sbagenx_backend.shutil.which", lambda _: None)
    monkeypatch.setattr("pysbagen.sbagenx_backend.ctypes.util.find_library", lambda _: None)

    report = probe_sbagenx()

    assert not report.available
    assert not report.candidate_found
    assert report.warnings
    assert "existing Python backend remains available" in report.warnings[-1]
