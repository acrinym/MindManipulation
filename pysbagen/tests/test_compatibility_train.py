from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from drg_decoder import rc4_decrypt
from pysbagen.api import render_schedule, write_audio
from pysbagen.compatibility import CompatibilityState, RenderDisposition
from pysbagen.drg import parse_drg_package, preserve_drg_package
from pysbagen.importers import import_drg, import_sbg, import_sbg_text
from pysbagen.inspector import build_timeline, inspect_audio_source, qualify_audio_path
from pysbagen.library import LocalLibrary
from pysbagen.matrix import load_compatibility_matrix


def _closed_schedule() -> str:
    return "alpha: 200+10/20 pink/5\nNOW alpha\n0:01 off\n"


def _make_drg(path: Path, schedule: str) -> Path:
    image = b"\x89PNG\r\n\x1a\nsynthetic"
    elements = [
        base64.b64encode(b"title=synthetic fixture").decode("ascii"),
        base64.b64encode(rc4_decrypt(base64.b64encode(image))).decode("ascii"),
        base64.b64encode(b"platform=test\nvariant=desktop").decode("ascii"),
        base64.b64encode(rc4_decrypt(schedule.encode("utf-8"))).decode("ascii"),
        base64.b64encode(b"opaque-extra").decode("ascii"),
    ]
    path.write_text("I-Doser synthetic fixture\n@" + "@".join(elements), encoding="latin-1")
    return path


def test_supported_sbg_has_honest_report_and_timeline(tmp_path: Path):
    source = tmp_path / "supported.sbg"
    source.write_text(_closed_schedule(), encoding="utf-8")
    artifact = import_sbg(source)
    assert artifact.report.render_disposition is RenderDisposition.SAFE
    assert artifact.report.inferred_duration == 1.0
    assert artifact.report.source_sha256
    assert artifact.report.schema_version == "pysbagen.import-report.v1"
    payload = artifact.report.to_dict()
    assert payload["render_disposition"] == "safe"
    assert payload["findings"][0]["state"] == "supported"
    timeline = build_timeline(artifact)
    assert timeline[0].active_tone_sets == ("alpha",)
    assert timeline[-1].active_tone_sets == ()


def test_open_ended_schedule_requires_explicit_render_duration():
    artifact = import_sbg_text("alpha: 200+10/20\nNOW alpha\n", source_path="open.sbg")
    assert artifact.report.render_disposition is RenderDisposition.SAFE
    assert artifact.report.inferred_duration is None
    with pytest.raises(ValueError, match="explicit render duration"):
        artifact.require_duration(None)
    assert artifact.require_duration(12) == 12


def test_approximated_motion_requires_acknowledgement(tmp_path: Path):
    source = tmp_path / "motion.sbg"
    source.write_text("alpha: slide:200+10/20\nNOW alpha\n0:01 off\n", encoding="utf-8")
    artifact = import_sbg(source)
    assert artifact.report.render_disposition is RenderDisposition.SAFE_WITH_DISCLOSED_CHANGES
    assert any(finding.state is CompatibilityState.APPROXIMATED for finding in artifact.report.findings)
    with pytest.raises(ValueError, match="explicitly allow"):
        artifact.report.require_renderable()
    artifact.report.require_renderable(allow_disclosed_changes=True)


def test_duplicate_label_blocks_instead_of_overwriting():
    artifact = import_sbg_text(
        "alpha: 200+10\nalpha: 210+8\nNOW alpha\n0:01 off\n",
        source_path="duplicate.sbg",
    )
    assert artifact.report.render_disposition is RenderDisposition.BLOCKED
    assert any(finding.code == "duplicate-tone-set" for finding in artifact.report.findings)


def test_missing_source_remains_visible_and_inspection_only(tmp_path: Path):
    source = tmp_path / "missing.sbg"
    source.write_text('bed: "not-here.wav/20"\nNOW bed\n0:01 off\n', encoding="utf-8")
    artifact = import_sbg(source)
    assert artifact.report.render_disposition is RenderDisposition.INSPECTION_ONLY
    assert artifact.report.missing_sources
    assert artifact.report.missing_sources[0].path.endswith("not-here.wav")


def test_drg_preserves_all_elements_and_nested_schedule(tmp_path: Path):
    source = _make_drg(tmp_path / "fixture.drg", _closed_schedule())
    package = parse_drg_package(source)
    assert package.schedule_text == _closed_schedule()
    assert package.image_bytes and package.image_bytes.startswith(b"\x89PNG")
    assert len(package.elements) == 6

    preserved = preserve_drg_package(package, tmp_path / "preserved")
    manifest = json.loads((preserved / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["sha256"] == package.source_sha256
    assert len(manifest["elements"]) == len(package.elements)
    assert (preserved / "schedule.sbg").is_file()

    artifact = import_drg(source)
    assert artifact.report.source_type == "drg"
    assert artifact.report.render_disposition is RenderDisposition.SAFE
    assert len(artifact.report.package_elements) == len(package.elements)


def test_rendered_schedule_writes_compatibility_sidecar(tmp_path: Path):
    source = tmp_path / "render.sbg"
    output = tmp_path / "render.wav"
    source.write_text(_closed_schedule(), encoding="utf-8")
    result = write_audio(render_schedule(source), output)
    assert result.outfile == output.resolve()
    assert result.manifest and result.manifest.is_file()
    payload = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert payload["source_import_report"]["render_disposition"] == "safe"
    assert payload["output"]["sha256"]


def test_local_library_is_offline_verifiable_and_keeps_provenance(tmp_path: Path):
    first = tmp_path / "one.sbg"
    second = tmp_path / "same-bytes.sbg"
    first.write_text(_closed_schedule(), encoding="utf-8")
    second.write_text(_closed_schedule(), encoding="utf-8")
    library = LocalLibrary(tmp_path / "library")
    item = library.add(import_sbg(first))
    same_item = library.add(import_sbg(second))
    assert same_item.item_id == item.item_id
    assert len(same_item.manifest["provenance"]["records"]) == 2
    verification = library.verify(item.item_id)
    assert verification["valid"]
    exported = library.export_manifest(item.item_id, tmp_path / "export.json")
    assert json.loads(exported.read_text(encoding="utf-8"))["verification"]["valid"]


def test_audio_source_and_listening_path_qualification(tmp_path: Path):
    source = tmp_path / "stereo.wav"
    samples = np.zeros((4410, 2), dtype=np.float32)
    samples[:, 0] = np.linspace(-0.25, 0.25, len(samples), dtype=np.float32)
    samples[:, 1] = -samples[:, 0]
    sf.write(source, samples, 44100)
    report = inspect_audio_source(source)
    assert report.channels == 2
    assert report.sample_rate == 44100
    assert report.state in {CompatibilityState.SUPPORTED, CompatibilityState.PARTIAL}

    blocked = qualify_audio_path(method="binaural", route="speakers", channels=2, sample_rate=44100)
    assert blocked.state is CompatibilityState.UNSAFE_TO_RENDER
    assert not blocked.safe_to_start

    disclosed = qualify_audio_path(method="binaural", route="headphones", channels=2, sample_rate=44100, bluetooth=True)
    assert disclosed.state is CompatibilityState.PARTIAL
    assert disclosed.safe_to_start


def test_matrix_is_machine_readable_and_documented():
    matrix = load_compatibility_matrix()
    rows = matrix["rows"]
    assert rows
    states = {state.value for state in CompatibilityState}
    assert all(row["state"] in states and row["fixtures"] for row in rows)
    document = Path("docs/compatibility/SBAGEN_SEMANTIC_COMPATIBILITY_MATRIX.md").read_text(encoding="utf-8")
    assert all(row["id"] in document for row in rows)
