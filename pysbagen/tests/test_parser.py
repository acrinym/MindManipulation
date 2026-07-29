from pathlib import Path

import pytest

from pysbagen.generators import FileSpec, HarmonicBoxSpec, IsochronicSpec, NoiseSpec, ToneSpec
from pysbagen.parser import parse_sbg_from_string, parse_tone_component


def test_parse_complete_schedule():
    source = """
    # Full component coverage
    alpha: 200+10/50 pink/10 iso:220,8/40 hbox:180,5,8/30
    NOW alpha
    0:10 off
    """
    tone_sets, schedule = parse_sbg_from_string(source)

    assert [type(item) for item in tone_sets["alpha"]] == [
        ToneSpec,
        NoiseSpec,
        IsochronicSpec,
        HarmonicBoxSpec,
    ]
    assert schedule == [(0.0, ["alpha"]), (10.0, ["off"])]


def test_file_component_resolves_paths_with_slashes(tmp_path: Path):
    audio = tmp_path / "soundscapes" / "rain.wav"
    audio.parent.mkdir()
    audio.touch()

    parsed = parse_tone_component("soundscapes/rain.wav/45", base_dir=tmp_path)

    assert isinstance(parsed, FileSpec)
    assert parsed.path == str(audio.resolve())
    assert parsed.amp == 45


def test_quoted_file_path_is_one_component(tmp_path: Path):
    audio = tmp_path / "soft rain.wav"
    audio.touch()
    source = 'rain: "soft rain.wav/35"\nNOW rain\n0:01 off\n'

    tone_sets, _ = parse_sbg_from_string(source, base_dir=tmp_path)

    assert isinstance(tone_sets["rain"][0], FileSpec)
    assert tone_sets["rain"][0].path == str(audio.resolve())


def test_invalid_line_reports_line_number():
    with pytest.raises(ValueError, match="line 2"):
        parse_sbg_from_string("alpha: 200+10\n0:99 alpha")
