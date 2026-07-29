from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import pysbagen.generators.file as file_generator
from pysbagen.generators import FileSpec
from pysbagen.mixer import SR, mix_generators


def stack(generator):
    return np.vstack([chunk for chunk, _ in generator])


def test_file_generator_resamples_and_expands_mono(tmp_path: Path):
    path = tmp_path / "mono.wav"
    source_rate = 22050
    sf.write(path, np.linspace(-0.5, 0.5, source_rate // 10, dtype=np.float32), source_rate)

    audio = stack(mix_generators([FileSpec(path=str(path))], 0.1))

    assert audio.shape == (SR // 10, 2)
    np.testing.assert_allclose(audio[:, 0], audio[:, 1], atol=1e-5)


def test_short_file_is_padded_and_loop_option_repeats(tmp_path: Path):
    path = tmp_path / "short.wav"
    sf.write(path, np.ones((100, 2), dtype=np.float32) * 0.25, SR)

    padded = stack(mix_generators([FileSpec(path=str(path))], 0.01))
    looped = stack(mix_generators([FileSpec(path=str(path), loop=True)], 0.01))

    assert np.allclose(padded[100:], 0)
    assert np.any(looped[100:] != 0)


def test_mp3_requires_ffmpeg_without_importing_legacy_audioop(tmp_path: Path, monkeypatch):
    path = tmp_path / "session.mp3"
    path.write_bytes(b"not-real-mp3")
    monkeypatch.setattr(file_generator.shutil, "which", lambda executable: None)

    with pytest.raises(RuntimeError, match="FFmpeg"):
        FileSpec(path=str(path))._load()


def test_mp3_decoder_reads_ffmpeg_float_output(tmp_path: Path, monkeypatch):
    path = tmp_path / "session.mp3"
    path.write_bytes(b"not-real-mp3")
    expected = np.array([[0.25, -0.25], [0.5, -0.5]], dtype="<f4")
    captured = {}

    class Result:
        returncode = 0
        stdout = expected.tobytes()
        stderr = b""

    def fake_run(command, capture_output, check):
        captured["command"] = command
        assert capture_output is True
        assert check is False
        return Result()

    monkeypatch.setattr(file_generator.shutil, "which", lambda executable: "/usr/bin/ffmpeg")
    monkeypatch.setattr(file_generator.subprocess, "run", fake_run)

    decoded = FileSpec(path=str(path))._load()

    np.testing.assert_array_equal(decoded, expected)
    assert captured["command"][-1] == "pipe:1"
    assert captured["command"][captured["command"].index("-ar") + 1] == str(SR)
