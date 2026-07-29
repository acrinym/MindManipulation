from pathlib import Path

import soundfile as sf

from pysbagen.cli import main
from pysbagen.mixer import SR


def test_cli_writes_streamed_quick_session(tmp_path: Path):
    output = tmp_path / "quick.wav"

    result = main([
        "--base", "200",
        "--beat", "10",
        "--duration", "0.1",
        "--outfile", str(output),
    ])

    audio, rate = sf.read(output, always_2d=True)
    assert result == 0
    assert rate == SR
    assert audio.shape == (SR // 10, 2)


def test_cli_schedule_uses_schedule_directory_for_audio(tmp_path: Path):
    background = tmp_path / "background.wav"
    sf.write(background, [[0.25, 0.25]] * 100, SR)
    schedule = tmp_path / "session.sbg"
    schedule.write_text("bed: background.wav/50\nNOW bed\n0:01 off\n", encoding="latin-1")
    output = tmp_path / "scheduled.wav"

    result = main([str(schedule), "--outfile", str(output)])

    audio, rate = sf.read(output, always_2d=True)
    assert result == 0
    assert rate == SR
    assert audio.shape == (SR, 2)
