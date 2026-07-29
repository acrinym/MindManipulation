from pathlib import Path

import numpy as np
import soundfile as sf

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
