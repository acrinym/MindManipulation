import numpy as np
from pysbagen.generators.binaural import ToneSpec
from pysbagen.generators.noise import NoiseSpec
from pysbagen.generators.file import FileSpec
from pysbagen.generators.isochronic import IsochronicSpec
from pysbagen.generators.harmonic_box import HarmonicBoxSpec
from pysbagen.generators.generic import GenericToneSpec

def test_tone_spec():
    spec = ToneSpec(base=200, beat=10, amp=50)
    duration = 0.1
    generator = spec.generator(duration)

    chunks = [chunk for chunk, info in generator]
    audio = np.vstack(chunks)

    assert audio.shape[0] == int(44100 * duration)
    assert audio.shape[1] == 2
    assert np.max(np.abs(audio)) <= 0.5

def test_noise_spec():
    spec = NoiseSpec(amp=50, kind="white")
    duration = 0.1
    generator = spec.generator(duration)

    chunks = [chunk for chunk, info in generator]
    audio = np.vstack(chunks)

    assert audio.shape[0] == int(44100 * duration)
    assert audio.shape[1] == 2

def test_isochronic_spec():
    spec = IsochronicSpec(freq=200, beat=10, amp=50)
    duration = 0.1
    generator = spec.generator(duration)

    chunks = [chunk for chunk, info in generator]
    audio = np.vstack(chunks)

    assert audio.shape[0] == int(44100 * duration)
    assert audio.shape[1] == 2
    assert np.max(np.abs(audio)) <= 0.5

def test_harmonic_box_spec():
    spec = HarmonicBoxSpec(base=180, diff=5, mod=8, amp=50)
    duration = 0.1
    generator = spec.generator(duration)

    chunks = [chunk for chunk, info in generator]
    audio = np.vstack(chunks)

    assert audio.shape[0] == int(44100 * duration)
    assert audio.shape[1] == 2
    assert np.max(np.abs(audio)) <= 0.5

def test_generic_tone_spec():
    spec = GenericToneSpec(freq=200, amp=50, waveform="square")
    duration = 0.1
    generator = spec.generator(duration)

    chunks = [chunk for chunk, info in generator]
    audio = np.vstack(chunks)

    assert audio.shape[0] == int(44100 * duration)
    assert audio.shape[1] == 2
    assert np.max(np.abs(audio)) <= 0.5
