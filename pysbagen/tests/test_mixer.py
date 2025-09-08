import numpy as np
from pysbagen.mixer import mix_generators
from pysbagen.generators.binaural import ToneSpec

def test_mix_generators():
    # Create a couple of tone specs
    spec1 = ToneSpec(base=200, beat=10, amp=50)
    spec2 = ToneSpec(base=300, beat=20, amp=50)

    # Mix them for 1 second
    duration = 1.0
    mixed_generator = mix_generators([spec1, spec2], duration)

    # Get all the chunks and stack them
    chunks = [chunk for chunk, info in mixed_generator]
    audio = np.vstack(chunks)

    # Check the shape of the output
    assert audio.shape[0] == int(44100 * duration)
    assert audio.shape[1] == 2

    # Check the peak level
    peak = np.max(np.abs(audio))
    assert peak <= 1.0
