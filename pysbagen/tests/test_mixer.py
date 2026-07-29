import numpy as np
import pytest
from pysbagen.generators import ToneSpec
from pysbagen.mixer import SR, _apply_schedule_event, build_session_generator, mix_generators

class ShortGenerator:
    sample_rate = SR
    def generator(self, duration):
        yield np.ones((100, 1), dtype=np.float32), {"type": "short"}

def stack(generator):
    return np.vstack([chunk for chunk, _ in generator])

def test_mix_generators_has_exact_duration():
    audio = stack(mix_generators([ToneSpec(base=200, beat=10, amp=50)], 1.125))
    assert audio.shape == (int(SR * 1.125), 2)
    assert np.max(np.abs(audio)) <= 1.0

def test_exhausted_stream_does_not_truncate_other_generators():
    audio = stack(mix_generators([ShortGenerator(), ToneSpec()], 0.1))
    assert audio.shape == (int(SR * 0.1), 2)
    assert np.any(audio[100:] != 0)

def test_empty_mix_preserves_silent_timeline():
    audio = stack(mix_generators([], 0.25))
    assert audio.shape == (int(SR * 0.25), 2)
    assert np.all(audio == 0)

def test_schedule_preserves_leading_silence_and_honors_duration():
    tone = ToneSpec(base=200, beat=10)
    schedule = [(1.0, ["alpha"]), (3.0, ["off"])]
    audio = stack(build_session_generator({"alpha": [tone]}, schedule, duration=2.0))
    assert audio.shape == (int(SR * 2.0), 2)
    assert np.all(audio[:SR] == 0)
    assert np.any(audio[SR:] != 0)

def test_unknown_schedule_name_fails_loudly():
    with pytest.raises(ValueError, match="unknown tone set"):
        list(build_session_generator({}, [(0, ["missing"]), (1, ["off"])], duration=1))

def test_relative_removal_uses_identity_not_dataclass_equality():
    first = ToneSpec(base=200, beat=10)
    equal_but_distinct = ToneSpec(base=200, beat=10)
    tone_sets = {"first": [first], "second": [equal_but_distinct]}
    active = _apply_schedule_event([], tone_sets, ["first", "+second"])
    active = _apply_schedule_event(active, tone_sets, ["-first"])
    assert active == [equal_but_distinct]
    assert active[0] is equal_but_distinct


class CountingGenerator:
    sample_rate = SR

    def __init__(self, value=0.1):
        self.value = value
        self.calls = 0

    def generator(self, duration):
        self.calls += 1
        total = int(SR * duration)
        for start in range(0, total, 128):
            count = min(128, total - start)
            yield np.full((count, 2), self.value, dtype=np.float32), {"type": "counting"}


def test_schedule_reuses_still_active_streams_across_events():
    bed = CountingGenerator(0.1)
    overlay = CountingGenerator(0.05)
    schedule = [
        (0.0, ["bed"]),
        (0.01, ["+overlay"]),
        (0.02, ["-overlay"]),
        (0.03, ["off"]),
    ]
    audio = stack(build_session_generator({"bed": [bed], "overlay": [overlay]}, schedule, duration=0.03))
    assert audio.shape == (int(SR * 0.03), 2)
    assert bed.calls == 1
    assert overlay.calls == 1


def test_bare_dash_clears_every_active_generator():
    tone = ToneSpec(base=200, beat=10)
    active = _apply_schedule_event([], {"alpha": [tone]}, ["alpha"])
    assert _apply_schedule_event(active, {"alpha": [tone]}, ["-"]) == []


def test_arrow_crossfades_to_next_event_over_full_interval():
    low = CountingGenerator(0.2)
    high = CountingGenerator(0.8)
    schedule = [(0.0, ["low", "->"]), (0.02, ["high"])]
    audio = stack(build_session_generator({"low": [low], "high": [high]}, schedule, duration=0.02))
    assert float(np.mean(audio[0])) == pytest.approx(0.2, abs=0.01)
    assert float(np.mean(audio[-1])) == pytest.approx(0.8, abs=0.01)
