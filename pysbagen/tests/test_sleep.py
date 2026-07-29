from pathlib import Path
import json
import numpy as np
import pytest
import soundfile as sf

from pysbagen.generators.sleep import SleepJourneySpec
from pysbagen.sleep import (
    SleepLayers,
    SleepRequest,
    build_sleep_recipe,
    recommended_layers,
    recipe_manifest,
    write_recipe_manifest,
)


def request(**overrides):
    values = dict(problem="racing_mind", sound_world="warm_ambient", intensity="balanced", duration_minutes=10, seed=4)
    values.update(overrides)
    return SleepRequest(**values)


def test_problem_profiles_are_materially_different():
    racing = build_sleep_recipe(request(problem="racing_mind"))
    crossing = build_sleep_recipe(request(problem="cannot_cross"))
    waking = build_sleep_recipe(request(problem="waking_back_up"))
    assert racing.descent_seconds > crossing.descent_seconds > waking.descent_seconds
    assert racing.start_beat_hz > crossing.start_beat_hz >= waking.start_beat_hz
    assert waking.support_seconds > crossing.support_seconds > racing.support_seconds


def test_recommended_layers_change_by_problem_and_intensity():
    assert recommended_layers("racing_mind", "immersive").isochronic is True
    assert recommended_layers("waking_back_up", "immersive").harmonic_box is False
    assert recommended_layers("cannot_cross", "gentle").harmonic_box is False


def test_request_requires_user_audio_when_selected():
    with pytest.raises(ValueError, match="Choose an audio file"):
        build_sleep_recipe(request(sound_world="user_audio"))


def test_request_requires_at_least_one_custom_layer():
    with pytest.raises(ValueError, match="Enable at least one"):
        build_sleep_recipe(request(layers=SleepLayers(False, False, False, False)))


def test_low_rate_generator_is_stereo_bounded_and_reports_both_stages():
    recipe = build_sleep_recipe(request())
    spec = SleepJourneySpec(recipe=recipe, sample_rate=200, frame=64)
    frames = 0
    stages = set()
    for chunk, info in spec.generator(recipe.duration_seconds):
        assert chunk.ndim == 2 and chunk.shape[1] == 2
        assert np.max(np.abs(chunk)) <= 1.0
        frames += len(chunk)
        stages.add(info["stage"])
    assert frames == int(200 * recipe.duration_seconds)
    assert stages == {"sleep_descent", "sleep_support"}


def test_user_audio_can_be_layered_and_looped(tmp_path: Path):
    path = tmp_path / "short.flac"
    data = np.column_stack((np.linspace(-0.2, 0.2, 200), np.linspace(0.2, -0.2, 200))).astype(np.float32)
    sf.write(path, data, 200)
    recipe = build_sleep_recipe(request(sound_world="user_audio", user_audio=str(path)))
    first, info = next(SleepJourneySpec(recipe=recipe, sample_rate=200, frame=64).generator(recipe.duration_seconds))
    assert first.shape == (64, 2)
    assert info["sound_world"] == "user_audio"


def test_seed_makes_generated_bed_reproducible():
    first_recipe = build_sleep_recipe(request(seed=99))
    other_recipe = build_sleep_recipe(request(seed=100))
    first, _ = next(SleepJourneySpec(recipe=first_recipe, sample_rate=200, frame=64).generator(first_recipe.duration_seconds))
    second, _ = next(SleepJourneySpec(recipe=first_recipe, sample_rate=200, frame=64).generator(first_recipe.duration_seconds))
    other, _ = next(SleepJourneySpec(recipe=other_recipe, sample_rate=200, frame=64).generator(other_recipe.duration_seconds))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_recipe_manifest_records_exact_layers_and_source_hash(tmp_path: Path):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros((20, 2), dtype=np.float32), 200)
    recipe = build_sleep_recipe(request(sound_world="user_audio", user_audio=str(source)))
    manifest = recipe_manifest(recipe)
    assert manifest["format"] == "pysbagen-sleep-recipe-v1"
    assert manifest["request"]["layers"]["binaural"] is True
    assert len(manifest["source_audio"]["sha256"]) == 64
    output = tmp_path / "journey.wav"
    path = write_recipe_manifest(recipe, output)
    assert json.loads(path.read_text())["recipe"]["name"] == recipe.name
