from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

from .generators.sleep import SleepJourneySpec

SleepProblem = Literal["racing_mind", "cannot_cross", "waking_back_up"]
SleepSoundWorld = Literal["warm_ambient", "slow_night_music", "rain_room", "deep_night", "user_audio"]
SleepIntensity = Literal["gentle", "balanced", "immersive"]

PROBLEM_LABELS = {
    "racing_mind": "My mind will not stop",
    "cannot_cross": "I feel relaxed, but cannot cross into sleep",
    "waking_back_up": "I fall asleep, then keep waking back up",
}
SOUND_WORLD_LABELS = {
    "warm_ambient": "Warm, slowly changing ambient chords",
    "slow_night_music": "Slow, gentle night music",
    "rain_room": "Soft rain-like room",
    "deep_night": "Deep, dark, low-stimulation night sound",
    "user_audio": "Use my own music or audio",
}
INTENSITY_LABELS = {
    "gentle": "Very gentle — barely noticeable underlying layers",
    "balanced": "Balanced — present, but not demanding attention",
    "immersive": "Immersive — deeper and more enveloping",
}
DURATION_CHOICES = (30, 45, 60, 90)


@dataclass(frozen=True)
class SleepLayers:
    binaural: bool = True
    monaural: bool = True
    isochronic: bool = False
    harmonic_box: bool = True

    def enabled_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, enabled in (
                ("binaural", self.binaural),
                ("monaural", self.monaural),
                ("isochronic", self.isochronic),
                ("harmonic_box", self.harmonic_box),
            )
            if enabled
        )


def recommended_layers(problem: SleepProblem, intensity: SleepIntensity) -> SleepLayers:
    """Choose a tolerable starting blend without treating one technique as universally superior."""
    if problem == "waking_back_up":
        return SleepLayers(binaural=True, monaural=True, isochronic=False, harmonic_box=False)
    if problem == "cannot_cross":
        return SleepLayers(
            binaural=True,
            monaural=True,
            isochronic=False,
            harmonic_box=intensity != "gentle",
        )
    return SleepLayers(
        binaural=True,
        monaural=True,
        isochronic=intensity == "immersive",
        harmonic_box=True,
    )


@dataclass(frozen=True)
class SleepRequest:
    problem: SleepProblem
    sound_world: SleepSoundWorld
    intensity: SleepIntensity = "balanced"
    duration_minutes: float = 45.0
    user_audio: str | None = None
    layers: SleepLayers | None = None
    seed: int = 0

    def validate(self) -> None:
        if self.problem not in PROBLEM_LABELS:
            raise ValueError(f"Unknown sleep problem: {self.problem}")
        if self.sound_world not in SOUND_WORLD_LABELS:
            raise ValueError(f"Unknown sleep sound world: {self.sound_world}")
        if self.intensity not in INTENSITY_LABELS:
            raise ValueError(f"Unknown sleep intensity: {self.intensity}")
        if not 10 <= float(self.duration_minutes) <= 180:
            raise ValueError("Sleep journeys must be between 10 and 180 minutes")
        if self.sound_world == "user_audio" and not self.user_audio:
            raise ValueError("Choose an audio file when using your own audio")
        if self.user_audio and not Path(self.user_audio).expanduser().is_file():
            raise FileNotFoundError(f"Audio file not found: {Path(self.user_audio).expanduser()}")
        if self.layers is not None and not self.layers.enabled_names():
            raise ValueError("Enable at least one underlying audio layer")

    @property
    def duration_seconds(self) -> float:
        return float(self.duration_minutes) * 60.0


@dataclass(frozen=True)
class SleepRecipe:
    name: str
    request: SleepRequest
    descent_seconds: float
    support_seconds: float
    start_beat_hz: float
    end_beat_hz: float
    carrier_hz: float
    fade_out_seconds: float
    description: str

    @property
    def duration_seconds(self) -> float:
        return self.descent_seconds + self.support_seconds


def build_sleep_recipe(request: SleepRequest) -> SleepRecipe:
    request.validate()
    resolved = replace(
        request,
        layers=request.layers or recommended_layers(request.problem, request.intensity),
    )
    resolved.validate()
    duration = resolved.duration_seconds
    profiles = {
        "racing_mind": {
            "name": "Racing Mind Descent",
            "descent_ratio": 0.68,
            "start": 10.0,
            "end": 4.8,
            "carrier": 190.0,
            "description": "Begins with enough gentle movement to occupy a busy mind, then steadily removes novelty and intensity.",
        },
        "cannot_cross": {
            "name": "Crossing the Threshold",
            "descent_ratio": 0.52,
            "start": 7.5,
            "end": 4.5,
            "carrier": 175.0,
            "description": "Uses a quieter, shorter descent and a longer uneventful tail for a listener who is already relaxed.",
        },
        "waking_back_up": {
            "name": "Stay-Asleep Support",
            "descent_ratio": 0.30,
            "start": 7.0,
            "end": 5.0,
            "carrier": 160.0,
            "description": "Settles quickly, then keeps a long stable low-novelty bed before slowly disappearing.",
        },
    }
    profile = profiles[resolved.problem]
    descent = duration * profile["descent_ratio"]
    support = duration - descent
    fade_out = min(max(duration * 0.12, 90.0), 360.0)
    return SleepRecipe(
        name=profile["name"],
        request=resolved,
        descent_seconds=descent,
        support_seconds=support,
        start_beat_hz=profile["start"],
        end_beat_hz=profile["end"],
        carrier_hz=profile["carrier"],
        fade_out_seconds=fade_out,
        description=profile["description"],
    )


def build_sleep_spec(request: SleepRequest) -> SleepJourneySpec:
    return SleepJourneySpec(recipe=build_sleep_recipe(request))


def recipe_manifest(recipe: SleepRecipe) -> dict:
    request = recipe.request
    manifest = {
        "format": "pysbagen-sleep-recipe-v1",
        "recipe": {
            "name": recipe.name,
            "description": recipe.description,
            "descent_seconds": recipe.descent_seconds,
            "support_seconds": recipe.support_seconds,
            "start_beat_hz": recipe.start_beat_hz,
            "end_beat_hz": recipe.end_beat_hz,
            "carrier_hz": recipe.carrier_hz,
            "fade_out_seconds": recipe.fade_out_seconds,
        },
        "request": asdict(request),
    }
    if request.user_audio:
        source = Path(request.user_audio).expanduser()
        manifest["source_audio"] = {
            "path": str(source.resolve()),
            "sha256": _sha256(source),
        }
    return manifest


def write_recipe_manifest(recipe: SleepRecipe, audio_path: str | Path) -> Path:
    path = Path(audio_path).expanduser()
    manifest_path = path.with_suffix(path.suffix + ".sleep.json")
    manifest_path.write_text(
        json.dumps(recipe_manifest(recipe), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
