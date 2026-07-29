from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import lfilter

from .base import GenBase
from .file import iter_audio_chunks

if TYPE_CHECKING:
    from pysbagen.sleep import SleepRecipe


def _smoothstep(value: np.ndarray | float) -> np.ndarray:
    x = np.clip(value, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)




@dataclass
class SleepJourneySpec(GenBase):
    recipe: "SleepRecipe | None" = None

    def _generated_bed(
        self,
        t: np.ndarray,
        world: str,
        rng: np.random.Generator,
        filter_state: list[np.ndarray],
        novelty: np.ndarray,
        phases: np.ndarray,
    ) -> np.ndarray:
        if world == "warm_ambient":
            left = (
                0.36 * np.sin(2 * np.pi * 82.41 * t + phases[0] + 0.18 * np.sin(2 * np.pi * 0.009 * t))
                + 0.24 * np.sin(2 * np.pi * 123.47 * t + phases[1] + 0.12 * np.sin(2 * np.pi * 0.006 * t + 0.8))
                + 0.14 * np.sin(2 * np.pi * 164.81 * t + phases[2] + 0.10 * np.sin(2 * np.pi * 0.004 * t + 1.7))
            )
            right = (
                0.36 * np.sin(2 * np.pi * 82.55 * t + phases[3] + 0.18 * np.sin(2 * np.pi * 0.008 * t + 0.4))
                + 0.24 * np.sin(2 * np.pi * 123.32 * t + phases[4] + 0.12 * np.sin(2 * np.pi * 0.0065 * t + 1.1))
                + 0.14 * np.sin(2 * np.pi * 164.95 * t + phases[5] + 0.10 * np.sin(2 * np.pi * 0.0045 * t + 2.0))
            )
            shimmer = 0.04 * novelty * np.sin(2 * np.pi * 246.94 * t + 0.7 * np.sin(2 * np.pi * 0.012 * t))
            return np.column_stack((left + shimmer, right + shimmer))

        if world == "slow_night_music":
            chords = np.array([
                [65.41, 98.00, 130.81],
                [55.00, 82.41, 110.00],
                [73.42, 110.00, 146.83],
                [49.00, 73.42, 98.00],
            ])
            chord_position = t / 80.0
            chord_index = np.floor(chord_position).astype(int) % len(chords)
            next_index = (chord_index + 1) % len(chords)
            crossfade = _smoothstep((np.mod(chord_position, 1.0) - 0.68) / 0.32)
            left = np.zeros(len(t))
            right = np.zeros(len(t))
            for voice in range(3):
                current = chords[chord_index, voice]
                following = chords[next_index, voice]
                current_left = np.sin(2 * np.pi * current * t + phases[voice])
                next_left = np.sin(2 * np.pi * following * t + phases[voice + 3])
                current_right = np.sin(2 * np.pi * (current * 1.0012) * t + phases[voice + 3])
                next_right = np.sin(2 * np.pi * (following * 0.9988) * t + phases[voice])
                weight = (0.30, 0.22, 0.15)[voice]
                left += weight * ((1.0 - crossfade) * current_left + crossfade * next_left)
                right += weight * ((1.0 - crossfade) * current_right + crossfade * next_right)

            melody_notes = np.array([196.00, 164.81, 146.83, 130.81, 110.00, 130.81, 98.00, 82.41])
            melody_position = t / 24.0
            melody_index = np.floor(melody_position).astype(int) % len(melody_notes)
            melody_fraction = np.mod(melody_position, 1.0)
            melody_envelope = np.sin(np.pi * melody_fraction) ** 2 * novelty
            melody = 0.055 * melody_envelope * np.sin(
                2 * np.pi * melody_notes[melody_index] * t + phases[1]
            )
            return np.column_stack((left + melody, right + melody * 0.92))

        white = rng.normal(0.0, 1.0, (len(t), 2))
        filtered = np.empty_like(white)
        b = np.array([0.018], dtype=np.float64)
        a = np.array([1.0, -0.982], dtype=np.float64)
        for channel in range(2):
            filtered[:, channel], filter_state[channel] = lfilter(
                b, a, white[:, channel], zi=filter_state[channel]
            )

        if world == "rain_room":
            droplets = white * (0.035 + 0.02 * novelty[:, None])
            return filtered * 0.38 + droplets

        drone_left = 0.34 * np.sin(2 * np.pi * 55.0 * t + 0.10 * np.sin(2 * np.pi * 0.004 * t))
        drone_right = 0.34 * np.sin(2 * np.pi * 55.12 * t + 0.10 * np.sin(2 * np.pi * 0.0037 * t + 0.6))
        upper = 0.07 * novelty * np.sin(2 * np.pi * 110.0 * t + 0.4 * np.sin(2 * np.pi * 0.007 * t))
        return np.column_stack((drone_left + upper, drone_right + upper)) + filtered * 0.12

    def generator(self, duration: float):
        if self.recipe is None:
            raise ValueError("SleepJourneySpec requires a sleep recipe")
        recipe = self.recipe
        request = recipe.request
        request.validate()
        if request.layers is None:  # build_sleep_recipe always resolves this.
            raise ValueError("Sleep recipe has no resolved layer selection")
        requested_duration = recipe.duration_seconds
        if abs(duration - requested_duration) > 1.0 / self.sample_rate:
            raise ValueError(
                f"Sleep journey duration is fixed at {requested_duration:.2f}s by its recipe, got {duration:.2f}s"
            )

        total_frames = int(self.sample_rate * duration)
        descent = max(recipe.descent_seconds, 1.0)
        fade_in_seconds = min(30.0, duration * 0.08)
        user_audio_chunks = None
        if request.user_audio:
            user_audio_chunks = iter_audio_chunks(
                request.user_audio,
                total_frames,
                self.frame,
                self.sample_rate,
                loop=True,
            )

        intensity = {
            "gentle": {"bed": 0.42, "binaural": 0.035, "monaural": 0.018, "iso": 0.012, "hbox": 0.014},
            "balanced": {"bed": 0.52, "binaural": 0.055, "monaural": 0.028, "iso": 0.019, "hbox": 0.022},
            "immersive": {"bed": 0.60, "binaural": 0.075, "monaural": 0.040, "iso": 0.028, "hbox": 0.032},
        }[request.intensity]

        rng = np.random.default_rng(request.seed)
        filter_state = [np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)]
        phases = rng.uniform(0.0, 2 * np.pi, 6)
        beat_phase_state = 0.0

        for offset in range(0, total_frames, self.frame):
            n = min(self.frame, total_frames - offset)
            t = (offset + np.arange(n)) / self.sample_rate
            descent_progress = _smoothstep(t / descent)
            beat = recipe.start_beat_hz + (recipe.end_beat_hz - recipe.start_beat_hz) * descent_progress
            after_descent = np.clip((t - descent) / max(recipe.support_seconds, 1.0), 0.0, 1.0)

            fade_in = _smoothstep(t / max(fade_in_seconds, 1.0))
            fade_out = _smoothstep((duration - t) / max(recipe.fade_out_seconds, 1.0))
            global_envelope = fade_in * fade_out
            novelty = (1.0 - 0.82 * descent_progress) * (1.0 - 0.45 * after_descent)
            active_layer_envelope = global_envelope * (1.0 - 0.82 * after_descent)
            bed_envelope = global_envelope * (1.0 - 0.28 * after_descent)

            generated_world = "deep_night" if request.sound_world == "user_audio" else request.sound_world
            bed = self._generated_bed(t, generated_world, rng, filter_state, novelty, phases)
            if user_audio_chunks is not None:
                supplied = next(user_audio_chunks)
                if len(supplied) != n:
                    raise ValueError(
                        f"User audio returned {len(supplied)} frames where {n} were required"
                    )
                bed = supplied * 0.92 + bed * 0.08
            output = bed * (intensity["bed"] * bed_envelope[:, None])

            increments = 2 * np.pi * beat / self.sample_rate
            beat_phase = beat_phase_state + np.concatenate(([0.0], np.cumsum(increments[:-1])))
            beat_phase_state = float(beat_phase[-1] + increments[-1])
            carrier_phase = 2 * np.pi * recipe.carrier_hz * t
            layers = request.layers

            if layers.binaural:
                left = np.sin(carrier_phase)
                right = np.sin(carrier_phase + beat_phase)
                output += np.column_stack((left, right)) * (intensity["binaural"] * active_layer_envelope[:, None])

            if layers.monaural:
                mono_phase = 2 * np.pi * (recipe.carrier_hz + 27.0) * t
                mono = 0.5 * (np.sin(mono_phase) + np.sin(mono_phase + beat_phase))
                output += np.column_stack((mono, mono)) * (intensity["monaural"] * active_layer_envelope[:, None])

            if layers.isochronic:
                pulse = (0.5 - 0.5 * np.cos(beat_phase)) ** 2
                iso = np.sin(2 * np.pi * (recipe.carrier_hz * 0.72) * t) * pulse
                output += np.column_stack((iso, iso)) * (intensity["iso"] * active_layer_envelope[:, None])

            if layers.harmonic_box:
                hleft = np.zeros(n)
                hright = np.zeros(n)
                for index, phase in enumerate((0.0, np.pi / 2, np.pi, 3 * np.pi / 2)):
                    gate = 0.5 + 0.5 * np.sin(beat_phase + phase)
                    hleft += np.sin(carrier_phase + index * 0.5 * beat_phase) * gate
                    hright += np.sin(carrier_phase + (1.0 + index * 0.5) * beat_phase) * gate
                output += np.column_stack((hleft, hright)) / 4.0 * (
                    intensity["hbox"] * active_layer_envelope[:, None]
                )

            output = np.tanh(output * self._amp_scale()).astype(np.float32)
            stage = "sleep_descent" if float(t[-1]) < descent else "sleep_support"
            yield output, {
                "type": "sleep_journey",
                "recipe": recipe.name,
                "stage": stage,
                "problem": request.problem,
                "sound_world": request.sound_world,
                "beat_hz": float(np.mean(beat)),
                "layers": layers.enabled_names(),
            }
