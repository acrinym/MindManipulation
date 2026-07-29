from __future__ import annotations

import argparse
from pathlib import Path

from .api import render_sleep, write_audio
from .playback import play_chunks
from .sleep import (
    DURATION_CHOICES,
    INTENSITY_LABELS,
    PROBLEM_LABELS,
    SOUND_WORLD_LABELS,
    SleepRequest,
    build_sleep_recipe,
    write_recipe_manifest,
)


def _choose(question: str, choices: dict[str, str], input_fn=input, print_fn=print) -> str:
    print_fn(f"\n{question}")
    items = list(choices.items())
    for index, (_, label) in enumerate(items, start=1):
        print_fn(f"  {index}. {label}")
    while True:
        answer = input_fn("Choose a number: ").strip()
        try:
            selected = int(answer) - 1
        except ValueError:
            selected = -1
        if 0 <= selected < len(items):
            return items[selected][0]
        print_fn("Please choose one of the listed numbers.")


def collect_sleep_request(input_fn=input, print_fn=print) -> SleepRequest:
    print_fn("PySbagen Sleep Guide")
    print_fn("Answer four brief questions. You do not need to know anything about frequencies.")
    problem = _choose("What is keeping you awake tonight?", PROBLEM_LABELS, input_fn, print_fn)
    sound_world = _choose("What kind of sound feels tolerable or pleasant tonight?", SOUND_WORLD_LABELS, input_fn, print_fn)
    user_audio = None
    if sound_world == "user_audio":
        user_audio = input_fn("Path to your music or audio file: ").strip()
    intensity = _choose("How present should the underlying layers feel?", INTENSITY_LABELS, input_fn, print_fn)
    durations = {str(value): f"{value} minutes" for value in DURATION_CHOICES}
    duration = float(_choose("How long should the journey stay with you?", durations, input_fn, print_fn))
    return SleepRequest(
        problem=problem,
        sound_world=sound_world,
        intensity=intensity,
        duration_minutes=duration,
        user_audio=user_audio,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conversational PySbagen sleep guide")
    parser.add_argument("-o", "--outfile", default="sleep-journey.wav")
    parser.add_argument("--play", action="store_true", help="Play immediately instead of only writing a file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        request = collect_sleep_request()
        recipe = build_sleep_recipe(request)
        print(f"\nMatched journey: {recipe.name}")
        print(recipe.description)
        print("This is a sleep-preparation audio experience, not medical, addiction, or emergency treatment.")
        print("Use normal professional or emergency support for dangerous withdrawal, severe or unusual symptoms, or an urgent crisis.")
        if args.play:
            print("Starting playback. Press Ctrl+C to stop.")
            play_chunks(render_sleep(request))
        else:
            result = write_audio(render_sleep(request), args.outfile)
            manifest = write_recipe_manifest(recipe, result.outfile)
            print(f"Wrote {result.duration:.2f}s to {result.outfile}")
            print(f"Saved the exact recipe to {manifest}")
    except KeyboardInterrupt:
        print("\nStopped.")
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"pysbagen sleep guide: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
