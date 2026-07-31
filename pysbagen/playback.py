from __future__ import annotations

from threading import Event
from typing import Callable, Iterable

import numpy as np


def play_chunks(
    chunks: Iterable[tuple[np.ndarray, list[dict]]],
    *,
    stop_event: Event | None = None,
    on_chunk: Callable[[list[dict]], None] | None = None,
) -> int:
    """Play a generated stream immediately through PyAudio and return frames played."""
    try:
        import pyaudio
    except ImportError as exc:
        raise RuntimeError(
            "Live playback requires the GUI extra and a working PyAudio installation: pip install 'pysbagen[playback]'"
        ) from exc

    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paFloat32, channels=2, rate=44100, output=True)
    frames = 0
    try:
        for chunk, info in chunks:
            if stop_event is not None and stop_event.is_set():
                break
            normalized = np.asarray(chunk, dtype=np.float32)
            if normalized.ndim != 2 or normalized.shape[1] != 2:
                raise ValueError(f"Playback expected stereo chunks, got {normalized.shape}")
            stream.write(normalized.tobytes())
            frames += len(normalized)
            if on_chunk is not None:
                on_chunk(info)
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
    return frames
