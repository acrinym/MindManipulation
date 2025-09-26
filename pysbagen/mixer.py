from typing import Generator, Iterable, List, Tuple
import numpy as np

FRAME = 1024
SR = 44100

def _pad_chunk(chunk: np.ndarray, frame_len: int) -> np.ndarray:
    """Pad a chunk to ``frame_len`` without modifying the input."""
    if chunk.shape[0] == frame_len:
        return chunk
    padded = np.zeros((frame_len, chunk.shape[1]), dtype=chunk.dtype)
    padded[: chunk.shape[0]] = chunk
    return padded


def mix_generators(gens: Iterable, duration: float) -> Generator[Tuple[np.ndarray, list], None, None]:
    if not gens or duration <= 0:
        return

    streams = [g.generator(duration) for g in gens]
    while True:
        try:
            chunks: List[np.ndarray] = []
            infos: List[dict] = []
            for stream in streams:
                chunk, info = next(stream)
                chunks.append(chunk)
                infos.append(info)

            frame_len = max(chunk.shape[0] for chunk in chunks)
            acc = np.zeros((frame_len, 2), dtype=np.float32)
            for chunk in chunks:
                acc += _pad_chunk(chunk.astype(np.float32), frame_len)

            peak = float(np.max(np.abs(acc)))
            if peak > 1.0:
                acc /= peak

            yield acc, infos
        except StopIteration:
            break

def build_session_generator(tone_sets, schedule, duration=None):
    if not schedule:
        return
    times = [t for t, _ in schedule]
    if duration is None:
        duration = times[-1]
    schedule = schedule + [(duration, ["off"])]

    active = []
    last = 0.0
    for start, names in schedule:
        seg = start - last
        if seg > 0:
            yield from mix_generators(active, seg)

        absolute = not names[0].startswith(('+', '-'))
        if absolute:
            active.clear()

        tokens = {n.lower() for n in names}
        if "alloff" in tokens:
            active.clear()

        for n in names:
            if n.lower() in {"-", "off", "alloff"}:
                continue
            clean = n.lstrip("+-")
            if n.startswith('-'):
                # remove by identity
                to_remove = tone_sets.get(clean, [])
                active = [g for g in active if g not in to_remove]
            else:
                active.extend(tone_sets.get(clean, []))
        last = start
