from typing import List, Generator, Tuple
import numpy as np

FRAME = 1024
SR = 44100

def mix_generators(gens, duration: float) -> Generator[Tuple[np.ndarray, list], None, None]:
    if not gens or duration <= 0:
        return
    streams = [g.generator(duration) for g in gens]
    while True:
        try:
            acc = np.zeros((FRAME, 2), dtype=np.float32)
            infos = []
            for st in streams:
                chunk, info = next(st)
                if len(chunk) < len(acc):
                    tmp = np.zeros_like(acc)
                    tmp[:len(chunk)] = chunk
                    acc += tmp
                else:
                    acc += chunk
                infos.append(info)
            # simple limiter (soft clip)
            peak = np.max(np.abs(acc))
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
