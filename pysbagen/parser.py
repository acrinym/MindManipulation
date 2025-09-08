from typing import Dict, List, Tuple, Optional, Union
import os
from .types import AnySpec
from .generators import *

def _parse_hms(s: str) -> int:
    # supports "NOW", "M:S", "H:M:S"
    ts = [int(x) for x in s.split(":")]
    if len(ts) == 1:  # seconds
        h, m, s = 0, 0, ts[0]
    elif len(ts) == 2:
        h, m, s = 0, ts[0], ts[1]
    else:
        h, m, s = ts
    return h*3600 + m*60 + s

def parse_tone_component(spec: str) -> Optional[AnySpec]:
    """Parses a single component string from an SBG file into a generator spec."""
    spec = spec.strip()
    if spec.lower() in {"-", "off"}:
        return None

    if ':' in spec and not spec[0].isdigit():
        # Handle special prefixes like 'iso:', 'hbox:', 'spin:', etc.
        prefix, rest = spec.split(':', 1)
        prefix = prefix.lower()

        if prefix == "iso":
            params_str, amp_str = rest.split('/') if '/' in rest else (rest, '100')
            freq, beat = [float(x) for x in params_str.split(',')]
            return IsochronicSpec(freq=freq, beat=beat, amp=float(amp_str))

        if prefix == "hbox":
            params_str, amp_str = rest.split('/') if '/' in rest else (rest, '100')
            base, diff, mod = [float(x) for x in params_str.split(',')]
            return HarmonicBoxSpec(base=base, diff=diff, mod=mod, amp=float(amp_str))

        # For other unsupported prefixes (spin, slide), just use the core spec.
        spec = rest

    if spec.lower().startswith("pink") or spec.lower().startswith("white"):
        kind, amp_str = spec.split("/")
        return NoiseSpec(kind=kind, amp=float(amp_str))

    if "/" in spec and os.path.isfile(spec.split("/")[0]):
        path, amp_str = spec.split("/")
        return FileSpec(path=path, amp=float(amp_str))

    # Default case: standard binaural tone (e.g., "200+10/50")
    amp = 100.0
    core_spec = spec
    if '/' in spec:
        core_spec, amp_str = spec.rsplit('/', 1)
        amp = float(amp_str)

    beat = 0.0
    if '+' in core_spec:
        base_str, beat_str = core_spec.split('+', 1)
        beat = float(beat_str)
    elif '-' in core_spec and core_spec.count('-') == 1 and not core_spec.startswith('-'):
        # Handle subtraction for beat frequency, but avoid negative base frequencies
        base_str, beat_str = core_spec.split('-', 1)
        beat = -float(beat_str)
    else:
        base_str = core_spec

    return ToneSpec(base=float(base_str), beat=beat, amp=amp)

def parse_sbg_from_string(s: str):
    tone_sets: Dict[str, List[AnySpec]] = {}
    schedule: List[Tuple[float, List[str]]] = []
    for raw in s.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if ":" in line and not line[0].isdigit() and not line.startswith("+"):
            label, rest = line.split(":", 1)
            parts = [p for p in rest.split() if p]
            comps = [c for p in parts if (c := parse_tone_component(p)) is not None]
            tone_sets[label.strip()] = comps
        else:
            if line.upper().startswith("NOW"):
                t = 0
                rest = line[3:].strip()
            else:
                time_str, rest = line.split(maxsplit=1)
                t = _parse_hms(time_str.lstrip("+"))
            items = [x for x in rest.replace("->", " ").split() if x]
            schedule.append((t, items))
    schedule.sort(key=lambda x: x[0])
    return tone_sets, schedule

def parse_sbg(path: str):
    with open(path, "r", encoding="latin-1") as f:
        return parse_sbg_from_string(f.read())
