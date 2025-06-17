# MindManipulation

Sbagen Upgrade + isochronic and brain entrainment apps and ideas

## Python Rewrite

`sbagen.py` is a Python implementation of SBAGEN's scheduling engine.  It can
parse most of the sample `.sbg` files distributed with SBAGEN and produce a WAV
file containing the rendered session.  When no schedule file is supplied it can
still generate a simple binaural beat tone using command line parameters.

Basic example:

```bash
python sbagen.py focus/ts1-focus10.sbg --outfile session.wav
```

To synthesize a single tone without a schedule:

```bash
python sbagen.py --base 200 --beat 10 --duration 120 --outfile tone.wav
```

Use `--noise` to mix in white noise for the quick-tone mode.

### Advanced Quick Modes

Generate an isochronic tone layered behind music:

```bash
python sbagen.py --isochronic 200 10 --duration 60 --music background.wav \
  --outfile iso.wav
```

Generate a Harmonic Box X sequence and mix with music:

```bash
python sbagen.py --harmonic-box 180 5 8 --duration 120 --music song.wav \
  --outfile hbox.wav

