# PySbagen Sleep Guide

## What it is

The Sleep Guide is PySbagen’s ordinary-person front door. A tired listener answers four human questions and receives a complete time-changing journey. The advanced studio and future Research Dose Environment remain separate.

## The four questions

1. **What is keeping you awake?**
   - My mind will not stop.
   - I feel relaxed, but cannot cross into sleep.
   - I fall asleep, then keep waking back up.
2. **What feels pleasant or tolerable tonight?**
   - warm evolving ambience;
   - slow night music;
   - a soft rain-like room;
   - a deep low-stimulation night environment;
   - the listener’s own audio.
3. **How present should the underlying layers feel?**
   - gentle;
   - balanced;
   - immersive.
4. **How long should the journey remain?**
   - 30, 45, 60, or 90 minutes in the guide;
   - 10–180 minutes through the API.

The listener does not choose frequencies unless they deliberately enter advanced work.

## Three matched routes

### Racing Mind Descent

Starts with enough musical or textural movement to occupy a busy mind, then progressively reduces novelty, beat rate, and active-layer strength.

### Crossing the Threshold

Begins more quietly for someone already relaxed and leaves a longer stable support tail.

### Stay-Asleep Support

Uses a shorter descent, longer low-novelty support bed, and a less-active recommended blend.

These are the only ordinary routes currently implemented. Pain, migraine, craving, and substance-use pathways are future work, not hidden interpretations of these three choices.

## Two phases

Every journey contains:

1. **Sleep Descent** for the awake listener;
2. **Sleep Support**, a quieter period after sleep is more likely.

The present transition is time-planned. PySbagen does not claim to detect sleep stage. Sensor-driven support begins only when real supported hardware and an end-to-end validation path exist.

## Pleasant audio

PySbagen can generate:

- slowly moving warm chords;
- long-crossfade night music with sparse fading melody;
- stateful filtered rain-like texture;
- dark low-stimulation drones and subdued noise.

A listener may instead supply their own audio. Common formats use SoundFile; other locally decodable formats use streaming FFmpeg fallback. Audio is looped in bounded chunks rather than loaded or tiled into a session-sized array.

## Underlying layers

Recommended blends may combine:

- binaural relationships;
- within-channel monaural beating;
- smooth isochronic modulation;
- Harmonic Box X-style multi-phase layering.

Layers change over time, maintain phase between chunks, and recede strongly during support while the pleasant bed fades more slowly.

## Playback and export

- `sbgpy-sleep-gui` opens the desktop guide.
- `sbgpy-sleep` runs the terminal guide.
- `sbgpy-sleep --play` starts immediate playback after the questions.

Install export-only GUI support with `pysbagen[gui]`, playback with `pysbagen[playback]`, or both with `pysbagen[desktop]`.

Saved sessions receive a `.sleep.json` sidecar with the route, exact timings, layers, carriers/beat movement, deterministic seed, and source-audio SHA-256 when used.

## Safety boundary

Use a comfortable volume and stop on pain, dizziness, agitation, worsened symptoms, or another unwanted effect. Do not use sleep audio while driving or doing anything requiring alertness.

This is sleep-preparation audio, not medical, addiction, withdrawal, or emergency treatment. Severe, unusual, worsening, or persistent symptoms need ordinary professional care. Dangerous withdrawal, overdose, self-harm risk, or an urgent crisis needs real-world emergency or crisis support, not PySbagen.
