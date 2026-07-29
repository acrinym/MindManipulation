# PySbagen Sleep Guide

## Purpose

The Sleep Guide is the ordinary-person front door to PySbagen. A tired listener explains what is happening in human terms, chooses a tolerable sound environment, chooses how present the underlying layers should feel, and starts a complete audio journey.

It is deliberately separate from the advanced tone laboratory and from the future Research Dose Environment.

## The conversation

The guide asks four short questions:

1. **What is keeping you awake?**
   - My mind will not stop.
   - I feel relaxed, but cannot cross into sleep.
   - I fall asleep, then keep waking back up.
2. **What would feel pleasant tonight?**
   - warm evolving ambient chords;
   - slow night music;
   - a soft rain-like room;
   - a deep low-stimulation night environment;
   - the listener’s own audio.
3. **How present should the hidden layers feel?**
   - gentle;
   - balanced;
   - immersive.
4. **How long should it remain?**
   - 30, 45, 60, or 90 minutes in the standard guide;
   - the programmatic API accepts 10–180 minutes.

The listener does not need to choose frequencies or understand entrainment terminology.

## Matched routes

### Racing Mind Descent

For a listener whose thoughts keep demanding attention.

- The descent occupies approximately 68% of the session.
- It begins with more musical or textural movement.
- Novelty, beat rate, and active-layer strength recede progressively.
- The remaining support period becomes increasingly uneventful before fading.

### Crossing the Threshold

For a listener who feels relaxed but does not cross into sleep.

- The descent occupies approximately 52% of the session.
- It begins more quietly than Racing Mind Descent.
- It leaves a longer stable support period.

### Stay-Asleep Support

For a listener who falls asleep and wakes repeatedly.

- The descent occupies approximately 30% of the session.
- The support bed is materially longer.
- The recommended blend avoids the most active optional layers.

These are support recipes, not medical guarantees.

## Two sleep systems

Every current recipe contains:

1. **Sleep Descent** — audio for a person who is still awake;
2. **Sleep Support** — a quieter, lower-novelty period after sleep is more likely.

The current transition is planned by time. PySbagen does not claim to detect sleep stage.

Future closed-loop support may use real hardware such as supported EEG, movement, heart-rate, or breathing sensors. No device adapters, placeholder endpoints, or fake sensor services are included now. Hardware work begins only when a real device and an end-to-end validation path exist.

## Pleasant sound generation

### Warm ambient chords

A deterministic stereo pad built from slowly phase-moving harmonic voices. A subtle upper component diminishes as the journey progresses.

### Slow night music

A procedural four-chord cycle with long crossfades and a sparse, smoothly enveloped melody. Both chord motion and melody lose prominence during descent and support.

### Rain-like room

A stateful filtered-noise room plus a quieter high-frequency droplet texture. Filter state persists across chunks so the sound does not restart every buffer.

### Deep night

Low stereo drones, a fading upper voice, and a very subdued filtered-noise floor.

### User-provided audio

The chosen audio is looped when shorter than the journey, faded with the session, and combined with a small generated underlay. Common formats use SoundFile. Other formats fall back to the system’s FFmpeg decoder.

## Underlying layers

The recipe can combine:

- evolving binaural relationships;
- within-channel monaural beating;
- smooth isochronic amplitude modulation;
- a Harmonic Box X-style multi-phase, multi-carrier layer.

The normal guide offers a recommended blend based on the reported problem and desired intensity. Optional controls allow the listener to change the layer selection.

All active layers use continuously accumulated phase rather than restarting at chunk boundaries. Their strength recedes sharply after the descent period, while the pleasant bed continues longer and fades gradually.

## Immediate playback and export

The desktop guide can stream the journey directly to PyAudio so the listener can put on headphones and begin without pre-rendering a long file.

It can also save the audio. Saved journeys receive a companion `.sleep.json` manifest containing:

- the problem and sound selection;
- intensity and duration;
- resolved layer blend;
- descent/support timing;
- carrier and beat movement;
- fade timing;
- deterministic generation seed;
- user-audio path and SHA-256 when applicable.

This preserves reproducibility without placing research controls in ordinary use.

## Research boundary

Ordinary Sleep Guide sessions are not research doses.

Blinding, sham conditions, protocol assignment, consent, eligibility rules, pre/post measures, adverse-effect reporting, and study data export belong in a separately launched Research Dose Environment. See `docs/research/SLEEP_AUDIO_RESEARCH_FOUNDATIONS.md`.

## Safety boundary

The guide tells the listener to use a comfortable volume and stop if the audio causes pain, dizziness, agitation, worsened symptoms, or other unwanted effects.

It must not claim to diagnose or treat insomnia, migraine, chronic pain, substance use disorder, or another condition. It must not promise direct dopamine delivery or guaranteed sleep.
