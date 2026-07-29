# PySbagen Sleep Audio Research Foundations

**Status:** Product direction recorded before the next build beadtrain  
**Date:** July 29, 2026  
**Scope:** Sleep support, audio-layer research, and future voluntary participant research

## Why this document exists

PySbagen must not collapse back into a frequency picker or a utility that only produces exposed sine tones, static, and noise. Its deeper purpose is to accept a human problem, construct a pleasant and technically reproducible audio experience for that problem, and learn what parts of the experience are actually useful.

The first complete use-case family is **sleep difficulty**:

- a racing mind that will not disengage;
- a person who becomes relaxed but does not cross into sleep;
- a person who falls asleep and repeatedly wakes;
- combinations of the above, including sleep difficulty complicated by pain, migraine, agitation, or cravings.

The ordinary user should be able to answer a human question about what is happening and receive an audio experience designed for that situation. They should not need to understand carrier frequencies, beat matrices, modulation depth, or schedule syntax.

The technical and research layers remain available underneath that experience.

---

## Product decisions already made

### 1. Two sleep systems

PySbagen will treat sleep onset and sleep-state support as different problems.

#### Sleep Descent

For a person who is still awake. It should:

- meet the listener where they are;
- reduce mental and sensory engagement without becoming irritating;
- use pleasant generated or supplied audio as a real part of the intervention;
- change its underlying audio structures over time rather than presenting one static beat indefinitely;
- fade gradually after the intended descent period.

#### Sleep Support

For use after sleep is likely or has been detected. It may eventually use closed-loop timing to support sleep-stage rhythms without waking the listener.

**Sensors are a documented future path, not part of the next implementation.** Do not add dead device adapters, fake endpoints, placeholder services, or hardware integrations before supported hardware and a real validation path exist. Future smellchecks must not find systems that were wired only because they might be useful someday.

### 2. Pleasant audio from two sources

PySbagen should support both:

1. **Self-generated sound worlds** — evolving ambient material, musical beds, environmental textures, and other composed audio.
2. **User-provided audio** — music, ambience, recordings, or other source material in broadly encountered formats, not a WAV-only workflow.

The product should use broad decoder support, including FFmpeg fallback where appropriate, rather than exposing arbitrary file-format restrictions in the user experience.

### 3. Separate research-dose environment

Blinded comparisons, research conditions, consent, condition assignment, and study data do **not** belong in the standard user GUI, TUI, or ordinary session flow.

They belong in a separate **Research Dose Environment** with its own:

- informed consent;
- study protocol and eligibility rules;
- condition definitions;
- blinded or partially blinded assignment;
- session identity and exact recipe capture;
- pre-session and post-session measures;
- adverse-effect reporting;
- data export and protocol versioning.

Normal users receive help-oriented experiences. Research volunteers enter a deliberately marked research setting.

### 4. Ask what is happening, then choose the experience

The product should not force one generic “sleep” preset on every sleep complaint.

The ordinary flow should ask what is happening in human terms, distinguish the major sleep problems, and select or construct an appropriate journey. The goal is to deliver the best-matched audio intervention for the reported problem while remaining honest that no single recipe can be guaranteed to work for every listener.

---

## Reconnaissance method

The research pass used the uploaded **InvisiSynth / Holo Scout** framework as a map rather than relying on a shallow “binaural beats for sleep” search.

The useful scout lines were:

- historical patents and expired technical lineage;
- academic lab-to-market work;
- adjacent sleep and closed-loop stimulation research;
- original SBaGen documentation and maker lineage;
- current systematic reviews that test whether the popular claims survive comparison.

The findings below distinguish:

- **documented mechanism or observed result**;
- **patent claim or product lineage**;
- **promising but unsettled evidence**;
- **future hypothesis that PySbagen should test rather than advertise as fact**.

---

## Research finding 1: pleasant sound is part of the system, not decoration

Robert Monroe’s 1970 sleep patent describes a familiar, pleasing, repetitive sound modulated by a waveform shaped like an EEG sleep pattern. The chosen sound was intended to be pleasant to the individual, repetitive, and strong enough to mask environmental sound, with timed shutoff.

This does not prove all of the patent’s outcome claims. It does show that the original design lineage was not based on a bare sine wave alone. Preference, masking, repetition, modulation, and time progression were part of the proposed mechanism.

### Product implication

The pleasant layer is not merely camouflage over the “real” tone. PySbagen should treat soundscape choice, familiarity, musical movement, masking, and gradual reduction of novelty as first-class session parameters.

**Primary source:**  
https://patents.google.com/patent/US3884218A/en

---

## Research finding 2: Monroe’s later work used layered beat geometry

Monroe’s later patent describes an algorithmic sound system combining binaural relationships, monaural amplitude beats, amplitude and frequency modulation, pink-sound masking, optional voice, multiple carriers, and changing sequences.

Its named “Septon” example uses three tones in each ear. The spacing creates three cross-ear binaural relationships and two monaural beat relationships per channel, yielding seven beat signals rather than one static left/right difference.

A patent records a design and its claims; it does not by itself establish clinical effectiveness. The important engineering lesson is the **multi-layer, time-varying signal architecture**.

### Product implication

PySbagen should support independently controllable and reproducible layers, including:

- multiple carrier groups;
- cross-ear binaural relationships;
- within-channel monaural relationships;
- isochronic or shaped amplitude modulation;
- phase and frequency movement;
- masking or musical beds;
- scheduled changes rather than a single unchanging tone.

**Primary source:**  
https://patents.google.com/patent/US5356368A/en

---

## Research finding 3: state and timing can matter more than a frequency label

A human study using 0.8 Hz rhythmic acoustic stimulation found that stimulation delivered while participants were still awake delayed sleep onset. Once sleep was established, the same rhythm increased and entrained endogenous slow-oscillation activity.

Closed-loop studies further show that sound timed to particular phases of ongoing slow oscillations can alter slow-oscillation and spindle activity. The effect depends on timing relative to the brain’s current state; merely selecting “delta” is not enough.

### Product implication

PySbagen must not assume:

> slow frequency = immediate sleep induction

Sleep Descent and Sleep Support require different schedules and different evidence standards. Until real sensing exists, the ordinary product can provide a planned descent and fade, but it must not pretend to know the listener’s exact sleep stage.

**Primary sources:**

- https://pubmed.ncbi.nlm.nih.gov/22913273/
- https://pubmed.ncbi.nlm.nih.gov/23583623/
- https://pubmed.ncbi.nlm.nih.gov/37660843/

---

## Research finding 4: music provides a more credible reward path than a magical “dopamine frequency”

Human PET and fMRI work found endogenous striatal dopamine release during intense musical pleasure. Anticipation and peak emotional response involved distinguishable reward-system activity.

This does **not** show that an arbitrary tone or beat rate directly “gives dopamine.” It does support a stronger research path: rewarding musical expectation, emotional movement, personalization, and tension/release may help make a non-drug audio experience compelling and repeatable.

### Product implication

For sleep, the audio may begin with enough pleasant structure to satisfy and hold a restless mind, then reduce novelty and stimulation during the descent. Candidate dimensions include:

- anticipation without startling changes;
- gentle harmonic arrival and release;
- personally preferred timbre;
- evolving but slowing texture;
- an early emotional settling point;
- a long, increasingly uneventful fade.

The hidden beat layers and the pleasant musical journey should be tested independently and together.

**Primary source:**  
https://pubmed.ncbi.nlm.nih.gov/21217764/

---

## Research finding 5: monaural and binaural are different tools, not a quality ladder

A human auditory steady-state response study found a measurable 40 Hz binaural response at a low mean carrier near 400 Hz, but not above 3 kHz. The binaural response was smaller than the acoustic/monaural beat response.

Other studies report differing EEG or behavioral effects for monaural and binaural conditions. Results across the wider binaural-beat literature remain inconsistent and methodologically heterogeneous.

### Product implication

PySbagen should not encode:

- binaural as automatically superior;
- monaural as a fallback;
- isochronic as merely crude;
- one carrier range as interchangeable with another.

Research recipes must record carrier frequency, beat frequency, amplitude, channel arrangement, masking, duration, ramping, and other signal details. The product should compare techniques rather than choose a winner in advance.

**Primary sources:**

- https://pubmed.ncbi.nlm.nih.gov/15721080/
- https://pubmed.ncbi.nlm.nih.gov/25345689/
- https://pubmed.ncbi.nlm.nih.gov/37205669/

---

## Research finding 6: closed-loop sleep audio is technically plausible, but belongs later

Research has demonstrated slow-oscillation detection using in-ear EEG and brain-state-dependent auditory stimulation. Newer work continues to investigate timing informed by EEG and heart rhythms.

This makes future state-responsive sleep support plausible. It does not justify coding adapters for hardware PySbagen does not possess or support.

### Documented future path

Potential future signals include:

- in-ear or scalp EEG;
- a supported consumer sleep headband;
- movement or stillness;
- heart rate and heart-rate variability;
- breathing information;
- phone interaction stopping;
- agreement among several weaker signals.

Hardware integration begins only when there is a real device, documented access path, test fixture, and end-to-end validation plan.

**Primary sources:**

- https://pubmed.ncbi.nlm.nih.gov/35124848/
- https://pubmed.ncbi.nlm.nih.gov/41593748/

---

## Research finding 7: H-box X should be translated into measurable signal structure

The reconnaissance did not find a peer-reviewed research field using “H-box,” “HboxX,” or “Harmonic Box X” as a standardized scientific intervention name.

That is not a reason to remove it. It is a reason to describe and test the actual signal rather than treating the label as proof.

### Product implication

Every H-box X recipe should be inspectable in ordinary DSP terms:

- carrier count and carrier frequencies;
- left/right assignment;
- cross-ear frequency differences;
- within-channel beat products;
- phase relationships;
- amplitude envelopes;
- modulation frequency and depth;
- transitions over time.

Research conditions should compare the resulting structure with simpler controls.

---

## Research finding 8: original SBaGen was an experimenter’s laboratory

Original SBaGen documentation describes scheduled smooth changes, mixtures of multiple brain-wave frequencies, pink noise and background MP3/OGG audio, user-supplied soundtracks, randomized looping river sounds, analysis tools, and ideas for live-event response and independent channels.

The lineage is broader than a static tone generator. It explicitly supports personal experimentation and precise tuning while warning users that the claims are not guaranteed.

### Product implication

PySbagen should preserve SBaGen’s inspectability and scheduling power while adding a human front door, broader audio handling, better composition, exact recipe capture, and a separate controlled-research environment.

**Primary sources:**

- https://uazu.net/sbagen/
- https://uazu.net/sbagen/faq.html

---

## Current evidence position

The evidence does not support a universal chart where one frequency reliably causes one mental outcome in every person.

The useful evidence supports a more careful product:

- pleasant and personally acceptable sound matters;
- audio composition may contribute reward and engagement;
- timing and current brain state matter;
- carrier and modulation details matter;
- monaural, binaural, isochronic, noise, music, and multi-carrier structures should be compared rather than conflated;
- effects can differ by person and context;
- research recipes must be reproducible;
- standard use and experimental use require separate environments.

A 2023 systematic review found binaural-entrainment EEG results inconsistent across heterogeneous studies. A 2026 systematic review of music and binaural interventions reported promising outcomes across some sleep, anxiety, and cognition studies while emphasizing small samples and design heterogeneity. These are reasons to build better controlled tools, not reasons to make guaranteed medical claims.

**Reviews:**

- https://pubmed.ncbi.nlm.nih.gov/37205669/
- https://pubmed.ncbi.nlm.nih.gov/41656644/
- https://pubmed.ncbi.nlm.nih.gov/34964434/

---

## Proposed ordinary-user model

A tired person should not be confronted with a laboratory console.

The product asks what is happening, such as:

- “My mind keeps racing.”
- “I feel relaxed, but I cannot cross into sleep.”
- “I keep waking back up.”
- “Pain or migraine is keeping me awake.”
- “I am agitated or craving something and cannot settle.”

PySbagen then chooses a complete journey using:

- an appropriate descent shape;
- a generated or user-supplied pleasant audio bed;
- independently layered beat and modulation structures;
- tolerability settings;
- a session duration and fade plan;
- previous personal feedback when available.

Advanced users may inspect and alter the recipe, but the normal listener should not have to engineer it.

---

## Proposed Research Dose Environment

The separate research product should make it possible to compare:

- pleasant soundscape alone;
- identical soundscape plus binaural layers;
- identical soundscape plus monaural layers;
- identical soundscape plus isochronic modulation;
- H-box X structures versus decomposed controls;
- one carrier pair versus multi-carrier matrices;
- static stimulation versus gradual movement;
- generated audio versus participant-selected audio;
- fixed-time descent versus future state-responsive support;
- stimulation continuing after likely sleep versus an earlier fade.

Each dose should preserve:

- source-audio identity and hash;
- generated-audio seed and composition parameters;
- every carrier and modulation parameter;
- exact timing and fades;
- software and protocol versions;
- listener volume calibration or reported level;
- equipment and headphone/speaker route;
- pre-session condition;
- outcome and adverse-effect reports.

The research environment should support control and sham conditions without making the ordinary app feel like an experiment.

---

## Boundaries for the next build discussion

Before implementation begins, the remaining product conversation should settle:

1. How PySbagen asks about the user’s sleep problem without resembling a medical intake form.
2. What the first generated pleasant sound worlds should sound like.
3. How much user-supplied music is transformed versus left recognizable.
4. Whether the first journey is selected from trusted recipes, generated from constraints, or built through a hybrid approach.
5. How the listener reports “worked,” “did nothing,” or “made it worse.”
6. How the system learns personal preference without turning each ordinary night into an uncontrolled experiment.
7. What safety and exclusion language belongs in ordinary use versus the Research Dose Environment.

No next build beadtrain begins until the conversation establishes the intended human experience.