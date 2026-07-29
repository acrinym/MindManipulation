# PySbagen Sleep Audio Research Foundations

**Status:** Product direction recorded before and during the sleep-experience beadtrain  
**Date:** July 29, 2026

## Purpose

PySbagen is not a frequency picker. Its ordinary product asks what a person is experiencing, builds a pleasant and reproducible audio journey, and learns—without pretending certainty—which parts are useful.

The first supported family is sleep difficulty:

- a racing mind that will not disengage;
- feeling relaxed but not crossing into sleep;
- falling asleep and repeatedly waking.

Pain, migraine, cravings, substance-use support, and other use cases remain future product and research paths. They must not be silently inferred from the current sleep routes.

## Product decisions

### Two sleep systems

**Sleep Descent** serves a person who is still awake. It should begin where their attention is, reduce novelty and stimulation over time, and fade gradually.

**Sleep Support** is the quieter period after sleep is more likely. Future closed-loop timing may use real sensors, but no EEG, watch, movement, heart-rate, breathing, or other device adapters are coded until supported hardware and an end-to-end validation path exist.

### Pleasant audio is first-class

PySbagen supports both:

1. self-generated music, ambience, environmental texture, and sound worlds;
2. user-provided audio in broadly decodable formats, not a WAV-only workflow.

### Ordinary use and research use are separate

The normal GUI/TUI/CLI provides help-oriented sessions. Blinding, sham conditions, consent, eligibility, protocol assignment, pre/post measures, adverse-event reporting, and study export belong in a separately launched **Research Dose Environment**.

## Reconnaissance method

The uploaded InvisiSynth / Holo Scout framework guided separate searches through:

- historical patents and technical lineage;
- overlooked academic and lab-to-market work;
- adjacent sleep and closed-loop stimulation research;
- original SBaGen documentation and maker lineage;
- systematic reviews that test popular claims.

Patents document designs and claims, not clinical proof. Observed study results, unsettled hypotheses, and product lineage are kept distinct.

## Findings

### 1. Pleasant sound was part of the mechanism

Robert Monroe’s 1970 sleep patent described a familiar, pleasing, repetitive sound modulated by a sleep-pattern-shaped waveform, used for environmental masking and timed shutoff. This does not prove its outcome claims, but it shows that preference, masking, repetition, modulation, and progression were central—not decorative material covering a bare tone.

Source: https://patents.google.com/patent/US3884218A/en

### 2. Monroe’s later system used layered beat geometry

A later Monroe patent described multiple carriers, cross-ear binaural relationships, within-channel monaural beats, amplitude and frequency modulation, phased pink sound, optional voice, and changing sequences. Its “Septon” example creates several simultaneous relationships rather than one static left/right difference.

Product implication: recipes must expose carrier groups, channel assignment, amplitude envelopes, modulation, phase/frequency movement, masking beds, and timing.

Source: https://patents.google.com/patent/US5356368A/en

### 3. State and timing can matter more than a frequency label

A human study found that 0.8 Hz rhythmic acoustic stimulation begun while participants were awake delayed sleep onset; after stable non-REM sleep existed, the same rhythm increased slow-oscillation activity. Other closed-loop work found effects depended on stimulus timing relative to the ongoing slow-oscillation phase.

Product implication: “slow frequency = immediate sleep” is not a safe design rule. Descent and support require different schedules and evidence standards.

Sources:

- https://pubmed.ncbi.nlm.nih.gov/22913273/
- https://pubmed.ncbi.nlm.nih.gov/23583623/
- https://pubmed.ncbi.nlm.nih.gov/37660843/

### 4. Music is a more credible reward path than a magical dopamine frequency

Human PET/fMRI work found endogenous striatal dopamine release during intense musical pleasure, with distinguishable anticipation and peak-response activity. It does not show that an arbitrary beat rate directly “gives dopamine.”

Product implication: personally rewarding timbre, expectation, gentle harmonic movement, tension/release, and decreasing novelty are stronger research dimensions than advertising a dopamine frequency. Pleasant music and hidden modulation layers should be compared separately and together.

Source: https://pubmed.ncbi.nlm.nih.gov/21217764/

### 5. Monaural and binaural are different tools, not a quality ladder

A human auditory steady-state study detected a 40 Hz binaural response near a 400 Hz carrier but not above 3 kHz; the binaural response was smaller than the monaural/acoustic beat response. Wider findings remain inconsistent and methodologically heterogeneous.

Product implication: record carrier, beat, amplitude, channels, masking, ramping, and duration. Do not encode binaural as automatically superior, monaural as a fallback, or isochronic as crude.

Sources:

- https://pubmed.ncbi.nlm.nih.gov/15721080/
- https://pubmed.ncbi.nlm.nih.gov/25345689/
- https://pubmed.ncbi.nlm.nih.gov/37205669/

### 6. Closed-loop sleep audio is plausible but later

Researchers have demonstrated in-ear EEG detection and brain-state-dependent auditory stimulation. Newer work also explores timing informed by EEG and heart rhythms. This supports a future hardware path, not placeholder endpoints now.

Sources:

- https://pubmed.ncbi.nlm.nih.gov/35124848/
- https://pubmed.ncbi.nlm.nih.gov/41593748/

### 7. H-box X should be translated into measurable structure

The reconnaissance did not find “H-box X” as a standardized peer-reviewed intervention name. It remains useful as a synthesis structure, but research must describe its carrier count, frequencies, channel assignments, cross-ear and within-channel beat products, phase relationships, envelopes, modulation frequency/depth, and transitions.

### 8. Original SBaGen was an experimenter’s laboratory

Original SBaGen documentation includes scheduled smooth changes, mixtures of frequencies, pink noise, background MP3/OGG, user soundtracks, randomized loops, analysis tools, and experimental tuning. PySbagen preserves that inspectability while adding a human front door, broad decoding, pleasant generation, exact recipe capture, and separate research operation.

Sources:

- https://uazu.net/sbagen/
- https://uazu.net/sbagen/faq.html

## Evidence position

There is no established universal chart where one frequency reliably creates one mental outcome for every person. Evidence supports careful comparison, precise recipes, individual tolerability, and attention to timing and state.

A 2023 review found binaural-entrainment EEG results inconsistent across heterogeneous methods. A 2026 review reported promising findings across music and binaural interventions, but it covered only 10 heterogeneous trials involving young adults aged 19–24. It does not justify broad clinical or age-group generalization.

Reviews:

- https://pubmed.ncbi.nlm.nih.gov/37205669/
- https://pubmed.ncbi.nlm.nih.gov/41656644/
- https://pubmed.ncbi.nlm.nih.gov/34964434/

## Research Dose Environment requirements

A research dose preserves:

- consent, eligibility, protocol and software versions;
- blinded/control assignment where appropriate;
- source-audio identity/hash and generated seed;
- every carrier, modulation, channel, timing, fade, and volume parameter;
- listening equipment and route;
- pre/post measures and adverse-effect reports.

Candidate comparisons include soundscape alone, identical soundscape plus each hidden layer, simple versus multi-carrier structures, static versus moving stimulation, generated versus selected audio, and fixed versus future state-responsive support.

## Safety and exclusion boundary

Ordinary PySbagen sleep sessions are non-diagnostic sleep-preparation audio, not treatment for insomnia, migraine, chronic pain, substance use disorder, withdrawal, or another condition. They do not promise dopamine delivery or guaranteed outcomes.

The product must:

- use comfortable-volume guidance and an immediate stop path;
- warn against use while driving or doing alertness-critical work;
- direct severe, unusual, worsening, or persistent symptoms to ordinary professional care;
- direct dangerous withdrawal, overdose, self-harm risk, or urgent crisis to real-world emergency or crisis support—not an audio session;
- exclude unsupported populations or conditions from research protocols until appropriate review and evidence exist;
- require explicit informed consent for research participation.

No next product train may erase these boundaries or quietly convert ordinary users into research participants.
