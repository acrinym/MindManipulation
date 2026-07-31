# I-Doser and SBaGen Pain-Point Scout

**Date:** July 31, 2026  
**Status:** Research and product-direction record before implementation  
**Scope:** User, workflow, compatibility, preservation, transparency, and authoring pain points across I-Doser, original SBaGen, and the lineage between them

## Why this scout exists

PySbagen should not rush into another use-case product while the I-Doser and SBaGen foundations still contain unresolved user and compatibility pain.

The goal of this pass was to identify:

- what ordinary I-Doser users struggle with;
- what experimenters struggle with in original SBaGen;
- where the two ecosystems fragment or obscure recipes;
- which problems PySbagen already repairs;
- which problems still need a deliberate compatibility/product train.

This document records observed evidence separately from product inference. App-store reviews are examples of reported user experiences, not proof that every user encounters the same issue.

---

## Executive finding

I-Doser and SBaGen fail users in nearly opposite ways:

- **I-Doser is approachable but closed, fragmented, opaque, account-dependent, and repeatedly reported as unreliable.**
- **Original SBaGen is transparent and powerful but unmaintained, text-file-driven, technically demanding, and difficult to use safely or confidently without understanding its syntax and audio environment.**

PySbagen’s opening is not merely “support both formats.” It is:

> Preserve the inspectability and freedom of SBaGen while providing the reliability, discoverability, portability, and ordinary-person experience that I-Doser attempts but does not consistently deliver.

---

# I-Doser pain points

## 1. Playback and download reliability

Recent verified app-store reviews report:

- frequent crashes when attempting to start a dose;
- downloads disappearing and requiring repeated re-downloads;
- temporary region/country errors preventing downloads;
- purchased content becoming difficult to restore after returning to the product.

These reports matter because an audio session depends on uninterrupted playback. A dose that crashes, disappears, or depends on a temporarily unavailable server cannot be trusted as a repeatable experience.

**Sources:**

- https://play.google.com/store/apps/details?id=com.yudiz.idoser
- https://apps.apple.com/us/app/idoser-premium/id295536778

### PySbagen implication

- Local files and locally generated recipes must remain usable without an account, cloud authorization, region check, or vendor server.
- A session should be resumable or restartable without losing its recipe.
- Purchased or imported source material should have a clear local ownership and backup story where licensing permits.

---

## 2. Cloud/account dependence and restoration anxiety

I-Doser’s mobile product uses cloud downloads and restore-purchase flows. Reviews describe account/support difficulty and losing access to previously purchased material. Current product pages also distinguish computer, mobile, and MP3 delivery paths rather than one portable library.

**Sources:**

- https://apps.apple.com/us/app/idoser-premium/id295536778
- https://i-doser.com/about.html

### PySbagen implication

A PySbagen library should be:

- local-first;
- exportable;
- backup-friendly;
- independent of a single vendor account;
- capable of showing whether an item is a recipe, source audio, rendered file, or imported proprietary artifact.

---

## 3. Platform and format fragmentation

I-Doser publicly separates:

- shorter mobile doses with integrated ambient sound;
- longer computer doses described as purer binaural sequences;
- MP3 products;
- desktop DRG-style dose files;
- custom-designed doses.

Historically, I-Doser stated that mobile and desktop products sharing a name could be completely different designs. Current product pages still describe materially different mobile, desktop, and MP3 catalogs and experiences.

**Sources:**

- https://i-doser.com/about.html
- https://www.binauralforum.com/viewtopic.php?t=11893

### PySbagen implication

Do not treat a product name as a stable protocol identity. Imported items need:

- platform/source provenance;
- exact recipe or rendered-audio identity;
- duration;
- channel layout;
- sample rate;
- whether ambient audio is embedded;
- whether the artifact is editable or only playable;
- hashes for source and rendered material.

---

## 4. Recipe opacity

A DRG file extends the SBG lineage with encryption, metadata, and an image. The ordinary I-Doser user receives a branded “dose,” but generally cannot inspect the actual sequence, carriers, transitions, amplitudes, or evidence basis.

**Sources:**

- https://fileinfo.com/extension/drg
- https://uazu.net/sbagen/i-doser.html

### PySbagen implication

When legally and technically possible, PySbagen should expose:

- the decoded schedule;
- tone sets;
- transitions;
- embedded image and metadata;
- warnings for unsupported semantics;
- an exact import report describing what was preserved, approximated, or lost.

An import that merely plays audio is not enough for a research-capable product.

---

## 5. Limited ordinary-user control over the sound bed

An App Store review specifically reports wanting to disable the music layered with a dose but finding no option. I-Doser’s desktop product pages also gate ambient moodscapes and MP3 export behind premium editions.

**Sources:**

- https://apps.apple.com/us/app/idoser-premium/id295536778
- https://i-doser.com/software.html

### PySbagen implication

Pleasant sound, noise, voice, and entrainment layers must be independently controllable when the recipe permits it. The user should be able to know whether a layer is:

- structurally required;
- recommended;
- optional;
- embedded permanently in a rendered source;
- removable because the recipe remains available.

---

## 6. Missing or removed content and weak library transparency

Reviews ask where previously known doses went. A large catalog without clear lifecycle states leaves users unable to tell whether an item was renamed, removed, superseded, region-limited, or simply unavailable on that platform.

**Source:**

- https://apps.apple.com/us/app/idoser-premium/id295536778

### PySbagen implication

Library entries should have explicit states:

- available;
- imported;
- missing source;
- superseded;
- archived;
- incompatible;
- withdrawn;
- research-only.

Never silently hide a recipe or imported item.

---

## 7. Marketing and evidence ambiguity

I-Doser’s public material promotes strong effectiveness language and user-effect percentages, while another official page says the product makes no effectiveness claims and is for entertainment only.

**Sources:**

- https://i-doser.com/about.html
- https://i-doser.com/research.html

### Product inference

This mixed message creates an expectation problem: users are sold a named outcome while the actual protocol, evidence quality, responder variability, and uncertainty remain obscure.

### PySbagen implication

PySbagen should attach an evidence position to a recipe or protocol:

- historical lineage;
- anecdotal use;
- plausible mechanism;
- pilot evidence;
- replicated evidence;
- conflicting evidence;
- unsupported claim.

The product should distinguish “designed for,” “reported by some users,” and “demonstrated to.”

---

## 8. Support and troubleshooting are detached from the session

Reviews report difficulty obtaining useful account or technical support. The product’s historical forum directed order and technical problems away from public discussion into a contact form.

**Sources:**

- https://apps.apple.com/us/app/idoser-premium/id295536778
- https://www.binauralforum.com/viewtopic.php?t=11893

### PySbagen implication

A failed session should produce a local, understandable diagnostic package:

- audio device and route;
- source format;
- decoder used;
- sample rate and channels;
- recipe identity;
- exact error;
- whether output was preserved;
- a privacy-safe export for support.

---

# Original SBaGen pain points

## 1. The project is unmaintained

The official SBaGen site explicitly identifies the original project as unmaintained and points users toward community alternatives.

**Source:**

- https://uazu.net/sbagen/

### PySbagen implication

Compatibility must be based on documented behavior and a growing fixture corpus, not on expecting upstream fixes.

---

## 2. The ordinary workflow requires editing text files

The official description says SBaGen is suited to experimenters who do not mind editing text files. Its power is encoded in a terse schedule language rather than a discoverable interface.

**Source:**

- https://uazu.net/sbagen/

### User pain

- unclear syntax;
- difficult authoring;
- easy-to-miss punctuation semantics;
- no immediate visual understanding of what will happen over time;
- no simple distinction between a tone set, schedule event, transition, audio bed, and output option.

### PySbagen implication

Keep raw SBG editing available, but add:

- timeline visualization;
- syntax-aware editing;
- live validation;
- human explanations;
- reversible visual editing;
- an import report for unsupported constructs.

---

## 3. Time-of-day semantics can produce confusing silence

Original SBaGen schedules can follow the wall clock. Its FAQ explains that users may hear nothing because an overnight sequence is being played during the day unless special options are used.

**Source:**

- https://uazu.net/sbagen/faq.html

### PySbagen implication

The interface must explicitly distinguish:

- play now from the beginning;
- honor original wall-clock timing;
- preview rapidly;
- render a selected interval;
- render the full defined program.

No silent output caused by an invisible timing mode.

---

## 4. Export length and infinite schedules are awkward

The FAQ calls the WAV export process awkward when a schedule has no end; users must edit command options into the SBG file to specify a length.

**Source:**

- https://uazu.net/sbagen/faq.html

### PySbagen implication

Before rendering, show:

- whether the schedule terminates;
- inferred duration;
- requested export duration;
- loop behavior;
- estimated file size;
- what happens after the final event.

---

## 5. Old platform and audio-backend assumptions

The original download/build documentation discusses 32-bit libraries, `/dev/dsp`, Win32 audio calls, old CoreAudio paths, GCC multilib, Windows XP/Vista behavior, DOS, and PocketPC variants. The TODO proposed JACK and a callback-model rewrite.

**Sources:**

- https://uazu.net/sbagen/
- https://uazu.net/sbagen/faq.html

### PySbagen implication

Modern audio handling needs:

- explicit device selection;
- sample-rate negotiation;
- clear resampling reporting;
- channel-routing validation;
- an export-only path that works without live playback dependencies;
- modern backend abstraction rather than OS-specific core logic.

---

## 6. Audio-device processing can invalidate the intended signal

The SBaGen FAQ describes spatializer effects mixing stereo channels and sample-rate conversion causing low-frequency buzzing.

**Source:**

- https://uazu.net/sbagen/faq.html

### PySbagen implication

Add a listening-path qualification tool that can check or explain:

- stereo separation;
- mono downmixing;
- spatial enhancement;
- sample-rate conversion;
- Bluetooth processing;
- equalization and loudness normalization;
- whether headphones or speakers fit the selected method.

---

## 7. Background-audio handling is technically fragile

Original SBaGen documentation requires attention to source sample rate, stereo format, and MP3 joint-stereo behavior. Background sound is powerful, but the user has to understand encoding details that can alter channel separation.

**Source:**

- https://uazu.net/sbagen/faq.html

### PySbagen implication

Every imported or user-provided source should receive an analysis summary:

- codec/container;
- channels;
- sample rate;
- duration;
- clipping/headroom;
- detected mono or near-mono content;
- resampling/conversion applied;
- whether the source is suitable for binaural layering.

---

## 8. Important requested features never entered original SBaGen

The official TODO records demand for:

- an easy GUI;
- a reusable sequencing/audio library;
- independent channels and slides;
- triggered voice or sample cues;
- in-session event marking;
- random variation within ranges;
- global volume control;
- real-time keyboard fades;
- user-event timing records;
- additional modulation and colored noise;
- isochronic tones;
- conversion to and from other formats.

**Source:**

- https://uazu.net/sbagen/

### PySbagen implication

These are not random feature wishes. Together they describe the missing product layer between a synthesis engine and an experimenter’s workstation.

---

# Cross-ecosystem pain points

## 1. No stable, inspectable protocol identity

A name can refer to different mobile, desktop, or rendered products. DRG adds opaque packaging around SBG lineage. Exported MP3/WAV files may no longer preserve the exact recipe.

PySbagen needs a canonical protocol manifest independent of display name or rendered file.

## 2. Weak round-trip portability

The ecosystems do not provide one dependable path among:

- SBG source;
- DRG package;
- rendered WAV/MP3;
- editable visual timeline;
- exact research recipe.

Conversions must state whether they are:

- lossless;
- semantically equivalent;
- approximated;
- rendered-only;
- impossible to reverse.

## 3. No provenance-first library

Users need to know where an item came from, whether it was changed, and whether two similarly named items are actually identical.

## 4. No outcome history tied to exact recipes

Neither lineage gives ordinary users a strong local record of:

- what they played;
- exact parameters;
- listening route;
- what happened;
- whether it helped;
- whether it caused discomfort;
- whether the same recipe was actually repeated.

## 5. No graceful unsupported-feature story

A parser should never silently strip or flatten unfamiliar semantics. Unsupported features must be visible and block a misleading “successful” conversion when necessary.

---

# What PySbagen already repairs

The current PySbagen `main` branch already improves several historic pain points:

- modern Python packaging;
- local generation and export;
- broad user-audio decoding through SoundFile and FFmpeg;
- bounded streaming instead of session-sized buffering;
- atomic output replacement;
- persistent generator state across schedule events;
- functional `off`/bare-dash silence;
- interval crossfades for trailing `->`;
- a guarded desktop studio;
- independent GUI and playback dependencies;
- normal-user Sleep Guide separate from the laboratory interface.

These should not be reopened as unsolved gaps unless new evidence shows a regression.

---

# Priority pain points for the next compatibility train

## Priority 1: honest import report

For every SBG or DRG import, report:

- source type and hash;
- detected encoding/version clues;
- metadata found;
- constructs preserved;
- constructs approximated;
- constructs ignored or unsupported;
- missing referenced audio;
- final inferred duration;
- whether rendering is safe to proceed.

## Priority 2: DRG preservation rather than schedule extraction only

Preserve and expose all available package elements instead of extracting only the schedule and image.

## Priority 3: full SBG semantic compatibility matrix

Build fixtures covering original syntax and explicitly classify every construct as supported, partial, unsupported, or intentionally excluded.

## Priority 4: timeline and source inspector

A user should be able to see a schedule before playing it and inspect every active layer over time.

## Priority 5: audio-path qualification

Make stereo separation, sample-rate conversion, channel routing, and source suitability understandable before a session begins.

## Priority 6: local library and provenance

Create a local-first library that distinguishes recipes, packages, rendered audio, missing sources, and platform variants.

---

## Boundary

This scout does not authorize copying or redistributing proprietary I-Doser content. Compatibility work should operate on files the user lawfully possesses and should preserve source licensing/provenance.

No creativity implementation is part of this pain-point pass.
