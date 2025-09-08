# Test Plan

This document outlines the test plan for the `pysbagen` application.

## 1. Unit Tests

### 1.1. Generator Classes

- **`ToneSpec`**:
  - Verify that the generated audio has the correct shape and data type.
  - Verify that the frequency information is correct.
- **`NoiseSpec`**:
  - Verify that the generated audio has the correct shape and data type.
  - Verify that the frequency information is correct.
  - Test both "white" and "pink" noise.
- **`FileSpec`**:
  - Verify that the generated audio has the correct shape and data type.
  - Verify that the frequency information is correct.
  - Test with WAV, OGG, and MP3 files.
  - Test the looping functionality.
- **`IsochronicSpec`**:
  - Verify that the generated audio has the correct shape and data type.
  - Verify that the frequency information is correct.
- **`HarmonicBoxSpec`**:
  - Verify that the generated audio has the correct shape and data type.
  - Verify that the frequency information is correct.
- **`GenericToneSpec`**:
  - Verify that the generated audio has the correct shape and data type.
  - Verify that the frequency information is correct.
  - Test all supported waveforms (sine, square, triangle, sawtooth).

### 1.2. Parser

- **`parse_sbg`**:
  - Test with a valid `.sbg` file.
  - Test with an empty `.sbg` file.
  - Test with an `.sbg` file with comments.
  - Test with an `.sbg` file with invalid syntax.
- **`parse_sbg_from_string`**:
  - Test with a valid `.sbg` string.
  - Test with an empty string.

### 1.3. Mixer

- **`mix_generators`**:
  - Test with a single generator.
  - Test with multiple generators.
  - Verify that the output audio has the correct shape and data type.
  - Verify that the peak level is within the correct range.
- **`build_session_generator`**:
  - Test with a valid schedule.
  - Test with an empty schedule.

### 1.4. DRG Decoder

- **`decode_drg`**:
  - Test with a sample `.drg` file (if one can be obtained).
  - If no sample file is available, create a synthetic `.drg` file and test with that.
  - Verify that the decoded `.sbg` data is correct.
  - Verify that the extracted image data is correct.

### 1.5. Visualization

- **`generate_chladni_pattern`**:
  - Test with different `(n, m)` pairs.
  - Verify that the function returns a valid matplotlib figure.
- **`map_freq_to_params`**:
  - Test with different frequencies.
  - Verify that the function returns a valid `(n, m)` pair from the `GOOD_PAIRS` list.

## 2. Integration Tests

- **GUI and Controller**:
  - Test the interaction between the `SbagenGui` and `SbagenController` classes.
  - Verify that the GUI correctly calls the controller methods and that the controller returns the expected results.
- **GUI and Audio Engine**:
  - Test the end-to-end functionality of the application, from the GUI to the audio engine.
  - This will be a "dry run" test, as I cannot hear the audio output. I will verify that the correct functions are called and that no errors occur.

## 3. GUI Tests (Manual)

- **Main Window**:
  - Verify that the main window and all its widgets are displayed correctly.
- **Quick Generate Tab**:
  - Verify that the input fields and the "Generate" button are working correctly.
- **Schedule File Tab**:
  - Verify that the file browser opens and that the selected file is displayed.
  - Verify that the "Generate from Schedule" button is working correctly.
- **Advanced Tab**:
  - Verify that all the input fields and buttons are working correctly.
- **Tone Generator Tab**:
  - Verify that the "Add Tone" and "Remove" buttons are working correctly.
  - Verify that the sliders and dropdown menus are working correctly.
  - Verify that the "Generate Tones" button is working correctly.
- **Visualization Tab**:
  - Verify that the matplotlib canvas is displayed correctly.
  - Verify that the "Play with Visualization" button is working correctly.
- **File Menu**:
  - Verify that the "Open I-Doser file" and "Exit" menu items are working correctly.

## 4. Testing in a Difficult Environment

- **Audio Playback**:
  - I cannot hear the audio output in this environment. To test the audio playback, I will create a mock `pyaudio` object that simulates the behavior of the real `pyaudio` library. This will allow me to verify that the correct functions are called and that the audio data is being written to the stream correctly.
- **Visualization**:
  - I cannot see the GUI in this environment. To test the visualization, I will save the generated matplotlib figures to a file and then inspect them manually. This will allow me to verify that the patterns are being generated correctly.
