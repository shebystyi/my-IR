# Neural Guitar IR Reconstruction Studio (v0.1)

Estimates a guitar cabinet/chain **impulse response** from reference recordings and exports it as a
48 kHz / 24-bit WAV for custom IR loaders (e.g. Neural DSP). DSP core is independent of the GUI.

## Install (Windows 10/11, Python 3.11+)
```
run.bat            # creates .venv, installs requirements, starts the GUI
run_tests.bat      # runs the automated tests
```
Headless: `python -m nigr.cli paired --di di.wav --target amp.wav --out irs/ --ms 100 --auto-reg`

## What it does
| Case | Input | Result |
|---|---|---|
| 1. Paired linear | DI + processed WAV(s) | Regularised least-squares FIR (frequency-domain Wiener/Tikhonov or time-domain Toeplitz), per-frequency **coherence** shows where the linear model is valid. Physically meaningful IR. |
| 2. Nonlinear | DI + processed | Indicators only: nonlinear residual, compression vs level, tone change vs level, clipping character. Exports aligned pair for NAM-style training. **Never written into the IR.** |
| 3. No DI | finished recordings | Minimum-phase *spectral match* (optional: your guitar through the plugin as "source"). **Not a recovered IR**; flagged everywhere. |

Candidates: Balanced, Aggressive, Tight_Punchy, Deep_Thick, Bright_Articulate, Max_Similarity (no tonal shaping).
Shaping = linear minimum-phase EQ + windowing on the IR. There is deliberately no "saturation", "attack" or "release" knob on the IR.

Metrics: log-spectral distance, multi-resolution STFT (spectral convergence + log-mag L1), time-domain NMSE, envelope difference, per-band error. No single similarity %.

## Verified (automated tests, 21)
Known-FIR recovery (~-48 dB error, freq & time methods), phase/delay preservation of a non-minimum-phase IR, noise robustness, stereo, numerical stability, 24-bit WAV round-trip, resampling, export length/normalisation/no-clipping, nonlinear detection, approximate-mode flagging.

## Honest limitations
* A mathematically exact IR **cannot** be recovered from arbitrary mastered recordings. Without a DI, the source spectrum is unknown; only a spectral match is possible.
* An IR is linear and time-invariant: amp distortion, compression, pick-dynamics response and level-dependent tone are outside it. Use a NAM capture for those.
* Mastering EQ/compression/reverb in the reference are folded into the estimate (or the residual).
* Best paired results: broadband DI material (varied notes, palm mutes, chords), ≥ 20 s, same level/gain as the target, no time-varying effects (delay/chorus).
* Neural DSP's loader limits (max length, channel handling) are not verified here: test exported files in the plugin. 20/100/200 ms presets are provided.
* NAM: the official trainer expects its own calibrated input signal; arbitrary DI pairs are only usable in custom training setups. No NAM training is bundled.
* Not yet done: GPU/PyTorch modules, true real-time (streaming) A/B, GUI tested on a real Windows desktop (see status in release notes), listening tests inside Neural DSP.
