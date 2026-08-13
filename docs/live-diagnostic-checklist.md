# Live AUX engine diagnostic checklist

_Completed manual Lights-laptop evidence for the reusable live source and disposable CLI._

This record covers the Lights laptop's physical AUX path, stable capture, and live-engine
integration. The earlier built-in laptop-microphone observations in
`docs/hardware-probe-checklist.md` remain a separate M1.5 probe record. The diagnostic does not
hardcode or preserve an audio-device index across runs.

## Timing interpretation

The initial read size is 960 frames. At 48 kHz, that block contains 20 ms of samples and directly
supplies the energy detector's current 960-sample analysis window. The block-fill interval and the
analysis window describe the same captured samples; do not add them as independent sequential
waits. Any PortAudio input-latency value printed by the separate hardware probe is
driver-reported metadata, not a measured physical-event-to-Python or end-to-end latency.

This milestone preserves frame starts and detector event times from integer sample positions so a
later instrumented test can compare a known physical stimulus with an observed result. It does not
claim that latency has been measured, and aggressive latency optimization is out of scope.

## Hardware and physical input path

The validated physical path was:

```text
music laptop
    -> analog AUX splitter
    -> one path to speaker
    -> one path to Lights laptop external Realtek jack configured as Mic In
    -> Windows / PortAudio
    -> lights-audio-engine
```

The Lights laptop ran Windows with Python 3.12.13, `sounddevice` 0.5.5, and PortAudio
V19.7.0-devel. In the validated fresh enumeration:

- `[12] Microphone (Realtek(R) Audio)` was the Windows WASAPI external/default input.
- `[11] Microphone Array (Realtek(R) Audio)` was the built-in microphone array.

These indexes are run-local observations only and must not be treated as persistent device
identity. Windows-level input checks produced about 12% on the external Realtek input with wired
music and about 1% when clapping near the laptop with music paused. This supports the conclusion
that the selected input primarily received the wired AUX signal rather than the built-in array.

## Setup and exact live test

From PowerShell in the repository root, install the optional hardware runtime and enumerate the
current devices:

```powershell
uv sync --extra dev --extra probe
uv run python -m lights_audio_engine.probe devices
```

Identify the external AUX/Realtek input by its current index or a unique name substring. Do not
reuse a remembered index, and do not select the laptop `Microphone Array` unless that is
physically confirmed to be the AUX input. The successful live diagnostic used this selector after
fresh enumeration:

```powershell
uv run python -m lights_audio_engine.diagnostic --device 12 --sample-rate 48000 --channels 1
```

Representative real music was used during multiple extended sessions. Ctrl+C exited cleanly and
printed `Live diagnostic interrupted.`.

## Windows-level format compatibility

The external WASAPI input had this compatibility matrix:

| Requested format | Observed result |
| --- | --- |
| 44.1 kHz x 1 channel | Unsupported |
| 44.1 kHz x 2 channels | Unsupported |
| 48 kHz x 1 channel | Supported |
| 48 kHz x 2 channels | Supported |

## Capture stability evidence

All successful captures below opened at 48 kHz and reported zero non-finite samples and zero
overflowed reads. The near-full-scale probe heuristic is `abs(sample) >= 0.999`; a zero count is
not proof that the input was never close to clipping.

| Capture | Observed result |
| --- | --- |
| Initial 10 seconds | Requested and received 480,000 frames; reported input latency 0.022 seconds; RMS 0.00421924; peak 0.02420576; near-full-scale 0; non-finite 0; overflowed reads 0 |
| Higher-gain 10 seconds | RMS 0.04824046; peak 0.28125125; near-full-scale 0; non-finite 0; overflowed reads 0 |
| Excessively high-gain 60 seconds | Requested and received 2,880,000 frames; reported input latency 0.022 seconds; RMS 0.31070468; peak 0.99438745; near-full-scale 0; non-finite 0; overflowed reads 0. This level was intentionally judged too hot because the peak was extremely close to full scale. |
| Healthy representative 10 seconds | RMS 0.09687306; peak 0.44667122; near-full-scale 0; non-finite 0; overflowed reads 0 |

The `0.022`-second value is PortAudio-reported input-latency metadata. It is not a measured
physical-event-to-Python latency or measured end-to-end system latency.

## Live engine integration evidence

The successful end-to-end path was:

```text
AUX input
    -> sounddevice / PortAudio
    -> SoundDeviceAudioSource
    -> FrameAssembler
    -> existing run_engine()
    -> AudioEngine
    -> terminal BeatEvent / BPM output
```

The live runs established the following:

- live frames reached `AudioEngine`;
- real `BeatEvent` output appeared;
- BPM estimates appeared;
- no discontinuity messages were observed in the collected successful runs; and
- Ctrl+C shut down cleanly with `Live diagnostic interrupted.`.

The integration and hardware path therefore passed its intended diagnostic milestone. No code
changes were made on the Lights laptop during testing. After pulling
`feat/live-engine-diagnostic`, repository validation on that laptop reported `207 passed`.

## Detector-quality limitations

These observations concern detector and tempo quality, not capture-path failures.

Across complex music, some sections tracked consecutive beats well, while other sections had long
detection gaps. BPM sometimes stabilized in plausible ranges, but the estimator repeatedly fell
to half-time when alternate beats were missed; some runs also produced inconsistent higher/lower
tempo estimates. Observed examples included plausible regions around 138-147 BPM and half-time
regions around 68-72 BPM. One run showed an extended detection gap around 29-45 seconds, and
another showed a long gap around 39-74 seconds. Strong regular transients often produced closely
spaced, consistent beat timestamps; weaker or changing transient structure often caused missed
beats or reacquisition problems.

Controlled known-tempo testing likewise showed unstable beat extraction and tempo estimation,
with estimates ranging roughly across 101 BPM, 114-120 BPM, 129 BPM, 140-146 BPM, and 170-180
BPM rather than one stable tempo.

## Result / verdict

- Live hardware/capture diagnostic: **PASS**
- Live engine integration: **PASS**
- Production-quality beat tracking: **NOT YET PASSING**

The remaining problem is primarily beat/onset extraction and tempo stability, not the
hardware/capture path. Detector improvement belongs in a later milestone or branch and is outside
this diagnostic milestone.

## Run record

- Branch: `feat/live-engine-diagnostic`
- Operating system: Windows
- Python: 3.12.13
- sounddevice / PortAudio: 0.5.5 / V19.7.0-devel
- Selected device during the validated live run: `[12] Microphone (Realtek(R) Audio)`
- Built-in microphone observed separately: `[11] Microphone Array (Realtek(R) Audio)`
- Connection path: music laptop -> analog AUX splitter -> speaker and Lights laptop external
  Realtek jack configured as Mic In
- Requested sample rate / channels: `48000 / 1`
- Live command: `uv run python -m lights_audio_engine.diagnostic --device 12 --sample-rate 48000 --channels 1`
- Observed beat/BPM behavior: BeatEvent and BPM output appeared, with missed beats, gaps, and
  repeated half-time or otherwise unstable tempo estimates in complex and controlled material
- Observed discontinuities: None in the collected successful runs
- Shutdown result: Clean Ctrl+C shutdown with `Live diagnostic interrupted.`; exact exit code was
  not separately recorded in the supplied evidence
- Repository validation after pulling the branch: `207 passed`

Device indexes and the PortAudio latency field are run-local/driver-reported observations. No
entry in this record is an empirical latency measurement unless a separate test defines and
instruments both the physical stimulus timestamp and the observed software timestamp.
