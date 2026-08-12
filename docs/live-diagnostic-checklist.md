# Live AUX engine diagnostic checklist

_Manual Lights-laptop evidence for the reusable live source and disposable CLI._

This checklist is intentionally unfilled. Automated tests use synthetic audio only, and no result
in this document should be marked passed until the AUX test is performed on the Lights laptop.
The diagnostic does not hardcode or preserve an audio-device index across runs.

## Timing interpretation

The initial read size is 960 frames. At 48 kHz, that block contains 20 ms of samples and directly
supplies the energy detector's current 960-sample analysis window. The block-fill interval and the
analysis window describe the same captured samples; do not add them as independent sequential
waits. Any PortAudio input-latency value printed by the separate hardware probe is driver-reported
metadata, not a measured physical-event-to-Python or end-to-end latency.

This milestone preserves frame starts and detector event times from integer sample positions so a
later instrumented test can compare a known physical stimulus with an observed result. It does not
claim that latency has been measured, and aggressive latency optimization is out of scope.

## Setup and exact live test

From PowerShell in the repository root, install the optional hardware runtime and enumerate the
current devices:

```powershell
uv sync --extra dev --extra probe
uv run python -m lights_audio_engine.probe devices
```

Identify the external AUX/Realtek input by its current index or a unique name substring. Do not
reuse a remembered index, and do not select the laptop `Microphone Array` unless that is physically
confirmed to be the AUX input. Run the exact live command:

```powershell
uv run python -m lights_audio_engine.diagnostic --device <SELECTOR-OR-INDEX> --sample-rate 48000 --channels 1
```

Play representative party music through the AUX chain for at least 60 seconds. Observe beat and
BPM lines, then press Ctrl+C. Ctrl+C should print `Live diagnostic interrupted.`, close the stream
without a traceback, and exit with code `130`.

## Acceptance observations

Record observed facts; do not tune beat-detection behavior during this hardware check.

| Check | Result | Notes |
| --- | --- | --- |
| External AUX input identified from fresh enumeration |  |  |
| 48 kHz mono stream opened |  |  |
| Beat lines appeared during representative music |  |  |
| BPM appeared after sufficient beat evidence |  |  |
| Overflow/discontinuity lines, if any, were visible |  |  |
| Output was sufficient to judge false or missed beats manually |  |  |
| At least 60 seconds completed before Ctrl+C |  |  |
| Ctrl+C closed cleanly with exit `130` and no traceback |  |  |

## Run record

- Date/time:
- Commit and branch:
- Windows version:
- Python version:
- sounddevice / PortAudio versions:
- Current selected device index and full name:
- Connection path from music source to AUX input:
- Requested sample rate / channels: `48000 / 1`
- Observed beat/BPM behavior:
- Observed discontinuities:
- False/missed-beat notes:
- Shutdown result:
- Follow-up risks or questions:

No entry in this record is an empirical latency measurement unless a separate test defines and
instruments both the physical stimulus timestamp and the observed software timestamp.
