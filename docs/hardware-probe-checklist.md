# Windows audio hardware probe checklist

_Evidence collection for the experimental M1.5 diagnostic sidecar._

The probe reports what sounddevice/PortAudio observes. It does not create stable device
identity, retain raw PCM, feed `AudioEngine`, or establish the production M4 capture design.
Leave every result blank until the corresponding physical test is performed.

## Setup and automated baseline

From PowerShell in the repository root:

```powershell
uv sync --extra dev --extra probe
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run basedpyright
```

Record the commit, branch, Windows version, Python version, sounddevice version, and PortAudio
version with the observations. Device indexes and names are run-local labels only.

### Exit codes

| Exit | Meaning |
| ---: | --- |
| `0` | Command completed; exact-zero, near-full-scale, and overflow findings are warnings |
| `2` | Invalid command syntax or non-positive numeric argument |
| `3` | Optional sounddevice/PortAudio backend unavailable |
| `4` | No usable input or invalid, stale, output-only, missing, or ambiguous selector |
| `5` | Unsupported input configuration or exclusive mode on a non-WASAPI device |
| `6` | Capture/runtime failure, short read, or non-finite captured samples |
| `130` | Capture interrupted with Ctrl+C; partial diagnostics are printed |

## Laptop microphone checks for August 10, 2026

1. Enumerate host APIs, devices, and defaults:

   ```powershell
   uv run python -m lights_audio_engine.probe devices
   ```

2. Choose the laptop input's current index and check its shared-mode matrix:

   ```powershell
   uv run python -m lights_audio_engine.probe check --device <INDEX>
   ```

3. Capture three seconds in a quiet room, while speaking or playing music, and during one
   sharp clap. Record per-channel RMS, peak, exact-zero, near-full-scale, non-finite, and
   overflow findings; do not turn them into thresholds.

   ```powershell
   uv run python -m lights_audio_engine.probe capture --device <INDEX> --duration 3
   ```

4. Repeat the three-second capture five times, then run a 30-second capture. Confirm every
   process terminates and note overflowed reads.

   ```powershell
   1..5 | ForEach-Object { uv run python -m lights_audio_engine.probe capture --device <INDEX> --duration 3 }
   uv run python -m lights_audio_engine.probe capture --device <INDEX> --duration 30
   ```

5. Exercise failures: invalid selector, ambiguous substring, output-only index, sample rate
   `12345`, channels `64`, negative duration, and exclusive mode on both WASAPI and non-WASAPI
   entries. Confirm the documented exit codes and that unsupported formats do not open a
   stream.

6. Interrupt a 30-second capture with Ctrl+C. Expect partial diagnostics and exit `130`.

7. If a removable USB or Bluetooth input is safely available, disconnect it during a
   30-second capture. Record whether the backend errors, stalls, terminates, or returns zeros.
   Skip this step for a built-in microphone and do not generalize the result to Lights hardware.

8. Run the combined report without a selector and verify its printed WASAPI-default or global-
   default fallback explanation:

   ```powershell
   uv run python -m lights_audio_engine.probe report
   ```

### Laptop-only results — August 10, 2026

These observations describe the built-in laptop microphone only. They are not acceptance
results for the actual Lights hardware and do not establish stable device identity, Lights
hardware compatibility, loopback support, the production `AudioSource` design, or direct
line-level input behavior.

Environment:

- Windows
- sounddevice 0.5.5
- PortAudio V19.7.0-devel
- WASAPI default input during this enumeration: device index `9`, `Microphone Array
  (Realtek(R) Audio)`
- Device index `9` is a run-local label and must not be treated as stable identity.

Shared-mode compatibility observed for device index `9`:

| Requested format | Observed result | Detail |
| --- | --- | --- |
| 44.1 kHz × 1 channel | Unsupported | `Invalid sample rate [PaErrorCode -9997]` |
| 44.1 kHz × 2 channels | Unsupported | `Invalid sample rate [PaErrorCode -9997]` |
| 48 kHz × 1 channel | Supported | WASAPI shared mode |
| 48 kHz × 2 channels | Supported | WASAPI shared mode |

Capture and report evidence:

| Check | Requested / setup | Observed result | Termination |
| --- | --- | --- | --- |
| Three-second capture | 144,000 frames | Received 144,000 frames at 48,000 Hz; reported input latency 22 ms; zero overflowed reads; zero non-finite samples | Completed normally |
| Music capture | 30 seconds / 1,440,000 frames | Received 1,440,000 frames; duration 30.000000 s at 48,000 Hz; RMS 0.03892087; peak 0.48922929; zero near-full-scale samples; zero non-finite samples; zero overflowed reads | Completed normally |
| Five repeats | 5 × 3 seconds | Every run received 144,000 / 144,000 frames, opened at 48 kHz, and reported 22 ms latency; zero overflows across all five runs; zero non-finite samples; no near-full-scale samples | All five completed normally |
| Ctrl+C | Requested 30 seconds | Interrupted after 252,000 frames / 5.25 seconds; partial diagnostics preserved; actual rate 48 kHz; zero overflowed reads; zero non-finite samples | Correctly reported as interrupted |
| Stability capture | 5 minutes / 14,400,000 frames | Received 14,400,000 frames; duration 300.000000 s at 48,000 Hz; reported input latency 22 ms; RMS 0.03938381; peak 0.55860990; zero near-full-scale samples; zero non-finite samples; zero overflowed reads | Completed normally |
| Default report | `report` | Selected the WASAPI default input successfully, reproduced the compatibility matrix, and completed its built-in three-second capture successfully | Completed normally |

The laptop run did not establish results for the separate quiet-room, sharp-clap, invalid-
input, exclusive-mode, or removable-device disconnect checks. Those observations remain
unrecorded and must not be inferred from the successful captures above.

## Lights hardware acceptance for tomorrow

Collect evidence without changing code unless the probe itself is broken.

**Status: pending.** None of the laptop-only observations above completes or changes any
Lights hardware acceptance item below.

1. Determine whether the feed entering Windows is a physical input or requires Windows
   playback loopback. If loopback is required, stop and record that dependency finding;
   stock sounddevice does not provide WASAPI loopback capture.
2. Map all PortAudio/host-API entries to the real interface.
3. Run shared and explicit-format exclusive compatibility matrices.
4. Compare the native/default rate, requested rate, and actual opened stream rate.
5. Identify channels that carry known program material.
6. Measure the quiet-chain RMS per channel three times; do not invent a silence threshold.
7. Exercise representative loud material and record peak and near-full-scale counts. The
   `abs(sample) >= 0.999` heuristic is not proof of ADC clipping.
8. Record reported input latency in shared and exclusive modes.
9. Run at least two 60-second captures and record overflowed reads and any frame shortfall.
10. Unplug the interface during capture and record error, stall, or clean termination behavior.
11. Replug and reboot, then compare indexes and names without treating either as identity.
12. Append the resulting M4 implications: endpoint identity, loopback-capable dependency,
    watchdog, resampling, reconnect policy, and channel mapping.

### Tomorrow results

| Evidence question | Shared mode | Exclusive mode | Notes / M4 implication |
| --- | --- | --- | --- |
| Physical input or loopback |  |  |  |
| Host API and device entry |  |  |  |
| Supported rates/channels |  |  |  |
| Default/requested/actual rate |  |  |  |
| Signal-bearing channels |  |  |  |
| Quiet RMS, three runs |  |  |  |
| Loud peak / near-full-scale |  |  |  |
| Reported latency |  |  |  |
| Two captures of at least 60 s |  |  |  |
| Unplug behavior |  |  |  |
| Replug/reboot labels |  |  |  |

## M4 implications memo

- Endpoint identity:
- Loopback requirement and dependency:
- Watchdog or stall detection:
- Resampling:
- Reconnect policy:
- Channel mapping:
- Other hardware-specific constraints:
