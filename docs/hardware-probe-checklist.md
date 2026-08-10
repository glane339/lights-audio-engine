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

## Laptop microphone checks for today

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

### Today results

| Check | Command or setup | Observed result | Exit | Notes |
| --- | --- | --- | ---: | --- |
| Inventory/defaults | `devices` |  |  |  |
| Shared compatibility | `check --device <INDEX>` |  |  |  |
| Quiet capture | 3 seconds |  |  |  |
| Speech/music capture | 3 seconds |  |  |  |
| Sharp-clap capture | 3 seconds |  |  |  |
| Five repeats | 5 × 3 seconds |  |  |  |
| Longer capture | 30 seconds |  |  |  |
| Invalid inputs | selector/rate/channels/duration |  |  |  |
| Exclusive mode | WASAPI and non-WASAPI |  |  |  |
| Ctrl+C | interrupt 30 seconds |  |  |  |
| Safe removable-device disconnect | optional |  |  |  |
| Default report | `report` |  |  |  |

## Lights hardware acceptance for tomorrow

Collect evidence without changing code unless the probe itself is broken.

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
