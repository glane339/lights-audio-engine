# M2B beat characterization protocol

_Operational baseline for real-audio detector evidence on the Lights laptop_

---

## 🎯 Purpose

Use the existing live capture and beat detector to produce repeatable JSONL evidence before
changing detector behavior. This protocol characterizes detected beats, interval stability,
tempo estimates, input level, discontinuities, and diagnostic software processing time. It does
not re-prove the already validated capture path and does not measure acoustic end-to-end latency.

Detector changes begin only after repeated runs demonstrate a reproducible baseline failure.

## 🔧 Prepare the laptop

From PowerShell in the repository root, install the existing optional hardware runtime and
enumerate devices afresh:

```powershell
uv sync --extra dev --extra probe
uv run python -m lights_audio_engine.probe devices

$m2bLogDir = Join-Path $env:USERPROFILE 'Documents\lights-audio-engine-m2b'
New-Item -ItemType Directory -Force -Path $m2bLogDir
```

Select the external AUX/Realtek input by its current index or a unique name substring. Device
indexes are enumeration-local; do not reuse a remembered index without checking it.

## 🎵 Run the baseline set

Keep the track section, Windows input settings, physical routing, and CLI detector settings fixed
except where the matrix explicitly varies input level.

| Test | Repeats | Label pattern | Evidence target |
| --- | ---: | --- | --- |
| Steady four-on-the-floor house/EDM section | 3 | `house-steady-normal-r1` through `r3` | Repeatability and interval stability |
| Same section at low, normal, and high clean input | 1 each | `house-level-low`, `house-level-normal`, `house-level-high` | Input-level sensitivity without clipping |
| Approximately 170–180 BPM track | 1 or more | `high-bpm-175-r1` | Misses and half-time collapse |
| Pause → silence → resume → skip | 1 | `transport-boundaries-r1` | False beats, reacquisition, and discontinuities |

### First hardware baseline run

Replace `<FRESH_DEVICE_SELECTOR>` with the current external input selector and `<EXPECTED_BPM>`
with the known tempo. If the tempo is unknown, omit `--expected-bpm` from the later report command.

```powershell
$m2bLog = Join-Path $m2bLogDir 'house-steady-normal-r1.jsonl'

uv run python -m lights_audio_engine.diagnostic `
  --device "<FRESH_DEVICE_SELECTOR>" `
  --sample-rate 48000 `
  --channels 1 `
  --sensitivity 0.5 `
  --min-bpm 50 `
  --max-bpm 240 `
  --label "house-steady-normal-r1" `
  --log-jsonl $m2bLog
```

Start the chosen section, let the representative passage run, then press Ctrl+C. The existing
console beat/BPM output remains visible while the JSONL sidecar is active.

### Offline report

```powershell
uv run python -m lights_audio_engine.diagnostic.report $m2bLog --expected-bpm <EXPECTED_BPM>
```

To compare repeats, pass each path explicitly:

```powershell
uv run python -m lights_audio_engine.diagnostic.report `
  (Join-Path $m2bLogDir 'house-steady-normal-r1.jsonl') `
  (Join-Path $m2bLogDir 'house-steady-normal-r2.jsonl') `
  (Join-Path $m2bLogDir 'house-steady-normal-r3.jsonl') `
  --expected-bpm <EXPECTED_BPM>
```

## 📋 Record each run

Capture this information beside each log:

| Field | Value to record |
| --- | --- |
| Label | Exact `--label` value |
| Track/section | Track identity and reproducible section boundaries |
| Expected BPM | Known tempo or `unknown` |
| Input-level notes | Low/normal/high setting, clipping observations, Windows setting |
| JSONL path | Exact local path |
| Perceived behavior | Obvious misses, false beats, or double beats with approximate times |
| Report summary | Beat count, interval median/IQR, flags, BPM summary, processing stats |

Do not commit music, captured audio, or local characterization logs. Share only the evidence
needed for the next detector decision through the project-approved path.

## ⏱️ Interpret timing correctly

- `beat_timestamp_seconds` is derived from stream samples and identifies the detector's event
  position within the current logical stream.
- `host_time_seconds` is when the diagnostic consumer observed or recorded the result relative to
  session start.
- PortAudio or driver-reported capture latency is metadata, not measured acoustic latency.
- `mean_process_seconds` and `max_process_seconds` measure synchronous software processing only.
- Differences between stream time and host time include unknown buffering and offsets. They must
  not be called end-to-end latency or acoustic-event-to-ping latency.

## ✅ Choose the next action

| Baseline outcome | Recommended next action |
| --- | --- |
| Detector looks good | Repeat on broader party-music sections before proposing any detector change |
| Obvious missed-beat problem | Reproduce on the same section and level in at least two runs, then scope missed-onset diagnosis |
| Obvious false/double-beat problem | Confirm short-interval flags and perceived false beats repeat, then scope refractory/onset diagnosis |
| Half-time collapse at high BPM | Confirm the 170–180 BPM result repeats and distinguish missed beats from estimator behavior |
| Strong input-level sensitivity | Compare low/normal/high clean runs, verify no clipping, then scope level normalization or detector robustness separately |

Do not tune the detector from a single run. Preserve these logs as the Phase 1 baseline for any
later detector-change acceptance test.
