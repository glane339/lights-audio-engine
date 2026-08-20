# Aubio causal benchmark

This is an optional, **causal advisory** benchmark for replaying one existing M2C PCM artifact
through Aubio. It is not a production detector, is not an M2C bake-off candidate, does not
participate in M2C ranking or recommendation gates, and does not use live capture or `AudioEngine`.

Until human-authored point labels exist, its output is comparison evidence only: it is neither
detector accuracy nor ground truth and must not be promoted automatically to a reference.

## Install

The optional dependency is intentionally separate from normal development installs:

```powershell
uv sync --extra aubio
```

The installed distribution is `aubio-ledfx` (validated on Windows with Python 3.12); its Python
API continues to be imported as `aubio`. Normal project installs do not include this extra.

## Run one artifact

```powershell
uv run python -m lights_audio_engine.evaluation.aubio_bench `
  .\m2c-artifacts\dev-house-noenh-01.npy `
  --output .\m2c-artifacts\dev-house-noenh-01.aubio-report.json
```

To export those exact report event timestamps as a Sonic Visualiser Time Instants layer in the
same run:

```powershell
uv run python -m lights_audio_engine.evaluation.aubio_bench `
  .\m2c-artifacts\dev-house-noenh-01.npy `
  --output .\m2c-artifacts\dev-house-noenh-01.aubio-report.json `
  --sonic-visualiser .\m2c-artifacts\dev-house-noenh-01.aubio.sv.txt
```

The annotation file is tab-separated `timestamp-in-seconds<TAB>aubio` Time Instants data. It is a
derived, local review artifact and is intentionally not tracked by Git.

For a discontinuous artifact, supply `--segment-index`. `--delivery-block-size` defaults to 240
frames to mirror the 5 ms M2C replay delivery experiment; it does not change the Aubio algorithm's
own internal hop size.

## Causal and timestamp semantics

The adapter creates Aubio's native `onset("default", 1024, 256, sample_rate_hz)` detector and
passes it exact normalized artifact samples, once, in causal order. Incoming replay frames are only
buffered until a full 256-sample native hop is available. The final incomplete hop is not padded or
flushed, so no synthetic future audio is supplied.

Each event's `timestamp_seconds` is Aubio's `get_last_s()` value, relative to the selected segment's
rebased zero origin. `emitted_stream_time_seconds` is the end of the replay frame in which the
adapter returns that event. Therefore `decision_latency_seconds` is the defensible causal replay
observation `emitted_stream_time_seconds - timestamp_seconds`; it includes any delivery-frame delay
in the same way as M2C's existing runner. The report emits a binary `strength: 1.0` because Aubio's
raw onset descriptor is not a normalized, cross-method strength scale.

The JSON report always identifies itself as `kind: "aubio_causal_advisory_benchmark"`,
`advisory_only: true`, `production_candidate: false`, and `ground_truth: false`.

Keep real audio, M2C artifacts, annotations, and generated reports local and uncommitted.

## Evidence status

### Measured benchmark observations

For the replayed `dev-house-noenh-01.npy` capture, the JSON report provides deterministic event
timestamps and causal decision-latency observations. These are benchmark measurements only; they
do not establish beat-detection accuracy, precision, recall, F1, or production readiness.

### Manual perceptual observations

Manual Sonic Visualiser inspection of the Aubio Time Instants export against the recording, desired
lighting reactions, and the QM Vamp beat layer suggested that Aubio was more closely aligned with
the desired lighting events on this recording and that QM Vamp omitted many events of interest.
The broader inspection informally suggested roughly 95% coverage, but that is not a measured recall
claim. Manual beat-by-beat inspection of the 25.0–30.0 s segment found Aubio aligned with every
desired lighting beat in that inspected segment. Aubio also missed some desired events elsewhere
in the recording. Authoritative human-reference labels are still required for quantitative accuracy
claims or any production decision.
