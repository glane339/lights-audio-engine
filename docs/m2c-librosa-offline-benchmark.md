# Librosa offline benchmark

This is an optional, **offline advisory** aid for reviewing one existing M2C PCM artifact. It is
not a production detector, is not Candidate A/B/C (or a new candidate), does not participate in
M2C recommendation gates, and does not use live capture or `AudioEngine`.

Human/DJ point labels remain the only authoritative ground truth. Librosa beat estimates may be
exported for Audacity review or compared against a human reference, but are never promoted to a
reference automatically. Because Librosa analyzes the full selected segment, its report records
decision latency as `not applicable`; it exposes no numeric causal-latency measurement.

## Install

The optional dependency is intentionally separate from the normal development install:

```powershell
uv sync --extra librosa
```

## Run one artifact

Use one schema-versioned M2C artifact and choose a segment explicitly when it contains a
discontinuity:

```powershell
python -m lights_audio_engine.evaluation.librosa_bench `
  .\m2c-artifacts\holdout-track-01.npy `
  --output .\m2c-artifacts\holdout-track-01.librosa-report.json `
  --audacity-labels .\m2c-artifacts\holdout-track-01.librosa-beats.txt `
  --candidate-reference .\m2c-artifacts\holdout-track-01.librosa-candidate.txt `
  --envelope-npy .\m2c-artifacts\holdout-track-01.envelope.npy `
  --human-reference .\m2c-artifacts\holdout-track-01.human.txt
```

`--human-reference` is optional and must name a human-authored point-label file. `--tolerance-ms`
defaults to `50.0`. The output JSON always identifies itself as
`kind: "librosa_offline_benchmark"`, `advisory_only: true`, and
`production_candidate: false`.

The `*.librosa-beats.txt` file is an Audacity point-label convenience export. The
`*.librosa-candidate.txt` file uses M2C's two-column reference grammar solely for side-by-side
review; it is not a human reference or an input to production recommendation logic.

Keep real audio, M2C artifacts, annotations, and generated reports local and uncommitted.
