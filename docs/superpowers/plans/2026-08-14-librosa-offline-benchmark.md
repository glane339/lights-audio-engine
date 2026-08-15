# Librosa offline benchmark — implementation plan

_Implements the approved design at
[docs/superpowers/specs/2026-08-14-librosa-offline-benchmark-design.md](../specs/2026-08-14-librosa-offline-benchmark-design.md).
Planning artifact only — no production code has been written._

- **Status:** ready for implementation
- **Branch:** `feat/librosa-offline-benchmark`
- **Workflow:** [docs/ai/WORKFLOW.md](../../ai/WORKFLOW.md) — `PLAN -> IMPLEMENT -> DETERMINISTIC VALIDATION -> INDEPENDENT REVIEW -> REPAIR IF REQUIRED -> FINAL VERIFICATION -> MERGE`, RED/GREEN TDD per task, one commit per task boundary.

## How to read this plan

Each task lists, in order: exact files, interfaces consumed/produced, the failing test to write
first, the exact command that proves it fails for the *expected* reason (missing module/attribute,
not a typo), the minimal implementation, the exact command that proves it then passes, and the
commit to make. Signatures given are the ones implementation must produce — this plan pins them
down so review can check the eventual diff against it directly.

All commands are shown as plain `python -m ...` invocations; they run unmodified in PowerShell or
Bash. Extras are always quoted (`".[librosa]"`) to match the existing `ci.yml` convention.

---

## Task 0 — Preconditions (no files changed)

Confirm the baseline is green before touching anything, so any later red test is attributable to
this feature, not pre-existing breakage:

```
python -m pytest
python -m ruff format --check .
python -m ruff check .
python -m basedpyright
```

All four must currently pass (they do — this branch starts from a clean `main` merge per the
gitStatus at session start). If any fails, stop and resolve it before proceeding; it is out of
scope for this feature.

---

## Task 1 — Optional `librosa` dependency extra

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_librosa_bench_import_boundary.py`

**Interfaces produced:** a new `[project.optional-dependencies.librosa]` key. No Python interface
yet.

**Design constraints enforced here:** decision 1 (extra name `librosa`), decision 6 (not added to
`dev` or mandatory CI).

### RED

Create `tests/test_librosa_bench_import_boundary.py` with the dependency-boundary half of its
final content (the import-isolation test is added later, in Task 7, once there's something to
isolate — see that task's note):

```python
from __future__ import annotations

import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


def test_core_dependencies_exclude_librosa_and_extra_is_declared() -> None:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    core_dependencies = data["project"]["dependencies"]
    assert not any(dep.lower().startswith("librosa") for dep in core_dependencies)

    optional = data["project"]["optional-dependencies"]
    assert "librosa" in optional
    assert any(dep.lower().startswith("librosa") for dep in optional["librosa"])
    assert not any(dep.lower().startswith("librosa") for dep in optional.get("dev", []))
```

Command:

```
python -m pytest tests/test_librosa_bench_import_boundary.py -q
```

Expected failure: `KeyError: 'librosa'` from `optional["librosa"]` — the extra does not exist yet.
This is the expected reason (missing config), not a test bug.

### GREEN

Add to `pyproject.toml`, immediately after the existing `probe` extra:

```toml
librosa = [
    "librosa>=0.10,<1",
]
```

Command:

```
python -m pytest tests/test_librosa_bench_import_boundary.py -q
```

Expected: 1 passed.

Also re-run the full baseline to confirm the `pyproject.toml` edit didn't disturb anything else:

```
python -m pytest
python -m ruff check .
```

### Commit

```
git add pyproject.toml tests/test_librosa_bench_import_boundary.py
git commit -m "chore: add optional librosa benchmark dependency extra"
```

---

## Task 2 — Isolated Librosa backend wrapper

**Files:**
- Create: `src/lights_audio_engine/evaluation/librosa_bench/__init__.py`
- Create: `src/lights_audio_engine/evaluation/librosa_bench/_librosa_backend.py`
- Create: `tests/evaluation/librosa_bench/test_librosa_backend.py`

No `tests/evaluation/librosa_bench/__init__.py` is needed — `tests/evaluation/` has no
`__init__.py` today and pytest discovers it via `testpaths`/rootdir, not package imports.

**Interfaces produced:**

```python
# _librosa_backend.py
class LibrosaUnavailableError(ImportError): ...


@dataclass(frozen=True, slots=True)
class OnsetEnvelope:
    values: tuple[float, ...]
    hop_length: int
    frame_rate_hz: float


def estimate_tempo_and_beats(
    samples: npt.NDArray[np.float64], sample_rate_hz: int
) -> tuple[float, tuple[float, ...]]: ...


def onset_envelope(samples: npt.NDArray[np.float64], sample_rate_hz: int) -> OnsetEnvelope: ...
```

`sample_rate_hz` is a required positional/keyword argument with no default on both functions —
this is what makes silently falling back to librosa's own default sample rate a type error rather
than a runtime footgun (design §9). `import librosa` happens lazily, once, inside a private
`_import_librosa()` helper called by both public functions — never at module import time.

**Interfaces consumed:** none from the rest of the repo. `numpy`/`numpy.typing` only (already a
core dependency).

**Design constraints enforced here:** decision 7's exact empirical typing procedure (below); this
is the *only* file in the whole feature allowed to `import librosa`.

### RED (behavior: missing dependency raises the documented error)

`tests/evaluation/librosa_bench/test_librosa_backend.py`:

```python
from __future__ import annotations

import sys

import numpy as np
import pytest


def test_estimate_tempo_and_beats_raises_when_librosa_is_unavailable(monkeypatch) -> None:
    from lights_audio_engine.evaluation.librosa_bench._librosa_backend import (
        LibrosaUnavailableError,
        estimate_tempo_and_beats,
    )

    monkeypatch.setitem(sys.modules, "librosa", None)
    with pytest.raises(LibrosaUnavailableError, match=r"pip install -e \"\.\[librosa\]\""):
        estimate_tempo_and_beats(np.zeros(48_000, dtype=np.float64), 48_000)


def test_onset_envelope_raises_when_librosa_is_unavailable(monkeypatch) -> None:
    from lights_audio_engine.evaluation.librosa_bench._librosa_backend import (
        LibrosaUnavailableError,
        onset_envelope,
    )

    monkeypatch.setitem(sys.modules, "librosa", None)
    with pytest.raises(LibrosaUnavailableError):
        onset_envelope(np.zeros(48_000, dtype=np.float64), 48_000)


def test_estimate_tempo_and_beats_passes_exact_sample_rate_no_default() -> None:
    librosa = pytest.importorskip("librosa")
    from lights_audio_engine.evaluation.librosa_bench._librosa_backend import (
        estimate_tempo_and_beats,
    )

    calls: list[int] = []
    original = librosa.beat.beat_track

    def _recording_beat_track(*, y, sr, **kwargs):
        calls.append(sr)
        return original(y=y, sr=sr, **kwargs)

    import unittest.mock as mock

    samples = np.zeros(48_000, dtype=np.float64)
    with mock.patch.object(librosa.beat, "beat_track", _recording_beat_track):
        estimate_tempo_and_beats(samples, 44_100)

    assert calls == [44_100]


def test_real_librosa_returns_finite_tempo_and_ordered_beats() -> None:
    pytest.importorskip("librosa")
    from lights_audio_engine.evaluation.librosa_bench._librosa_backend import (
        estimate_tempo_and_beats,
        onset_envelope,
    )

    sample_rate_hz = 22_050
    duration_seconds = 4.0
    t = np.arange(0, duration_seconds, 1 / sample_rate_hz)
    samples = (0.5 * np.sin(2 * np.pi * 2.0 * t)).astype(np.float64)  # 120 BPM-ish pulse-like tone

    tempo_bpm, beat_times = estimate_tempo_and_beats(samples, sample_rate_hz)
    envelope = onset_envelope(samples, sample_rate_hz)

    assert tempo_bpm > 0.0
    assert list(beat_times) == sorted(beat_times)
    assert all(0.0 <= b <= duration_seconds for b in beat_times)
    assert envelope.hop_length > 0
    assert envelope.frame_rate_hz == pytest.approx(sample_rate_hz / envelope.hop_length)
    assert len(envelope.values) > 0
```

Command:

```
python -m pytest tests/evaluation/librosa_bench/test_librosa_backend.py -q
```

Expected failure: `ModuleNotFoundError: No module named 'lights_audio_engine.evaluation.librosa_bench'`
— the package doesn't exist yet. This is the expected reason.

### GREEN

1. `src/lights_audio_engine/evaluation/librosa_bench/__init__.py`:
   ```python
   """Offline, advisory Librosa benchmark for M2C artifacts; requires the optional `librosa` extra."""
   ```
2. `src/lights_audio_engine/evaluation/librosa_bench/_librosa_backend.py` implementing the
   interface above. `estimate_tempo_and_beats` calls
   `librosa.beat.beat_track(y=samples, sr=sample_rate_hz, units="time")` and defensively coerces
   the tempo return value with `float(np.asarray(tempo).reshape(-1)[0])` to absorb the
   scalar-vs-array shape difference across librosa minor versions (design §12 risk 2).
   `onset_envelope` calls `librosa.onset.onset_strength(y=samples, sr=sample_rate_hz, hop_length=512)`.

Command:

```
python -m pytest tests/evaluation/librosa_bench/test_librosa_backend.py -q
```

Without `librosa` installed: the two `LibrosaUnavailableError` tests pass; the two
`importorskip`-guarded tests report **skipped**. Expected: `2 passed, 2 skipped`.

With `librosa` installed (`pip install -e ".[librosa]"`): expected `4 passed`.

### Typing check (decision 7 — executed here, not pre-decided)

Run in this exact order, on this file alone first (fast signal before running the whole tree):

```
pip install -e ".[librosa]"
python -m basedpyright src/lights_audio_engine/evaluation/librosa_bench/_librosa_backend.py
```

Record whatever diagnostics appear on the `import librosa` line or on calls into `librosa`'s API.
This checks the *usage* is well-typed once the package is present.

```
pip uninstall librosa -y
python -m basedpyright src/lights_audio_engine/evaluation/librosa_bench/_librosa_backend.py
```

This second run is the **binding** one — it mirrors mandatory CI's `.[dev]`-only environment
exactly. Record the diagnostic code (e.g. `reportMissingImports`, `reportMissingModuleSource`) it
produces, if any.

- If the second run produces no error, add **no** suppression anywhere.
- If it does, add the single narrowest suppression that resolves that exact diagnostic code,
  scoped to the `import librosa` line only:
  `import librosa  # pyright: ignore[<exact code observed>]`.
  Do not use a bare `# pyright: ignore`, a file-level directive, or a `pyproject.toml` exclusion.
- Re-run the first (`librosa`-installed) check once more after adding the suppression, to confirm
  it doesn't also silence a genuine API-misuse error in the same file:
  ```
  pip install -e ".[librosa]"
  python -m basedpyright src/lights_audio_engine/evaluation/librosa_bench/_librosa_backend.py
  ```

Whatever the outcome, run the full-tree check afterward to confirm nothing elsewhere regressed:

```
python -m basedpyright
```

(with `.[dev]` only — uninstall `librosa` again first if it's still present, so this matches
mandatory CI exactly).

### Commit

```
git add src/lights_audio_engine/evaluation/librosa_bench/__init__.py
git add src/lights_audio_engine/evaluation/librosa_bench/_librosa_backend.py
git add tests/evaluation/librosa_bench/test_librosa_backend.py
git commit -m "feat: add isolated librosa backend wrapper for offline benchmark"
```

(If Task 2's typing check required a suppression, that line is part of this same commit — it's
the file it belongs to.)

---

## Task 3 — Offline analysis over one M2C segment

**Files:**
- Create: `src/lights_audio_engine/evaluation/librosa_bench/analysis.py`
- Create: `tests/evaluation/librosa_bench/test_analysis.py`

**Interfaces produced:**

```python
class LibrosaAnalysisError(ValueError): ...


@dataclass(frozen=True, slots=True)
class OnsetSummary:
    hop_length: int
    frame_rate_hz: float
    mean_strength: float
    max_strength: float
    frame_count: int


@dataclass(frozen=True, slots=True)
class LibrosaAnalysis:
    label: str
    segment_index: int
    sample_rate_hz: int
    tempo_bpm: float
    beat_times_seconds: tuple[float, ...]
    onset_summary: OnsetSummary
    onset_envelope: OnsetEnvelope  # full-resolution; never serialized directly into the JSON report


def analyze_segment(
    artifact: PcmArtifact, *, segment_index: int | None = None
) -> LibrosaAnalysis: ...


def write_onset_envelope(path: Path, envelope: OnsetEnvelope) -> None: ...
def read_onset_envelope(path: Path) -> OnsetEnvelope: ...
```

**Interfaces consumed:** `lights_audio_engine.evaluation.artifact.PcmArtifact` (unmodified, read
only — `.samples`, `.sample_rate_hz`, `.segments`, `.label`); `_librosa_backend`'s
`estimate_tempo_and_beats`, `onset_envelope`, `OnsetEnvelope` (Task 2, unmodified).

`write_onset_envelope`/`read_onset_envelope` use a plain `.npy` array plus an adjacent
`.with_suffix(".json")` manifest tagged `"kind": "librosa_onset_envelope"` — deliberately **not**
`artifact.write_artifact`/`read_artifact`, since the envelope is a derived, non-authoritative
signal, not a re-encoding of authoritative PCM (design §3). `write_onset_envelope` requires a
`.npy`-suffixed path, mirroring `artifact.py`'s own convention, and raises `LibrosaAnalysisError`
otherwise.

**Design constraints enforced here:** no resampling (segment selection passes the artifact's exact
`sample_rate_hz` through, never a default); segment-index error messages match
`ReplayAudioSource`'s wording for a consistent operator experience; finite/non-negative validation
mirrors `models.py`/`candidates.py`'s defensive style.

### RED

`tests/evaluation/librosa_bench/test_analysis.py` (excerpt — full file covers all cases below):

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_artifact(tmp_path: Path, *, sample_rate_hz: int = 48_000, sample_count: int = 48_000):
    from lights_audio_engine.evaluation.artifact import write_artifact

    path = tmp_path / "track.npy"
    write_artifact(
        path,
        np.zeros(sample_count, dtype=np.float64),
        label="track",
        sample_rate_hz=sample_rate_hz,
    )
    return path


def test_analyze_segment_passes_exact_sample_rate_with_no_default(
    tmp_path: Path, monkeypatch
) -> None:
    from lights_audio_engine.evaluation.artifact import read_artifact
    from lights_audio_engine.evaluation.librosa_bench import analysis
    from lights_audio_engine.evaluation.librosa_bench._librosa_backend import OnsetEnvelope

    path = _write_artifact(tmp_path, sample_rate_hz=44_100)
    artifact = read_artifact(path)

    recorded: dict[str, int] = {}

    def _fake_estimate(samples, sample_rate_hz):
        recorded["tempo_sr"] = sample_rate_hz
        return 120.0, (0.5, 1.0)

    def _fake_envelope(samples, sample_rate_hz):
        recorded["envelope_sr"] = sample_rate_hz
        return OnsetEnvelope((0.1, 0.2), hop_length=512, frame_rate_hz=sample_rate_hz / 512)

    monkeypatch.setattr(analysis, "estimate_tempo_and_beats", _fake_estimate)
    monkeypatch.setattr(analysis, "onset_envelope", _fake_envelope)

    result = analysis.analyze_segment(artifact)

    assert recorded["tempo_sr"] == 44_100
    assert recorded["envelope_sr"] == 44_100
    assert result.sample_rate_hz == 44_100
    assert result.tempo_bpm == 120.0
    assert result.beat_times_seconds == (0.5, 1.0)


def test_analyze_segment_requires_explicit_index_for_discontinuous_artifact(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.artifact import SegmentInfo, read_artifact, write_artifact
    from lights_audio_engine.evaluation.librosa_bench.analysis import analyze_segment

    path = tmp_path / "split.npy"
    write_artifact(
        path,
        np.zeros(20, dtype=np.float64),
        label="split",
        sample_rate_hz=1_000,
        segments=(SegmentInfo(0, 10, 0.0, None), SegmentInfo(10, 10, 4.0, "overflow")),
    )
    artifact = read_artifact(path)

    with pytest.raises(ValueError, match="explicit segment index"):
        analyze_segment(artifact)


def test_analyze_segment_rejects_out_of_range_index(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.artifact import read_artifact
    from lights_audio_engine.evaluation.librosa_bench.analysis import analyze_segment

    artifact = read_artifact(_write_artifact(tmp_path))

    with pytest.raises(ValueError, match="segment index is out of range"):
        analyze_segment(artifact, segment_index=5)


def test_analyze_segment_raises_on_non_finite_backend_result(tmp_path: Path, monkeypatch) -> None:
    from lights_audio_engine.evaluation.artifact import read_artifact
    from lights_audio_engine.evaluation.librosa_bench import analysis
    from lights_audio_engine.evaluation.librosa_bench._librosa_backend import OnsetEnvelope

    artifact = read_artifact(_write_artifact(tmp_path))
    monkeypatch.setattr(analysis, "estimate_tempo_and_beats", lambda *a: (float("nan"), ()))
    monkeypatch.setattr(
        analysis, "onset_envelope", lambda *a: OnsetEnvelope((), hop_length=512, frame_rate_hz=1.0)
    )

    with pytest.raises(analysis.LibrosaAnalysisError, match="finite"):
        analysis.analyze_segment(artifact)


def test_onset_envelope_roundtrips_through_sidecar_files(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.librosa_bench._librosa_backend import OnsetEnvelope
    from lights_audio_engine.evaluation.librosa_bench.analysis import (
        read_onset_envelope,
        write_onset_envelope,
    )

    envelope = OnsetEnvelope((0.1, 0.4, 0.2), hop_length=512, frame_rate_hz=86.1328125)
    path = tmp_path / "track.envelope.npy"
    write_onset_envelope(path, envelope)

    loaded = read_onset_envelope(path)
    assert loaded.values == pytest.approx(envelope.values)
    assert loaded.hop_length == envelope.hop_length
    assert loaded.frame_rate_hz == pytest.approx(envelope.frame_rate_hz)


def test_analyze_segment_end_to_end_with_real_librosa(tmp_path: Path) -> None:
    pytest.importorskip("librosa")
    from lights_audio_engine.evaluation.artifact import read_artifact
    from lights_audio_engine.evaluation.librosa_bench.analysis import analyze_segment

    artifact = read_artifact(_write_artifact(tmp_path, sample_count=48_000 * 2))
    result = analyze_segment(artifact)

    assert result.tempo_bpm >= 0.0
    assert result.onset_summary.frame_count == len(result.onset_envelope.values)
```

Command:

```
python -m pytest tests/evaluation/librosa_bench/test_analysis.py -q
```

Expected failure: `ModuleNotFoundError: No module named 'lights_audio_engine.evaluation.librosa_bench.analysis'`.

### GREEN

Implement `analysis.py` per the interface above, including the local `_select_segment_samples()`
helper (not `ReplayAudioSource` — see design §3 rationale) with the two error messages copied
verbatim from `ReplayAudioSource.__init__`.

Command:

```
python -m pytest tests/evaluation/librosa_bench/test_analysis.py -q
```

Expected without `librosa` installed: 5 passed, 1 skipped (the real-librosa end-to-end test).
Expected with `librosa` installed: 6 passed.

### Commit

```
git add src/lights_audio_engine/evaluation/librosa_bench/analysis.py
git add tests/evaluation/librosa_bench/test_analysis.py
git commit -m "feat: add librosa offline analysis for M2C artifact segments"
```

---

## Task 4 — Export adapters (Audacity + M2C-shaped candidate file)

**Files:**
- Create: `src/lights_audio_engine/evaluation/librosa_bench/export.py`
- Create: `tests/evaluation/librosa_bench/test_export.py`

**Interfaces produced:**

```python
def write_audacity_labels(
    path: Path, beat_times_seconds: tuple[float, ...], *, label_prefix: str = "librosa-beat"
) -> None: ...


def write_candidate_reference(path: Path, beat_times_seconds: tuple[float, ...]) -> None: ...


def convert_audacity_export_to_reference(audacity_path: Path, output_path: Path) -> None: ...
```

**Interfaces consumed:** `lights_audio_engine.evaluation.reference.parse_reference` and
`ReferenceFormatError` (unmodified) — used only to (a) round-trip-validate
`convert_audacity_export_to_reference`'s own output before returning, and (b) let
`test_export.py` prove `write_candidate_reference`'s output is byte-compatible with the real
parser, not a lookalike.

**Design constraints enforced here:** decision 5 — none of these functions, and nothing that
calls them, defines the canonical model; they only ever take a plain `tuple[float, ...]` in. No
`librosa` import anywhere in this file — all tests run with zero optional dependencies (design
§10).

### RED

`tests/evaluation/librosa_bench/test_export.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest


def test_write_audacity_labels_uses_point_label_convention(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.librosa_bench.export import write_audacity_labels

    path = tmp_path / "beats.librosa-beats.txt"
    write_audacity_labels(path, (0.5, 1.0, 1.5))

    rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        ["0.500000", "0.500000", "librosa-beat-1"],
        ["1.000000", "1.000000", "librosa-beat-2"],
        ["1.500000", "1.500000", "librosa-beat-3"],
    ]


def test_write_candidate_reference_is_parseable_by_the_real_m2c_parser(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.librosa_bench.export import write_candidate_reference
    from lights_audio_engine.evaluation.reference import parse_reference

    path = tmp_path / "beats.librosa-candidate.txt"
    write_candidate_reference(path, (0.5, 1.0, 1.5))

    assert parse_reference(path) == (0.5, 1.0, 1.5)


def test_convert_audacity_export_round_trips_to_valid_reference(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.librosa_bench.export import (
        convert_audacity_export_to_reference,
    )
    from lights_audio_engine.evaluation.reference import parse_reference

    audacity_path = tmp_path / "reviewed.txt"
    audacity_path.write_text(
        "0.500000\t0.500000\tbeat\n1.000000\t1.000000\tbeat\n", encoding="utf-8"
    )
    output_path = tmp_path / "promoted-reference.txt"

    convert_audacity_export_to_reference(audacity_path, output_path)

    assert parse_reference(output_path) == (0.5, 1.0)


def test_convert_audacity_export_rejects_range_labels(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.librosa_bench.export import (
        convert_audacity_export_to_reference,
    )
    from lights_audio_engine.evaluation.reference import ReferenceFormatError

    audacity_path = tmp_path / "reviewed.txt"
    audacity_path.write_text("0.500000\t0.900000\tchorus\n", encoding="utf-8")

    with pytest.raises(ReferenceFormatError, match="range label"):
        convert_audacity_export_to_reference(audacity_path, tmp_path / "out.txt")
```

Command:

```
python -m pytest tests/evaluation/librosa_bench/test_export.py -q
```

Expected failure: `ModuleNotFoundError: No module named 'lights_audio_engine.evaluation.librosa_bench.export'`.

### GREEN

Implement `export.py` per the interface above.

Command:

```
python -m pytest tests/evaluation/librosa_bench/test_export.py -q
```

Expected: 4 passed. (No `librosa` needed — must pass identically with or without the extra
installed; run once in each state to confirm.)

### Commit

```
git add src/lights_audio_engine/evaluation/librosa_bench/export.py
git add tests/evaluation/librosa_bench/test_export.py
git commit -m "feat: add librosa export adapters for Audacity and M2C candidate review"
```

---

## Task 5 — Advisory report + human-reference scoring (latency-safe)

**Files:**
- Create: `src/lights_audio_engine/evaluation/librosa_bench/report.py`
- Create: `tests/evaluation/librosa_bench/test_report.py`

**Interfaces produced:**

```python
DECISION_LATENCY_NOT_APPLICABLE: str  # literal explanatory string, design §7


@dataclass(frozen=True, slots=True)
class LibrosaScore:
    reference_path: str
    tolerance_seconds: float
    true_positives: int
    false_positives: int
    false_negatives: int
    short_doubles: int
    precision: float
    recall: float
    f1: float
    median_absolute_timing_error_seconds: float | None
    p95_absolute_timing_error_seconds: float | None
    decision_latency: str = DECISION_LATENCY_NOT_APPLICABLE
    # deliberately no field of numeric type with "latency" in its name


def score_against_human_reference(
    reference_path: Path,
    beat_times_seconds: tuple[float, ...],
    *,
    tolerance_seconds: float = 0.05,
) -> LibrosaScore: ...


@dataclass(frozen=True, slots=True)
class LibrosaBenchmarkReport:
    kind: str  # always "librosa_offline_benchmark"
    advisory_only: bool  # always True
    production_candidate: bool  # always False
    label: str
    segment_index: int
    sample_rate_hz: int
    tempo_bpm: float
    beat_times_seconds: tuple[float, ...]
    beat_times_source: str  # always "librosa_offline_benchmark"
    onset_summary: dict[str, float | int]
    onset_envelope_path: str | None
    human_comparison: LibrosaScore | None


def build_report(
    analysis: LibrosaAnalysis,
    human_comparison: LibrosaScore | None,
    *,
    onset_envelope_path: Path | None = None,
) -> LibrosaBenchmarkReport: ...


def write_report(path: Path, report: LibrosaBenchmarkReport) -> None: ...
```

**Interfaces consumed:** `lights_audio_engine.evaluation.reference.parse_reference`,
`lights_audio_engine.evaluation.matching.match_events`,
`lights_audio_engine.evaluation.scoring.score_events` — all unmodified. `LibrosaAnalysis` (Task 3).

**Design constraint enforced here (decision 4, the one requiring the most scrutiny):**
`score_against_human_reference()` must call `score_events()` (required by reuse — its signature
cannot change) with `emission_times_seconds` set to the same values as `beat_times_seconds`, purely
to satisfy a required argument that has no offline equivalent. The resulting `TrackMetrics`
object's latency fields are `0.0` by construction of that placeholder and are **never** read,
returned, or copied onto `LibrosaScore` — `LibrosaScore` has no numeric latency field at all, only
the literal `decision_latency: str`. This is verified by an introspection test below, not just by
inspection.

### RED

`tests/evaluation/librosa_bench/test_report.py`:

```python
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest


def _analysis(label="track", segment_index=0, sample_rate_hz=48_000):
    from lights_audio_engine.evaluation.librosa_bench._librosa_backend import OnsetEnvelope
    from lights_audio_engine.evaluation.librosa_bench.analysis import LibrosaAnalysis, OnsetSummary

    envelope = OnsetEnvelope((0.1, 0.2, 0.05), hop_length=512, frame_rate_hz=93.75)
    return LibrosaAnalysis(
        label=label,
        segment_index=segment_index,
        sample_rate_hz=sample_rate_hz,
        tempo_bpm=120.0,
        beat_times_seconds=(0.5, 1.0, 1.5),
        onset_summary=OnsetSummary(
            hop_length=512,
            frame_rate_hz=93.75,
            mean_strength=0.1167,
            max_strength=0.2,
            frame_count=3,
        ),
        onset_envelope=envelope,
    )


def test_score_against_human_reference_reuses_real_matching_and_scoring(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.librosa_bench.report import score_against_human_reference

    reference_path = tmp_path / "human.txt"
    reference_path.write_text("0.51\tbeat\n1.00\tbeat\n2.00\tbeat\n", encoding="utf-8")

    score = score_against_human_reference(reference_path, (0.5, 1.0, 1.5), tolerance_seconds=0.05)

    assert score.true_positives == 2
    assert score.false_positives == 1
    assert score.false_negatives == 1
    assert score.precision == pytest.approx(2 / 3)
    assert score.recall == pytest.approx(2 / 3)


def test_librosa_score_has_no_numeric_latency_field() -> None:
    from lights_audio_engine.evaluation.librosa_bench.report import LibrosaScore

    field_names = {f.name: f.type for f in dataclasses.fields(LibrosaScore)}
    latency_fields = {name: type_ for name, type_ in field_names.items() if "latency" in name}

    assert latency_fields == {"decision_latency": "str"}


def test_librosa_score_decision_latency_is_the_documented_literal(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.librosa_bench.report import (
        DECISION_LATENCY_NOT_APPLICABLE,
        score_against_human_reference,
    )

    reference_path = tmp_path / "human.txt"
    reference_path.write_text("0.5\tbeat\n", encoding="utf-8")

    score = score_against_human_reference(reference_path, (0.5,))

    assert score.decision_latency == DECISION_LATENCY_NOT_APPLICABLE
    assert isinstance(score.decision_latency, str)


def test_build_report_is_self_labeled_advisory_and_not_a_production_candidate() -> None:
    from lights_audio_engine.evaluation.librosa_bench.report import build_report

    report = build_report(_analysis(), human_comparison=None)

    assert report.kind == "librosa_offline_benchmark"
    assert report.advisory_only is True
    assert report.production_candidate is False
    assert report.beat_times_source == "librosa_offline_benchmark"
    assert report.human_comparison is None


def test_write_report_serializes_schema_version_and_never_a_numeric_latency(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.librosa_bench.report import (
        build_report,
        score_against_human_reference,
        write_report,
    )

    reference_path = tmp_path / "human.txt"
    reference_path.write_text("0.5\tbeat\n1.0\tbeat\n1.5\tbeat\n", encoding="utf-8")
    analysis = _analysis()
    score = score_against_human_reference(reference_path, analysis.beat_times_seconds)
    report = build_report(analysis, score)
    output = tmp_path / "report.json"

    write_report(output, report)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["advisory_only"] is True
    assert payload["production_candidate"] is False
    assert payload["human_comparison"]["decision_latency"] == (
        "not applicable — Librosa runs fully offline over the whole segment; "
        "no causal emission time exists"
    )
    assert "decision_latencies_seconds" not in payload["human_comparison"]
    assert "median_decision_latency_seconds" not in payload["human_comparison"]
    assert "p95_decision_latency_seconds" not in payload["human_comparison"]


def test_librosa_benchmark_report_schema_has_no_bakeoff_report_fields() -> None:
    from lights_audio_engine.evaluation.bakeoff import BakeoffReport
    from lights_audio_engine.evaluation.librosa_bench.report import LibrosaBenchmarkReport

    bakeoff_fields = {f.name for f in dataclasses.fields(BakeoffReport)}
    librosa_fields = {f.name for f in dataclasses.fields(LibrosaBenchmarkReport)}

    assert not bakeoff_fields & librosa_fields  # zero shared field names, distinct schemas
```

Command:

```
python -m pytest tests/evaluation/librosa_bench/test_report.py -q
```

Expected failure: `ModuleNotFoundError: No module named 'lights_audio_engine.evaluation.librosa_bench.report'`.

### GREEN

Implement `report.py` per the interface above.

Command:

```
python -m pytest tests/evaluation/librosa_bench/test_report.py -q
```

Expected: 6 passed. No `librosa` needed for any of these (they operate on hand-built
`LibrosaAnalysis` values and the real, unmodified `match_events`/`score_events`).

### Commit

```
git add src/lights_audio_engine/evaluation/librosa_bench/report.py
git add tests/evaluation/librosa_bench/test_report.py
git commit -m "feat: add advisory librosa benchmark report and human-reference scoring"
```

---

## Task 6 — CLI entry point

**Files:**
- Create: `src/lights_audio_engine/evaluation/librosa_bench/cli.py`
- Create: `src/lights_audio_engine/evaluation/librosa_bench/__main__.py`
- Create: `tests/evaluation/librosa_bench/test_cli.py`

**Interfaces produced:**

```python
def main(argv: list[str] | None = None) -> int: ...
```

CLI surface (design §5):

```
python -m lights_audio_engine.evaluation.librosa_bench <artifact.npy> --output <report.json>
    [--segment-index N]
    [--audacity-labels <path.txt>]
    [--candidate-reference <path.txt>]
    [--envelope-npy <path.npy>]
    [--human-reference <path.txt>] [--tolerance-ms 50.0]
```

Exit codes: `0` success; `2` on `ArtifactError`, `ReferenceFormatError`, `LibrosaUnavailableError`,
`LibrosaAnalysisError`, `ValueError`, `OSError`, printed to stderr prefixed
`"Librosa benchmark error: ..."`.

**Interfaces consumed:** `read_artifact`/`ArtifactError` (Task-independent, unmodified),
`analyze_segment`/`write_onset_envelope`/`LibrosaAnalysisError` (Task 3),
`write_audacity_labels`/`write_candidate_reference` (Task 4),
`build_report`/`score_against_human_reference`/`write_report` (Task 5),
`LibrosaUnavailableError` (Task 2), `ReferenceFormatError` (unmodified).

### RED

`tests/evaluation/librosa_bench/test_cli.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _write_pulse_artifact(tmp_path: Path, beat_times=(0.5, 1.0, 1.5)) -> Path:
    from lights_audio_engine.evaluation.artifact import write_artifact

    sample_count = round((max(beat_times) + 0.5) * 48_000)
    samples = np.zeros(sample_count, dtype=np.float64)
    for t in beat_times:
        start = round(t * 48_000)
        samples[start : start + 960] = 0.8
    path = tmp_path / "track.npy"
    write_artifact(path, samples, label="track", sample_rate_hz=48_000)
    return path


def test_cli_missing_librosa_exits_with_documented_message(tmp_path: Path, monkeypatch) -> None:
    import sys

    from lights_audio_engine.evaluation.librosa_bench.cli import main

    monkeypatch.setitem(sys.modules, "librosa", None)
    artifact = _write_pulse_artifact(tmp_path)
    output = tmp_path / "report.json"

    exit_code = main([str(artifact), "--output", str(output)])

    assert exit_code == 2
    assert not output.exists()


def test_cli_end_to_end_with_real_librosa_produces_advisory_report(tmp_path: Path) -> None:
    pytest.importorskip("librosa")
    from lights_audio_engine.evaluation.librosa_bench.cli import main

    artifact = _write_pulse_artifact(tmp_path)
    reference = tmp_path / "human.txt"
    reference.write_text("0.500\tbeat\n1.000\tbeat\n1.500\tbeat\n", encoding="utf-8")
    output = tmp_path / "report.json"
    audacity_path = tmp_path / "track.librosa-beats.txt"
    candidate_path = tmp_path / "track.librosa-candidate.txt"
    envelope_path = tmp_path / "track.envelope.npy"

    exit_code = main(
        [
            str(artifact),
            "--output",
            str(output),
            "--audacity-labels",
            str(audacity_path),
            "--candidate-reference",
            str(candidate_path),
            "--envelope-npy",
            str(envelope_path),
            "--human-reference",
            str(reference),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["advisory_only"] is True
    assert payload["production_candidate"] is False
    assert audacity_path.exists()
    assert candidate_path.exists()
    assert envelope_path.exists()
    assert payload["human_comparison"] is not None
    assert payload["onset_envelope_path"] == str(envelope_path)


def test_cli_reports_bad_artifact_with_exit_code_two(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.librosa_bench.cli import main

    missing = tmp_path / "does-not-exist.npy"
    exit_code = main([str(missing), "--output", str(tmp_path / "report.json")])

    assert exit_code == 2
```

Command:

```
python -m pytest tests/evaluation/librosa_bench/test_cli.py -q
```

Expected failure: `ModuleNotFoundError: No module named 'lights_audio_engine.evaluation.librosa_bench.cli'`.

### GREEN

Implement `cli.py` (argparse wiring per §5) and `__main__.py`:

```python
"""Module entry point for the offline, advisory librosa benchmark."""

from lights_audio_engine.evaluation.librosa_bench.cli import main

raise SystemExit(main())
```

Command:

```
python -m pytest tests/evaluation/librosa_bench/test_cli.py -q
```

Expected without `librosa`: 2 passed, 1 skipped. Expected with `librosa` installed: 3 passed.

### Commit

```
git add src/lights_audio_engine/evaluation/librosa_bench/cli.py
git add src/lights_audio_engine/evaluation/librosa_bench/__main__.py
git add tests/evaluation/librosa_bench/test_cli.py
git commit -m "feat: add librosa offline benchmark cli entry point"
```

---

## Task 7 — Architectural boundary regression tests

**Files:**
- Modify: `tests/test_librosa_bench_import_boundary.py` (created in Task 1; add the isolation test
  now that there's a package to isolate)

**Interfaces produced/consumed:** none new — this is pure regression coverage over the boundary
described in design §2, using the same subprocess-isolated-import technique already established
by `tests/test_capture_import_is_numpy_only.py`.

### RED

Append to `tests/test_librosa_bench_import_boundary.py`:

```python
import os
import subprocess
import sys


def _run_isolated_import(code: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    return subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True, env=environment
    )


def test_bakeoff_import_does_not_pull_in_librosa_bench() -> None:
    result = _run_isolated_import(
        "import sys\n"
        "import lights_audio_engine.evaluation.bakeoff\n"
        "assert 'lights_audio_engine.evaluation.librosa_bench' not in sys.modules\n"
    )
    assert result.returncode == 0, result.stderr


def test_candidates_import_does_not_pull_in_librosa_bench() -> None:
    result = _run_isolated_import(
        "import sys\n"
        "import lights_audio_engine.evaluation.candidates\n"
        "assert 'lights_audio_engine.evaluation.librosa_bench' not in sys.modules\n"
    )
    assert result.returncode == 0, result.stderr


def test_evaluation_cli_import_does_not_pull_in_librosa_bench() -> None:
    result = _run_isolated_import(
        "import sys\n"
        "import lights_audio_engine.evaluation.cli\n"
        "assert 'lights_audio_engine.evaluation.librosa_bench' not in sys.modules\n"
    )
    assert result.returncode == 0, result.stderr


def test_librosa_bench_import_does_not_require_librosa_to_be_installed() -> None:
    result = _run_isolated_import(
        "import sys\n"
        "sys.modules['librosa'] = None\n"
        "import lights_audio_engine.evaluation.librosa_bench.cli\n"
    )
    assert result.returncode == 0, result.stderr
```

Command:

```
python -m pytest tests/test_librosa_bench_import_boundary.py -q
```

These assertions are true as soon as Task 6 lands (the modules genuinely don't reference each
other), so there is no traditional RED state at this point in the sequence — the meaningful "RED"
already happened implicitly: had any earlier task imported `librosa_bench` from `bakeoff.py`,
`candidates.py`, or `evaluation/cli.py`, this test would catch it then. Run it now to establish it
as a permanent regression guard, and confirm it passes:

```
python -m pytest tests/test_librosa_bench_import_boundary.py -q
```

Expected: 5 passed (1 from Task 1 + 4 new).

### Commit

```
git add tests/test_librosa_bench_import_boundary.py
git commit -m "test: guard librosa_bench import isolation from m2c production modules"
```

---

## Task 8 (optional, not required for acceptance) — Documentation

**Files:**
- Create: `docs/m2c-librosa-offline-benchmark.md`
- Modify: `docs/m2c-detector-bakeoff.md` (one-line cross-reference in "Manual evidence remaining")

Not gated by tests; not part of the acceptance criteria in the design spec §11. Do only after
Tasks 1–7 are merged-ready. Skip if the team wants to ship the code first and document later.

---

## Final validation (run once, after Task 7, before opening a PR)

Run in this exact order. Each command's expected result is stated so a failure is unambiguous.

### 1. Librosa-specific focused tests, extra installed

```
pip install -e ".[librosa]"
python -m pytest tests/evaluation/librosa_bench -q
```
Expected: all tests pass, none skipped (every `importorskip` guard now has its dependency).

### 2. Full pytest suite

```
python -m pytest
```
Expected: all tests pass, including every pre-existing M2C test
(`tests/evaluation/test_bakeoff.py`, `test_candidates.py`, `test_cli_report.py`,
`test_reference_matching_scoring.py`, `test_artifact_replay.py`) with **identical** results to
Task 0's baseline run — this is the direct check that existing M2C recommendation behavior is
unchanged.

### 3. Ruff

```
python -m ruff format --check .
python -m ruff check .
```
Expected: no diffs, no lint findings, in new or existing files.

### 4. BasedPyright, librosa installed

```
python -m basedpyright
```
(with `.[librosa]` still installed from step 1) — catches any real type misuse against librosa's
actual API surface.
Expected: 0 errors.

### 5. BasedPyright, mandatory-CI configuration

```
pip uninstall librosa -y
python -m basedpyright
```
Expected: 0 errors, using only whatever suppression (if any) was determined empirically in Task 2
— this is the exact mandatory-CI gate from `.github/workflows/ci.yml`, reproduced locally.

### 6. `git diff --check`

```
git diff --check main
```
Expected: no whitespace-error output. Run against the merge-base with `main`, not just the working
tree, to catch the whole branch.

### 7. Real-artifact smoke command (no private audio committed)

There are two tiers here, and only the first is scriptable/repeatable:

**a. Synthetic smoke test (already covered by `test_cli.py`'s end-to-end case in Task 6, re-run
standalone for visibility):**
```
python -m pytest tests/evaluation/librosa_bench/test_cli.py::test_cli_end_to_end_with_real_librosa_produces_advisory_report -q
```
Expected: 1 passed. This is the CI-safe substitute — it never touches real captured audio.

**b. Optional manual smoke test against a real M2C artifact, if one exists locally** (this cannot
be scripted into CI or committed, since real captured audio is `.gitignore`d local evidence per
`docs/m2c-detector-bakeoff.md`; run manually, output stays local):
```
python -m lights_audio_engine.evaluation.librosa_bench `
  (Join-Path $env:USERPROFILE 'Documents\lights-audio-engine-m2c\holdout-track-01.npy') `
  --output (Join-Path $env:USERPROFILE 'Documents\lights-audio-engine-m2c\holdout-track-01.librosa-report.json') `
  --audacity-labels (Join-Path $env:USERPROFILE 'Documents\lights-audio-engine-m2c\holdout-track-01.librosa-beats.txt')
```
Expected: exit code 0, a report JSON with `tempo_bpm > 0` and a non-empty `beat_times_seconds`,
and the Audacity file opens correctly in Audacity. If no real artifact exists locally at plan time,
this step is deferred to whoever runs the implementation with access to `m2c-artifacts/` — it is
not a blocker for the tiered acceptance criteria above, which are fully covered by 7a.

---

## Explicit constraint verification checklist

Each of the user's required verifications, with the exact evidence that proves it:

| Requirement | Evidence |
| --- | --- |
| Existing M2C recommendation behavior is unchanged | Final validation step 2: full suite green, identical to Task 0 baseline; `git diff --stat main -- src/lights_audio_engine/evaluation/bakeoff.py src/lights_audio_engine/evaluation/candidates.py src/lights_audio_engine/evaluation/cli.py src/lights_audio_engine/evaluation/report.py src/lights_audio_engine/evaluation/replay_source.py` returns empty. |
| Librosa cannot appear as Candidate D/L | `tests/test_librosa_bench_import_boundary.py::test_candidates_import_does_not_pull_in_librosa_bench` (Task 7); `candidates.py`'s zero-diff confirmed above; `create_candidate()` still only accepts `"baseline" \| "broadband" \| "multiband"`. |
| Librosa cannot contribute to M2C production gates | `tests/test_librosa_bench_import_boundary.py::test_bakeoff_import_does_not_pull_in_librosa_bench`; `bakeoff.py`'s zero-diff confirmed above; `_quality_gates()` and `run_bakeoff()` source unchanged. |
| Report clearly identifies itself as advisory/offline | `test_report.py::test_build_report_is_self_labeled_advisory_and_not_a_production_candidate` and `test_write_report_serializes_schema_version_and_never_a_numeric_latency` (Task 5) — both assert `advisory_only: true`, `production_candidate: false`, `kind: "librosa_offline_benchmark"` literally, in the serialized JSON. |
| No numeric latency claim is exposed | `test_report.py::test_librosa_score_has_no_numeric_latency_field` (dataclass introspection) and `test_write_report_serializes_schema_version_and_never_a_numeric_latency` (JSON payload has no `*decision_latenc*` numeric key) — both in Task 5. |

---

## Self-review

**Spec coverage:** every numbered file in the spec's §3 file list (7 source, 4 test files) has a
corresponding task with a pinned signature (Tasks 2–6). §4's typing procedure is reproduced
verbatim as executable steps inside Task 2, not paraphrased. §7's latency-safety mechanism is
implemented as the exact three guarantees the spec named (no numeric field, literal string,
introspection test) and each has its own test. §8's three redundant human/candidate distinctions
are all present: filesystem convention (`*.librosa-candidate.txt`/`*.librosa-beats.txt` in Task 4
and 6), schema marker (Task 5's `advisory_only`/`production_candidate`/`kind`), and code-path
isolation (only `score_against_human_reference()` treats a file as ground truth, and it's only
ever called with an operator-supplied path in `cli.py`). §10's full test matrix (analysis, export,
report, cli) plus both "additional required regression" bullets (import-graph check, core-
dependency exclusion) are covered — the former in Task 7, the latter in Task 1. §11's acceptance
criteria map onto the Final Validation section point-for-point. §13's implementation sequence is
followed task-for-task, including making the optional documentation step explicitly last and
explicitly non-blocking.

**Placeholders:** none. Every test in this plan is a complete, runnable test body, not a stub or a
"TODO: write assertions" marker. Every command is the literal command to run, not a description of
one.

**Inconsistent types/signatures checked:**
- `LibrosaAnalysis.onset_envelope: OnsetEnvelope` (full resolution, Task 3) vs.
  `LibrosaBenchmarkReport.onset_summary: dict[str, float | int]` (aggregated only, Task 5) — the
  full envelope is deliberately never passed into `build_report()`; only `analysis.onset_summary`
  is. Verified consistent across Task 3's production interface and Task 5's consumption of it.
- `cli.py`'s `--envelope-npy` flag (Task 6) calls `analysis.write_onset_envelope`, which needs the
  *raw* `OnsetEnvelope`, not the `OnsetSummary` — confirmed `LibrosaAnalysis` carries both fields
  so the CLI has access to the raw envelope without recomputing it, and `LibrosaBenchmarkReport`
  gained an `onset_envelope_path: str | None` field (set by `cli.py`, not by `build_report`'s
  defaults) specifically so the report can point at the sidecar file when one was written — this
  field was implicit in the spec's example JSON sketch but not spelled out in its dataclass
  signature; it's made explicit and consistent here across Tasks 3, 5, and 6.
- `score_against_human_reference`'s `tolerance_seconds` (Task 5) vs. `cli.py`'s `--tolerance-ms`
  (Task 6) — confirmed the CLI divides by 1000 before passing through, matching the existing M2C
  CLI's own `--tolerance-ms` → `tolerance_seconds` conversion pattern in `evaluation/cli.py`.
- `LibrosaScore.reference_path: str` (not `Path`) — matches how `LibrosaBenchmarkReport` and
  `BakeoffReport` are both serialized via `dataclasses.asdict()` + `json.dumps`; a raw `Path` field
  would fail `json.dumps` without a custom encoder, so it's stored pre-stringified, consistent with
  how the rest of the dataclass is built to serialize directly.

**Unnecessary scope avoided:** no task modifies `bakeoff.py`, `candidates.py`,
`evaluation/cli.py`, `evaluation/report.py`, `replay_source.py`, `config.py`, `engine.py`,
`models.py`, `detectors/`, live-capture code, or `.github/workflows/ci.yml` — confirmed by the
explicit zero-diff checks in Final Validation and the Explicit Constraint Verification table.
Task 8 (documentation) is marked optional and ordered last specifically so it can be dropped
without touching the acceptance path. No task introduces a dataset-manifest/batch mode, CI
workflow change, or candidate-tuning code path — consistent with the spec's non-goals and its own
§12 scope notes.

**Missing validation checked:** the seven numbered items in "REQUIRED FINAL VALIDATION" each have
a dedicated, numbered subsection with an exact command and an expected result — including the
two-tier treatment of item 7 (scriptable synthetic smoke test vs. optional manual real-artifact
smoke test), since a real M2C artifact is private local evidence that cannot be committed or
guaranteed present at plan-writing time. The five "explicitly verify" bullets each have a row in
the Explicit Constraint Verification checklist naming the exact test or diff command that proves
them, rather than being asserted as true without evidence.

No production code has been written or modified as part of producing this plan.
