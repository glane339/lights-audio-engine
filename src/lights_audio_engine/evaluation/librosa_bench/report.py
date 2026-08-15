"""Advisory-only reports for offline Librosa benchmark results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from lights_audio_engine.evaluation.librosa_bench.analysis import LibrosaAnalysis
from lights_audio_engine.evaluation.matching import match_events
from lights_audio_engine.evaluation.reference import parse_reference
from lights_audio_engine.evaluation.scoring import score_events

DECISION_LATENCY_NOT_APPLICABLE = (
    "not applicable — Librosa runs fully offline over the whole segment; "
    "no causal emission time exists"
)


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
    median_signed_timing_bias_seconds: float | None
    median_absolute_timing_error_seconds: float | None
    p95_absolute_timing_error_seconds: float | None
    decision_latency: str = DECISION_LATENCY_NOT_APPLICABLE


def score_against_human_reference(
    reference_path: Path,
    beat_times_seconds: tuple[float, ...],
    *,
    tolerance_seconds: float = 0.05,
) -> LibrosaScore:
    """Score offline estimates against an operator-supplied human reference."""

    references = parse_reference(reference_path)
    matching = match_events(
        references,
        beat_times_seconds,
        tolerance_seconds=tolerance_seconds,
    )
    metrics = score_events(
        references,
        beat_times_seconds,
        beat_times_seconds,
        matching,
        processing_seconds=0.0,
    )
    return LibrosaScore(
        reference_path=str(reference_path),
        tolerance_seconds=tolerance_seconds,
        true_positives=metrics.true_positives,
        false_positives=metrics.false_positives,
        false_negatives=metrics.false_negatives,
        short_doubles=metrics.short_doubles,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        median_signed_timing_bias_seconds=metrics.median_signed_timing_bias_seconds,
        median_absolute_timing_error_seconds=metrics.median_absolute_timing_error_seconds,
        p95_absolute_timing_error_seconds=metrics.p95_absolute_timing_error_seconds,
    )


@dataclass(frozen=True, slots=True)
class LibrosaBenchmarkReport:
    kind: str
    advisory_only: bool
    production_candidate: bool
    label: str
    segment_index: int
    sample_rate_hz: int
    tempo_bpm: float
    beat_times_seconds: tuple[float, ...]
    beat_times_source: str
    onset_summary: dict[str, float | int]
    onset_envelope_path: str | None
    human_comparison: LibrosaScore | None
    decision_latency: str = DECISION_LATENCY_NOT_APPLICABLE


def build_report(
    analysis: LibrosaAnalysis,
    human_comparison: LibrosaScore | None,
    *,
    onset_envelope_path: Path | None = None,
) -> LibrosaBenchmarkReport:
    """Build a self-labeled offline benchmark report from one analysis segment."""

    return LibrosaBenchmarkReport(
        kind="librosa_offline_benchmark",
        advisory_only=True,
        production_candidate=False,
        label=analysis.label,
        segment_index=analysis.segment_index,
        sample_rate_hz=analysis.sample_rate_hz,
        tempo_bpm=analysis.tempo_bpm,
        beat_times_seconds=analysis.beat_times_seconds,
        beat_times_source="librosa_offline_benchmark",
        onset_summary=asdict(analysis.onset_summary),
        onset_envelope_path=str(onset_envelope_path) if onset_envelope_path is not None else None,
        human_comparison=human_comparison,
    )


def write_report(path: Path, report: LibrosaBenchmarkReport) -> None:
    """Write a versioned JSON report without causal latency metrics."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, **asdict(report)}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
