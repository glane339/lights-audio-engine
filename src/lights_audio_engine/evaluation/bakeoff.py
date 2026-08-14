"""Deterministic multi-track, multi-candidate M2C orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from lights_audio_engine.evaluation.artifact import read_artifact
from lights_audio_engine.evaluation.candidates import Candidate, create_candidate
from lights_audio_engine.evaluation.matching import match_events
from lights_audio_engine.evaluation.reference import parse_reference
from lights_audio_engine.evaluation.replay_source import ReplayAudioSource
from lights_audio_engine.evaluation.runner import EmittedDetection, run_candidate
from lights_audio_engine.evaluation.scoring import TrackMetrics, aggregate_metrics, score_events

NO_PRODUCTION_CANDIDATE = "NO PRODUCTION CANDIDATE YET"


class DatasetSplit(StrEnum):
    """Predeclared tuning status for an evaluation track."""

    DEVELOPMENT = "development"
    HOLDOUT = "holdout"


@dataclass(frozen=True, slots=True)
class TrackSpec:
    label: str
    artifact_path: Path
    reference_path: Path
    split: DatasetSplit
    segment_index: int | None = None


@dataclass(frozen=True, slots=True)
class TrackEvaluation:
    label: str
    split: DatasetSplit
    segment_index: int
    detections: tuple[EmittedDetection, ...]
    metrics: TrackMetrics


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    candidate_name: str
    block_size_frames: int
    tracks: tuple[TrackEvaluation, ...]
    development_metrics: TrackMetrics
    holdout_metrics: TrackMetrics
    quality_gates_passed: bool


@dataclass(frozen=True, slots=True)
class BakeoffReport:
    candidate_reports: tuple[CandidateEvaluation, ...]
    ranking: tuple[str, ...]
    recommendation: str


def _quality_gates(candidate: CandidateEvaluation) -> bool:
    holdout_tracks = tuple(
        track for track in candidate.tracks if track.split is DatasetSplit.HOLDOUT
    )
    latency = candidate.holdout_metrics.p95_decision_latency_seconds
    return bool(holdout_tracks) and all(
        (
            candidate.block_size_frames == 240,
            candidate.holdout_metrics.precision >= 0.90,
            candidate.holdout_metrics.recall >= 0.90,
            all(track.metrics.recall >= 0.85 for track in holdout_tracks),
            candidate.holdout_metrics.longest_missed_run <= 2,
            latency is not None and latency <= 0.015,
            candidate.holdout_metrics.short_doubles == 0,
        )
    )


def run_bakeoff(
    tracks: tuple[TrackSpec, ...],
    *,
    candidate_factories: Mapping[str, Callable[[], Candidate]] | None = None,
    block_sizes: tuple[int, ...] = (240, 480, 960),
    tolerance_seconds: float = 0.05,
) -> BakeoffReport:
    """Evaluate every hardcoded candidate against identical exact track samples."""

    if not tracks:
        raise ValueError("bake-off requires at least one track")
    factories: Mapping[str, Callable[[], Candidate]] = candidate_factories or {
        name: (lambda candidate_name=name: create_candidate(candidate_name))
        for name in ("baseline", "broadband", "multiband")
    }
    evaluations: list[CandidateEvaluation] = []
    for candidate_name, factory in factories.items():
        for block_size in block_sizes:
            track_evaluations: list[TrackEvaluation] = []
            for track in tracks:
                artifact = read_artifact(track.artifact_path)
                references = parse_reference(track.reference_path)
                candidate = factory()
                run = run_candidate(
                    candidate,
                    ReplayAudioSource(
                        artifact,
                        block_size_frames=block_size,
                        segment_index=track.segment_index,
                    ).stream(),
                )
                detection_times = tuple(item.event.timestamp_seconds for item in run.detections)
                emission_times = tuple(item.emitted_stream_time_seconds for item in run.detections)
                matching = match_events(
                    references, detection_times, tolerance_seconds=tolerance_seconds
                )
                metrics = score_events(
                    references,
                    detection_times,
                    emission_times,
                    matching,
                    processing_seconds=run.processing_seconds,
                )
                track_evaluations.append(
                    TrackEvaluation(
                        track.label,
                        track.split,
                        0 if track.segment_index is None else track.segment_index,
                        run.detections,
                        metrics,
                    )
                )
            development_metrics = aggregate_metrics(
                tuple(
                    track.metrics
                    for track in track_evaluations
                    if track.split is DatasetSplit.DEVELOPMENT
                )
            )
            holdout_metrics = aggregate_metrics(
                tuple(
                    track.metrics
                    for track in track_evaluations
                    if track.split is DatasetSplit.HOLDOUT
                )
            )
            partial = CandidateEvaluation(
                candidate_name,
                block_size,
                tuple(track_evaluations),
                development_metrics,
                holdout_metrics,
                False,
            )
            evaluations.append(
                CandidateEvaluation(
                    partial.candidate_name,
                    partial.block_size_frames,
                    partial.tracks,
                    partial.development_metrics,
                    partial.holdout_metrics,
                    _quality_gates(partial),
                )
            )
    five_ms = [item for item in evaluations if item.block_size_frames == 240]
    ranked = sorted(
        five_ms,
        key=lambda item: (
            item.holdout_metrics.f1,
            item.holdout_metrics.recall,
            item.holdout_metrics.precision,
        ),
        reverse=True,
    )
    ranking = tuple(item.candidate_name for item in ranked)
    winner = next((item.candidate_name for item in ranked if item.quality_gates_passed), None)
    return BakeoffReport(tuple(evaluations), ranking, winner or NO_PRODUCTION_CANDIDATE)
