"""Per-track detector metrics, including timing and decision latency."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import numpy as np

from lights_audio_engine.evaluation.matching import MatchResult


@dataclass(frozen=True, slots=True)
class TrackMetrics:
    true_positives: int
    false_positives: int
    short_doubles: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    signed_timing_errors_seconds: tuple[float, ...]
    median_signed_timing_bias_seconds: float | None
    absolute_timing_errors_seconds: tuple[float, ...]
    median_absolute_timing_error_seconds: float | None
    p95_absolute_timing_error_seconds: float | None
    longest_missed_run: int
    processing_seconds: float
    decision_latencies_seconds: tuple[float, ...]
    median_decision_latency_seconds: float | None
    p95_decision_latency_seconds: float | None


def _percentile(values: tuple[float, ...], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _longest_run(indices: tuple[int, ...]) -> int:
    longest = current = 0
    previous: int | None = None
    for index in indices:
        current = current + 1 if previous is not None and index == previous + 1 else 1
        longest = max(longest, current)
        previous = index
    return longest


def score_events(
    references: tuple[float, ...],
    detections: tuple[float, ...],
    emission_times: tuple[float, ...],
    matching: MatchResult,
    *,
    processing_seconds: float,
) -> TrackMetrics:
    """Compute accuracy, gap, runtime, and emission-latency observations."""

    if len(detections) != len(emission_times):
        raise ValueError("each detection must have one emission time")
    signed = tuple(
        detections[match.detection_index] - references[match.reference_index]
        for match in matching.matches
    )
    absolute = tuple(abs(error) for error in signed)
    latencies = tuple(
        emission - event for event, emission in zip(detections, emission_times, strict=True)
    )
    if any(latency < -1e-12 for latency in latencies):
        raise ValueError("emission time cannot precede an event timestamp")
    true_positives = len(matching.matches)
    false_positives = len(matching.false_positive_detection_indices)
    false_negatives = len(matching.missed_reference_indices)
    precision = true_positives / (true_positives + false_positives) if detections else 0.0
    recall = true_positives / (true_positives + false_negatives) if references else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return TrackMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        short_doubles=len(matching.short_double_detection_indices),
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        signed_timing_errors_seconds=signed,
        median_signed_timing_bias_seconds=median(signed) if signed else None,
        absolute_timing_errors_seconds=absolute,
        median_absolute_timing_error_seconds=median(absolute) if absolute else None,
        p95_absolute_timing_error_seconds=_percentile(absolute, 95.0),
        longest_missed_run=_longest_run(matching.missed_reference_indices),
        processing_seconds=processing_seconds,
        decision_latencies_seconds=latencies,
        median_decision_latency_seconds=median(latencies) if latencies else None,
        p95_decision_latency_seconds=_percentile(latencies, 95.0),
    )


def aggregate_metrics(metrics: tuple[TrackMetrics, ...]) -> TrackMetrics:
    """Micro-average count metrics while retaining pooled timing observations."""

    true_positives = sum(item.true_positives for item in metrics)
    false_positives = sum(item.false_positives for item in metrics)
    false_negatives = sum(item.false_negatives for item in metrics)
    signed = tuple(value for item in metrics for value in item.signed_timing_errors_seconds)
    absolute = tuple(value for item in metrics for value in item.absolute_timing_errors_seconds)
    latencies = tuple(value for item in metrics for value in item.decision_latencies_seconds)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return TrackMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        short_doubles=sum(item.short_doubles for item in metrics),
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        signed_timing_errors_seconds=signed,
        median_signed_timing_bias_seconds=median(signed) if signed else None,
        absolute_timing_errors_seconds=absolute,
        median_absolute_timing_error_seconds=median(absolute) if absolute else None,
        p95_absolute_timing_error_seconds=_percentile(absolute, 95.0),
        longest_missed_run=max((item.longest_missed_run for item in metrics), default=0),
        processing_seconds=sum(item.processing_seconds for item in metrics),
        decision_latencies_seconds=latencies,
        median_decision_latency_seconds=median(latencies) if latencies else None,
        p95_decision_latency_seconds=_percentile(latencies, 95.0),
    )
