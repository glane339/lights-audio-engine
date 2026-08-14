"""Deterministic chronological one-to-one event matching."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class EventMatch:
    reference_index: int
    detection_index: int


@dataclass(frozen=True, slots=True)
class MatchResult:
    matches: tuple[EventMatch, ...]
    false_positive_detection_indices: tuple[int, ...]
    short_double_detection_indices: tuple[int, ...]
    missed_reference_indices: tuple[int, ...]


def match_events(
    references: tuple[float, ...],
    detections: tuple[float, ...],
    *,
    tolerance_seconds: float = 0.05,
) -> MatchResult:
    """Match detections chronologically to the earliest available reference in tolerance."""

    if not isfinite(tolerance_seconds) or tolerance_seconds < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    matches: list[EventMatch] = []
    matched_references: set[int] = set()
    matched_detections: set[int] = set()
    for detection_index, detection in enumerate(detections):
        for reference_index, reference in enumerate(references):
            if reference_index in matched_references:
                continue
            if abs(detection - reference) <= tolerance_seconds + 1e-12:
                matches.append(EventMatch(reference_index, detection_index))
                matched_references.add(reference_index)
                matched_detections.add(detection_index)
                break
    false_positives = tuple(
        index for index in range(len(detections)) if index not in matched_detections
    )
    matched_reference_times = tuple(references[match.reference_index] for match in matches)
    short_doubles = tuple(
        index
        for index in false_positives
        if any(
            abs(detections[index] - reference) <= tolerance_seconds + 1e-12
            for reference in matched_reference_times
        )
    )
    missed = tuple(index for index in range(len(references)) if index not in matched_references)
    return MatchResult(tuple(matches), false_positives, short_doubles, missed)
