from __future__ import annotations

from pathlib import Path

import pytest


def test_reference_parser_accepts_point_labels_and_zero_events(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.reference import parse_reference

    path = tmp_path / "beats.txt"
    path.write_text("0.100\tbeat\n0.500\tbeat 2\n", encoding="utf-8")
    assert parse_reference(path) == (0.1, 0.5)

    path.write_text("# no beats in this segment\n", encoding="utf-8")
    assert parse_reference(path) == ()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("nope\tbeat\n", "timestamp"),
        ("nan\tbeat\n", "finite"),
        ("-0.1\tbeat\n", "non-negative"),
        ("0.2\tbeat\n0.1\tbeat\n", "strictly increasing"),
        ("0.1\n", "tab-separated"),
    ],
)
def test_reference_parser_rejects_bad_labels(tmp_path: Path, text: str, message: str) -> None:
    from lights_audio_engine.evaluation.reference import ReferenceFormatError, parse_reference

    path = tmp_path / "beats.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ReferenceFormatError, match=message):
        parse_reference(path)


def test_matching_is_one_to_one_and_classifies_short_double() -> None:
    from lights_audio_engine.evaluation.matching import match_events

    result = match_events((1.0, 2.0), (0.96, 1.02, 2.05, 3.0), tolerance_seconds=0.05)

    assert [(match.reference_index, match.detection_index) for match in result.matches] == [
        (0, 0),
        (1, 2),
    ]
    assert result.short_double_detection_indices == (1,)
    assert result.false_positive_detection_indices == (1, 3)
    assert result.missed_reference_indices == ()


def test_matching_boundary_is_inclusive_and_ambiguity_is_chronological() -> None:
    from lights_audio_engine.evaluation.matching import match_events

    result = match_events((1.0, 1.08), (1.05,), tolerance_seconds=0.05)
    assert [(match.reference_index, match.detection_index) for match in result.matches] == [(0, 0)]
    assert result.missed_reference_indices == (1,)


def test_scoring_uses_absolute_error_and_measures_emission_latency() -> None:
    from lights_audio_engine.evaluation.matching import match_events
    from lights_audio_engine.evaluation.scoring import score_events

    references = (1.0, 2.0, 3.0, 4.0)
    detections = (0.98, 1.02, 2.04, 5.0)
    emissions = (1.01, 1.03, 2.10, 5.01)
    matching = match_events(references, detections, tolerance_seconds=0.05)
    metrics = score_events(references, detections, emissions, matching, processing_seconds=0.125)

    assert metrics.true_positives == 2
    assert metrics.false_positives == 2
    assert metrics.short_doubles == 1
    assert metrics.false_negatives == 2
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.signed_timing_errors_seconds == pytest.approx((-0.02, 0.04))
    assert metrics.absolute_timing_errors_seconds == pytest.approx((0.02, 0.04))
    assert metrics.p95_absolute_timing_error_seconds == pytest.approx(0.039)
    assert metrics.longest_missed_run == 2
    assert metrics.decision_latencies_seconds == pytest.approx((0.03, 0.01, 0.06, 0.01))
    assert metrics.processing_seconds == 0.125
