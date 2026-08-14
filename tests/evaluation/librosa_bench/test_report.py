from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest


def _analysis(label: str = "track", segment_index: int = 0, sample_rate_hz: int = 48_000):
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

    field_names = {field.name: field.type for field in dataclasses.fields(LibrosaScore)}
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

    bakeoff_fields = {field.name for field in dataclasses.fields(BakeoffReport)}
    librosa_fields = {field.name for field in dataclasses.fields(LibrosaBenchmarkReport)}

    assert not bakeoff_fields & librosa_fields
